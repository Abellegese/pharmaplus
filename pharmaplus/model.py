import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pharmaplus.helpers import gaussian_logpdf

class NodeEncoder(nn.Module):
  def __init__(self, x_dim: int, d: int, n_types: int = 256, use_score=False):
    super().__init__()
    self.use_score = use_score
    self.type_emb = nn.Embedding(n_types, d)
    self.x_mlp = nn.Sequential(nn.Linear(x_dim, d), nn.GELU(), nn.Linear(d, d))
    self.s_mlp = (
      nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d)) if use_score else None
    )
    self.ln = nn.LayerNorm(d)

  def forward(self, x, typ, score=None):
    # x: [...,x_dim], typ: [...]
    typ = typ.clamp(0, self.type_emb.num_embeddings - 1)
    h = self.x_mlp(x) + self.type_emb(typ)
    if self.use_score and (score is not None):
      h = h + self.s_mlp(score.unsqueeze(-1))
    return self.ln(h)


class PharmMatchNetFast(nn.Module):
  """
  Batched version:
    - StageA: logits -> logW (+dustbin), sharp/entropy, edge likelihood (sampled), pooled retrieval embeddings
    - StageB: pose term + refine term

  Key knobs:
    - dustbin: extra "no-match" column
    - edge_scale: edge_term = tanh(edge_raw / edge_scale)  (must match inference!)
  """

  def __init__(
    self,
    d=128,
    retr_d=128,
    topk=4,
    edge_pairs=16,
    dustbin=True,
    edge_scale=5.0,
  ):
    super().__init__()
    self.d = int(d)
    self.retr_d = int(retr_d)
    self.topk = int(topk)
    self.edge_pairs = int(edge_pairs)
    self.dustbin = bool(dustbin)
    self.edge_scale = float(edge_scale)

    self.l_enc = NodeEncoder(8, d, n_types=256, use_score=False)
    self.p_enc = NodeEncoder(192, d, n_types=256, use_score=True)
    self.proj_l = nn.Linear(d, d, bias=False)
    self.proj_p = nn.Linear(d, d, bias=False)

    if self.dustbin:
      self.dustbin_vec = nn.Parameter(torch.randn(d) * 0.02)

    self.log_beta = nn.Parameter(torch.tensor(math.log(0.5)))

    self.p_logsig_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    self.head_inv = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 1))
    self.head_pose = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 1))

    ref_in = (d + d) + 3 + 1
    self.refine_mlp = nn.Sequential(
      nn.Linear(ref_in, 128),
      nn.GELU(),
      nn.Linear(128, 64),
      nn.GELU(),
      nn.Linear(64, 1),
    )

    self.retr_l = nn.Linear(d, retr_d, bias=False)
    self.retr_p = nn.Linear(d, retr_d, bias=False)

  def encode_batch(self, l_x, l_typ, p_x, p_typ, p_score):
    # l_x [B,L,8], p_x [B,P,192]
    hL = self.proj_l(self.l_enc(l_x, l_typ))  # [B,L,d]
    hP = self.proj_p(self.p_enc(p_x, p_typ, p_score))  # [B,P,d]
    return hL, hP

  def feature_logits_batch(self, hL, hP, l_mask, p_mask):
    # logits [B,L,P]
    logits = torch.einsum("bld,bpd->blp", hL, hP) / math.sqrt(hL.shape[-1])
    neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~p_mask[:, None, :], neg)
    logits = logits.masked_fill(~l_mask[:, :, None], neg)
    return logits

  def assignment_logW_batch(self, logits_feat, hL, l_mask, p_mask):
    neg = torch.finfo(logits_feat.dtype).min
    if not self.dustbin:
      logits = logits_feat.masked_fill(~p_mask[:, None, :], neg)
      logits = logits.masked_fill(~l_mask[:, :, None], neg)
      return F.log_softmax(logits, dim=-1)  # [B,L,P]

    dust = torch.einsum("bld,d->bl", hL, self.dustbin_vec) / math.sqrt(
      hL.shape[-1]
    )  # [B,L]
    logits_real = logits_feat.masked_fill(~p_mask[:, None, :], neg)
    logits = torch.cat([logits_real, dust[:, :, None]], dim=-1)  # [B,L,P+1]
    logits = logits.masked_fill(~l_mask[:, :, None], neg)
    return F.log_softmax(logits, dim=-1)

  def pocket_sigma_nodes_batch(self, hP, p_rad):
    # hP [B,P,d], p_rad [B,P]
    log_sig = self.p_logsig_head(hP).squeeze(-1)  # [B,P]
    sig = F.softplus(log_sig) + 1e-3
    sig = sig + p_rad.clamp(min=0.0)
    return sig

  def invariant_feature_terms_batch(self, logW, l_mask, p_mask):
    # logW [B,L,P(+1)]
    W = torch.exp(logW)
    denom = l_mask.float().sum(dim=1).clamp(min=1.0)  # [B]

    if self.dustbin and (W.shape[-1] == p_mask.shape[1] + 1):
      W_real = W[:, :, :-1] * p_mask[:, None, :].float()
      W_dust = W[:, :, -1:]
      Wm = torch.cat([W_real, W_dust], dim=-1)
    else:
      Wm = W * p_mask[:, None, :].float()

    sharp = (Wm.max(dim=-1).values * l_mask.float()).sum(dim=1) / denom
    ent = (-(Wm * logW).sum(dim=-1) * l_mask.float()).sum(dim=1) / denom
    neg_ent = -ent
    return sharp, neg_ent

  def pooled_embeddings_batch(self, hL, hP, l_mask, p_mask):
    lm = l_mask.float().unsqueeze(-1)
    pm = p_mask.float().unsqueeze(-1)
    l_g = (hL * lm).sum(dim=1) / lm.sum(dim=1).clamp(min=1.0)
    p_g = (hP * pm).sum(dim=1) / pm.sum(dim=1).clamp(min=1.0)
    l_z = F.normalize(self.retr_l(l_g), dim=-1)
    p_z = F.normalize(self.retr_p(p_g), dim=-1)
    return l_z, p_z

  def edge_likelihood_batch(
    self,
    l_pos,
    p_pos,
    logits_feat,
    logW_real,
    l_mask,
    p_mask,
    p_sigma,
    topk: int,
    edge_pairs: int,
  ):
    """
    Vectorized sampled edge-likelihood across batch.

    Shapes:
      l_pos [B,L,3], p_pos [B,P,3]
      logits_feat [B,L,P], logW_real [B,L,P]
      p_sigma [B,P]
    Returns:
      edge_raw [B]
    """
    B, L, _ = l_pos.shape
    P = p_pos.shape[1]
    K = min(int(topk), P)
    M = int(edge_pairs)

    a = torch.randint(0, L, (B, M), device=l_pos.device)
    b = torch.randint(0, L, (B, M), device=l_pos.device)

    bi = torch.arange(B, device=l_pos.device)[:, None]  # [B,1]
    valid_pair = l_mask[bi, a] & l_mask[bi, b] & (a != b)  # [B,M]

    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      la = l_pos[bi, a].float()
      lb = l_pos[bi, b].float()
      d = torch.norm(la - lb, dim=-1)  # [B,M]
    d = d.to(logW_real.dtype)[:, :, None, None]  # [B,M,1,1]

    neg = torch.finfo(logits_feat.dtype).min
    lf = logits_feat.masked_fill(~p_mask[:, None, :], neg)  # [B,L,P]
    topv, topj = torch.topk(lf, k=K, dim=-1)  # [B,L,K]

    # gather p positions for each ligand node's topk: [B,L,K,3]
    p_exp = p_pos[:, None, :, :].expand(B, L, P, 3)
    topj3 = topj[..., None].expand(B, L, K, 3)
    p_sel = torch.gather(p_exp, dim=2, index=topj3)

    # gather sig nodes: [B,L,K]
    sig_exp = p_sigma[:, None, :].expand(B, L, P)
    sig_sel = torch.gather(sig_exp, dim=2, index=topj)

    # gather logW top: [B,L,K]
    logW_exp = logW_real
    logW_top = torch.gather(logW_exp, dim=2, index=topj)

    #NOTE: select per pair (a,b): [B,M,K,3] etc
    pa = p_sel[bi, a]  # [B,M,K,3]
    pb = p_sel[bi, b]  # [B,M,K,3]
    siga = sig_sel[bi, a]  # [B,M,K]
    sigb = sig_sel[bi, b]  # [B,M,K]
    lwa = logW_top[bi, a]  # [B,M,K]
    lwb = logW_top[bi, b]  # [B,M,K]

    #NOTE: mu distances: [B,M,K,K]
    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      mu = torch.norm(pa[:, :, :, None, :] - pb[:, :, None, :, :], dim=-1).float()
    mu = mu.to(logW_real.dtype)

    sigma = torch.sqrt(
      (siga[:, :, :, None] ** 2) + (sigb[:, :, None, :] ** 2) + 1e-6
    ).to(logW_real.dtype)
    logw_pair = lwa[:, :, :, None] + lwb[:, :, None, :]  # [B,M,K,K]

    logp = gaussian_logpdf(d, mu, sigma)
    logmix = torch.logsumexp(logw_pair + logp, dim=(-1, -2))  # [B,M]

    #NOTE: average only valid pairs
    vp = valid_pair.float()
    denom = vp.sum(dim=1).clamp(min=1.0)
    edge_raw = (logmix * vp).sum(dim=1) / denom  # [B]
    return edge_raw

  def pose_term_batch(self, l_pos, p_pos, logW, l_mask, p_mask):
    beta = torch.exp(self.log_beta).clamp(0.01, 10.0)

    #NOTE: drop dustbin for geometry
    if self.dustbin and (logW.shape[-1] == p_pos.shape[1] + 1):
      logW_real = logW[:, :, :-1]
    else:
      logW_real = logW

    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      d2 = torch.cdist(l_pos.float(), p_pos.float()) ** 2  # [B,L,P]
    d2 = d2.to(logW_real.dtype)

    W = torch.exp(logW_real) * p_mask[:, None, :].float()
    Z = W.sum(dim=2, keepdim=True).clamp(min=1e-6)
    Wn = W / Z

    exp_d2 = (Wn * d2).sum(dim=2)  # [B,L]
    denom = l_mask.float().sum(dim=1).clamp(min=1.0)
    return ((-beta * exp_d2) * l_mask.float()).sum(dim=1) / denom  # [B]

  def refine_term_batch(self, hL, hP, l_pos, p_pos, logW, l_mask, p_mask):
    if self.dustbin and (logW.shape[-1] == p_pos.shape[1] + 1):
      logW_real = logW[:, :, :-1]
    else:
      logW_real = logW

    W = torch.exp(logW_real) * p_mask[:, None, :].float()
    Z = W.sum(dim=2, keepdim=True).clamp(min=1e-6)
    Wn = W / Z

    p_pos_bar = torch.einsum("blp,bpc->blc", Wn, p_pos)  # [B,L,3]
    hP_bar = torch.einsum("blp,bpd->bld", Wn, hP)  # [B,L,d]
    delta = l_pos - p_pos_bar
    dn = torch.norm(delta, dim=-1, keepdim=True)

    feat = torch.cat([hL, hP_bar, delta, dn], dim=-1)  # [B,L,2d+4]
    per = self.refine_mlp(feat).squeeze(-1)  # [B,L]

    denom = l_mask.float().sum(dim=1).clamp(min=1.0)
    return (per * l_mask.float()).sum(dim=1) / denom  # [B]

  def precompute_stageA_batch(
    self,
    l_pos,
    l_x,
    l_typ,
    l_mask,
    p_pos,
    p_x,
    p_typ,
    p_score,
    p_rad,
    p_mask,
    topk: Optional[int] = None,
    edge_pairs: Optional[int] = None,
  ) -> Dict[str, torch.Tensor]:
    topk = self.topk if topk is None else int(topk)
    edge_pairs = self.edge_pairs if edge_pairs is None else int(edge_pairs)

    hL, hP = self.encode_batch(l_x, l_typ, p_x, p_typ, p_score)
    logits_feat = self.feature_logits_batch(hL, hP, l_mask, p_mask)
    logW = self.assignment_logW_batch(logits_feat, hL, l_mask, p_mask)

    sharp, neg_ent = self.invariant_feature_terms_batch(logW, l_mask, p_mask)
    p_sigma = self.pocket_sigma_nodes_batch(hP, p_rad)

    #NOTE: edge uses real pocket columns only 
    if self.dustbin and (logW.shape[-1] == p_pos.shape[1] + 1):
      logW_real = logW[:, :, :-1]
    else:
      logW_real = logW

    edge_raw = self.edge_likelihood_batch(
      l_pos=l_pos,
      p_pos=p_pos,
      logits_feat=logits_feat,
      logW_real=logW_real,
      l_mask=l_mask,
      p_mask=p_mask,
      p_sigma=p_sigma,
      topk=topk,
      edge_pairs=edge_pairs,
    )

    edge = torch.tanh(edge_raw / float(self.edge_scale))

    inv_vec = torch.stack([sharp, neg_ent, edge], dim=1)  # [B,3]
    score_inv_geom = self.head_inv(inv_vec).squeeze(-1)  # [B]

    l_z, p_z = self.pooled_embeddings_batch(hL, hP, l_mask, p_mask)
    retr_sim_diag = (l_z * p_z).sum(dim=-1)  # [B]

    return {
      "hL": hL,
      "hP": hP,
      "logW": logW,
      "score_inv_geom": score_inv_geom,
      "sharp": sharp,
      "neg_ent": neg_ent,
      "edge": edge,
      "edge_raw": edge_raw,
      "l_z": l_z,
      "p_z": p_z,
      "retr_sim": retr_sim_diag,
    }

  def score_pose_from_stageA_batch(
    self,
    stageA: Dict[str, torch.Tensor],
    l_pos,
    l_mask,
    p_pos,
    p_mask,
  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pose_t = self.pose_term_batch(l_pos, p_pos, stageA["logW"], l_mask, p_mask)
    ref_t = self.refine_term_batch(
      stageA["hL"], stageA["hP"], l_pos, p_pos, stageA["logW"], l_mask, p_mask
    )
    pose_vec = torch.stack([pose_t, ref_t], dim=1)  # [B,2]
    score_pose = stageA["score_inv_geom"] + self.head_pose(pose_vec).squeeze(-1)
    dbg = {"pose_term": pose_t.detach(), "refine_term": ref_t.detach()}
    return score_pose, dbg


