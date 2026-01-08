import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pharmaplus.helpers import gaussian_logpdf


class NodeEncoder(nn.Module):
  def __init__(self, x_dim: int, d: int, n_types: int = 256, use_score: bool = False):
    super().__init__()
    self.use_score = bool(use_score)
    self.type_emb = nn.Embedding(n_types, d)
    self.x_mlp = nn.Sequential(nn.Linear(x_dim, d), nn.GELU(), nn.Linear(d, d))
    self.s_mlp = (
      nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
      if self.use_score else None
    )
    self.ln = nn.LayerNorm(d)

  def forward(self, x, typ, score=None):
    typ = typ.clamp(0, self.type_emb.num_embeddings - 1)
    h = self.x_mlp(x) + self.type_emb(typ)
    if self.use_score and (score is not None):
      h = h + self.s_mlp(score.unsqueeze(-1))
    return self.ln(h)


class PharmMatchNetFast(nn.Module):
  """
  Batched model with:
    - StageA: correspondence logits -> logW, invariant stats, fast edge likelihood, retrieval embeddings
    - Optional StageB: pose-aware scoring (can be disabled for ultra-fast inference)

  Fixes vs old version:
    (1) refine term is SE(3)-invariant (no raw xyz delta in MLP)
    (2) pose terms use sparse top-K correspondences + explicit variance term
    (3) edge mean/uncertainty are learnable calibrations (mu_a/mu_b, sigma combiner, tau)
  """

  def __init__(
    self,
    d: int = 128,
    retr_d: int = 128,
    topk: int = 4,
    edge_pairs: int = 16,
    dustbin: bool = True,
    edge_scale: float = 5.0,
    # pose options
    pose_topk: int = 8,        # sparse K for pose terms (separate from edge topk)
    pose_use_var: bool = True, # include correspondence variance in refine feature
  ):
    super().__init__()
    self.d = int(d)
    self.retr_d = int(retr_d)
    self.topk = int(topk)
    self.edge_pairs = int(edge_pairs)
    self.dustbin = bool(dustbin)
    self.edge_scale = float(edge_scale)

    self.pose_topk = int(pose_topk)
    self.pose_use_var = bool(pose_use_var)

    self.l_enc = NodeEncoder(8, d, n_types=256, use_score=False)
    self.p_enc = NodeEncoder(192, d, n_types=256, use_score=True)
    self.proj_l = nn.Linear(d, d, bias=False)
    self.proj_p = nn.Linear(d, d, bias=False)

    if self.dustbin:
      self.dustbin_vec = nn.Parameter(torch.randn(d) * 0.02)

    # pose temperature (beta)
    self.log_beta = nn.Parameter(torch.tensor(math.log(0.5)))

    # per-pocket-node sigma head (node uncertainty prior)
    self.p_logsig_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # invariant head (sharp, neg_ent, edge)
    self.head_inv = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 1))

    # pose head (pose_energy, refine_energy)
    self.head_pose = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 1))

    # --- NEW: SE(3)-invariant refine head ---
    # Inputs per ligand node:
    #   - hL (d)
    #   - hP_bar (d)
    #   - ||l - p_bar|| (1)
    #   - (optional) sqrt(var_p) (1)  where var_p = sum_k w ||p_k - p_bar||^2
    ref_in = (d + d) + 1 + (1 if self.pose_use_var else 0)
    self.refine_mlp = nn.Sequential(
      nn.Linear(ref_in, 128),
      nn.GELU(),
      nn.Linear(128, 64),
      nn.GELU(),
      nn.Linear(64, 1),
    )

    # retrieval projections
    self.retr_l = nn.Linear(d, retr_d, bias=False)
    self.retr_p = nn.Linear(d, retr_d, bias=False)

    # --- NEW: learnable edge mean/uncertainty calibration ---
    # mu' = mu_a * mu + mu_b
    self.mu_a = nn.Parameter(torch.tensor(1.0))
    self.mu_b = nn.Parameter(torch.tensor(0.0))

    # sigma^2 = wa*siga^2 + wb*sigb^2 + tau^2  (softplus for positivity)
    self.sig_wa = nn.Parameter(torch.tensor(1.0))
    self.sig_wb = nn.Parameter(torch.tensor(1.0))
    self.log_tau = nn.Parameter(torch.tensor(math.log(0.10)))  # noise floor


  # ------------------------- encoders -------------------------

  def encode_batch(self, l_x, l_typ, p_x, p_typ, p_score):
    # l_x [B,L,8], p_x [B,P,192]
    hL = self.proj_l(self.l_enc(l_x, l_typ))            # [B,L,d]
    hP = self.proj_p(self.p_enc(p_x, p_typ, p_score))   # [B,P,d]
    return hL, hP

  def feature_logits_batch(self, hL, hP, l_mask, p_mask):
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

    dust = torch.einsum("bld,d->bl", hL, self.dustbin_vec) / math.sqrt(hL.shape[-1])  # [B,L]
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


  # ------------------------- invariant terms -------------------------

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
    lm = l_mask.float().unsqueeze(-1)  # [B,L,1]
    pm = p_mask.float().unsqueeze(-1)  # [B,P,1]
    l_g = (hL * lm).sum(dim=1) / lm.sum(dim=1).clamp(min=1.0)  # [B,d]
    p_g = (hP * pm).sum(dim=1) / pm.sum(dim=1).clamp(min=1.0)  # [B,d]
    l_z = F.normalize(self.retr_l(l_g), dim=-1)
    p_z = F.normalize(self.retr_p(p_g), dim=-1)
    return l_z, p_z


  # ------------------------- edge term (vectorized + learnable mu/sigma) -------------------------

  def _sample_valid_pairs(self, l_mask: torch.Tensor, M: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (a,b) indices only from valid ligand nodes.
    Returns:
      a [B,M], b [B,M], valid_pair [B,M]  (valid_pair includes a!=b)
    """
    B, L = l_mask.shape
    probs = l_mask.float()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

    # multinomial supports [B,L] -> [B,M]
    a = torch.multinomial(probs, num_samples=M, replacement=True)
    b = torch.multinomial(probs, num_samples=M, replacement=True)
    valid_pair = (a != b) & l_mask[torch.arange(B, device=l_mask.device)[:, None], a] & l_mask[torch.arange(B, device=l_mask.device)[:, None], b]
    return a, b, valid_pair

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

    a, b, valid_pair = self._sample_valid_pairs(l_mask, M)
    bi = torch.arange(B, device=l_pos.device)[:, None]  # [B,1]

    # ligand distances d: [B,M,1,1] (fp32 then cast)
    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      la = l_pos[bi, a].float()
      lb = l_pos[bi, b].float()
      d = torch.norm(la - lb, dim=-1)  # [B,M]
    d = d.to(logW_real.dtype)[:, :, None, None]

    # pick top-K pocket candidates for each ligand node using masked logits
    neg = torch.finfo(logits_feat.dtype).min
    lf = logits_feat.masked_fill(~p_mask[:, None, :], neg)  # [B,L,P]
    _, topj = torch.topk(lf, k=K, dim=-1)                   # [B,L,K]

    # gather pocket positions for those candidates: p_sel [B,L,K,3]
    p_exp = p_pos[:, None, :, :].expand(B, L, P, 3)
    topj3 = topj[..., None].expand(B, L, K, 3)
    p_sel = torch.gather(p_exp, dim=2, index=topj3)

    # gather sig nodes: sig_sel [B,L,K]
    sig_exp = p_sigma[:, None, :].expand(B, L, P)
    sig_sel = torch.gather(sig_exp, dim=2, index=topj)

    # gather logW: [B,L,K]
    logW_top = torch.gather(logW_real, dim=2, index=topj)

    # select per pair (a,b)
    pa = p_sel[bi, a]      # [B,M,K,3]
    pb = p_sel[bi, b]      # [B,M,K,3]
    siga = sig_sel[bi, a]  # [B,M,K]
    sigb = sig_sel[bi, b]  # [B,M,K]
    lwa = logW_top[bi, a]  # [B,M,K]
    lwb = logW_top[bi, b]  # [B,M,K]

    # mu distances: [B,M,K,K] (fp32 then cast)
    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      mu = torch.norm(pa[:, :, :, None, :] - pb[:, :, None, :, :], dim=-1).float()
    mu = mu.to(logW_real.dtype)

    # learnable mean calibration
    mu = self.mu_a * mu + self.mu_b

    # learnable sigma combiner + floor
    wa = F.softplus(self.sig_wa) + 1e-6
    wb = F.softplus(self.sig_wb) + 1e-6
    tau = F.softplus(self.log_tau) + 1e-6

    sigma2 = (wa * (siga ** 2))[:, :, :, None] + (wb * (sigb ** 2))[:, :, None, :] + (tau ** 2)
    sigma = torch.sqrt(sigma2 + 1e-6).to(logW_real.dtype)

    logw_pair = lwa[:, :, :, None] + lwb[:, :, None, :]  # [B,M,K,K]
    logp = gaussian_logpdf(d, mu, sigma)
    logmix = torch.logsumexp(logw_pair + logp, dim=(-1, -2))  # [B,M]

    vp = valid_pair.float()
    denom = vp.sum(dim=1).clamp(min=1.0)
    edge_raw = (logmix * vp).sum(dim=1) / denom
    return edge_raw


  # ------------------------- pose terms (optional, sparse, variance-aware, invariant refine) -------------------------

  def _sparse_pose_weights(
    self,
    logW: torch.Tensor,      # [B,L,P(+1)]
    p_mask: torch.Tensor,    # [B,P]
    l_mask: torch.Tensor,    # [B,L]
    p_pos: torch.Tensor,     # [B,P,3]
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sparse correspondences for pose scoring:
      - choose top-K pocket nodes per ligand node using logW_real
      - renormalize weights over that K
    Returns:
      p_sel: [B,L,K,3]
      w:     [B,L,K]  (sum to 1 over K)
      var_p: [B,L]    (correspondence variance in pocket space)
    """
    B, L, _ = logW.shape
    P = p_pos.shape[1]
    K = min(self.pose_topk, P)

    # drop dustbin
    if self.dustbin and (logW.shape[-1] == P + 1):
      logW_real = logW[:, :, :-1]
    else:
      logW_real = logW

    neg = torch.finfo(logW_real.dtype).min
    logWm = logW_real.masked_fill(~p_mask[:, None, :], neg)
    # [B,L,K]
    topv, topj = torch.topk(logWm, k=K, dim=-1)

    # convert to probs over top-K only
    w = torch.softmax(topv, dim=-1)  # [B,L,K]
    w = w * l_mask[:, :, None].float()

    # gather p_sel [B,L,K,3]
    p_exp = p_pos[:, None, :, :].expand(B, L, P, 3)
    topj3 = topj[..., None].expand(B, L, K, 3)
    p_sel = torch.gather(p_exp, dim=2, index=topj3)

    # barycenter: p_bar [B,L,3]
    p_bar = torch.einsum("blk,blkc->blc", w, p_sel)

    # variance: sum_k w ||p_k - p_bar||^2  -> [B,L]
    with torch.amp.autocast(device_type=p_pos.device.type, enabled=False):
      diff = (p_sel.float() - p_bar[:, :, None, :].float())
      var_p = (w.float() * (diff * diff).sum(dim=-1)).sum(dim=-1)  # [B,L]
    var_p = var_p.to(logW_real.dtype)

    return p_sel, w, var_p

  def pose_term_batch(self, l_pos, p_pos, logW, l_mask, p_mask) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pose_energy [B]  (negative)
      mean_var_p  [B]  (for logging/debug)
    """
    beta = torch.exp(self.log_beta).clamp(0.01, 10.0)

    p_sel, w, var_p = self._sparse_pose_weights(logW, p_mask, l_mask, p_pos)  # p_sel [B,L,K,3], w [B,L,K], var_p [B,L]

    # expected squared distance: sum_k w ||l - p_k||^2
    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      d2 = ((l_pos[:, :, None, :].float() - p_sel.float()) ** 2).sum(dim=-1)  # [B,L,K]
    d2 = d2.to(logW.dtype)

    exp_d2 = (w * d2).sum(dim=-1)  # [B,L]

    denom = l_mask.float().sum(dim=1).clamp(min=1.0)
    pose_energy = ((-beta * exp_d2) * l_mask.float()).sum(dim=1) / denom  # [B]

    mean_var = (var_p * l_mask.float()).sum(dim=1) / denom
    return pose_energy, mean_var

  def refine_term_batch(self, hL, hP, l_pos, p_pos, logW, l_mask, p_mask) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SE(3)-invariant refine term:
      - uses ||l - p_bar|| and optional sqrt(var_p)
      - no raw xyz delta fed to an MLP
    Returns:
      refine_score [B]
      mean_var_p   [B]
    """
    p_sel, w, var_p = self._sparse_pose_weights(logW, p_mask, l_mask, p_pos)

    p_bar = torch.einsum("blk,blkc->blc", w, p_sel)  # [B,L,3]
    hP_bar = torch.einsum("blp,bpd->bld", torch.exp(logW[:, :, :-1] if (self.dustbin and logW.shape[-1] == p_pos.shape[1] + 1) else logW) * p_mask[:, None, :].float(), hP)

    # normalize hP_bar safely (avoid magnitude drift)
    hP_bar = hP_bar / (hP_bar.norm(dim=-1, keepdim=True).clamp(min=1e-6))

    with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
      dn = torch.norm((l_pos.float() - p_bar.float()), dim=-1, keepdim=True)  # [B,L,1]
      var_s = torch.sqrt(var_p.float().clamp(min=0.0) + 1e-6).unsqueeze(-1)    # [B,L,1]
    dn = dn.to(hL.dtype)
    var_s = var_s.to(hL.dtype)

    if self.pose_use_var:
      feat = torch.cat([hL, hP_bar, dn, var_s], dim=-1)
    else:
      feat = torch.cat([hL, hP_bar, dn], dim=-1)

    per = self.refine_mlp(feat).squeeze(-1)  # [B,L]
    denom = l_mask.float().sum(dim=1).clamp(min=1.0)
    refine_score = (per * l_mask.float()).sum(dim=1) / denom

    mean_var = (var_p * l_mask.float()).sum(dim=1) / denom
    return refine_score, mean_var


  # ------------------------- StageA / StageB APIs -------------------------

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
    enable_edge: bool = True,
  ) -> Dict[str, torch.Tensor]:
    topk = self.topk if topk is None else int(topk)
    edge_pairs = self.edge_pairs if edge_pairs is None else int(edge_pairs)

    hL, hP = self.encode_batch(l_x, l_typ, p_x, p_typ, p_score)
    logits_feat = self.feature_logits_batch(hL, hP, l_mask, p_mask)
    logW = self.assignment_logW_batch(logits_feat, hL, l_mask, p_mask)

    sharp, neg_ent = self.invariant_feature_terms_batch(logW, l_mask, p_mask)
    p_sigma = self.pocket_sigma_nodes_batch(hP, p_rad)

    # edge uses real pocket columns only
    if self.dustbin and (logW.shape[-1] == p_pos.shape[1] + 1):
      logW_real = logW[:, :, :-1]
    else:
      logW_real = logW

    if enable_edge and edge_pairs > 0 and topk > 0:
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
    else:
      edge_raw = torch.zeros((l_pos.shape[0],), device=l_pos.device, dtype=logW.dtype)
      edge = torch.zeros_like(edge_raw)

    inv_vec = torch.stack([sharp, neg_ent, edge], dim=1)  # [B,3]
    score_inv_geom = self.head_inv(inv_vec).squeeze(-1)  # [B]

    l_z, p_z = self.pooled_embeddings_batch(hL, hP, l_mask, p_mask)
    retr_sim_diag = (l_z * p_z).sum(dim=-1)

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
    enable_pose: bool = True,
  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    If enable_pose=False, returns the invariant score only.
    """
    if not enable_pose:
      return stageA["score_inv_geom"], {"pose_disabled": torch.ones_like(stageA["score_inv_geom"])}

    pose_t, mean_var_pose = self.pose_term_batch(l_pos, p_pos, stageA["logW"], l_mask, p_mask)
    ref_t, mean_var_ref = self.refine_term_batch(stageA["hL"], stageA["hP"], l_pos, p_pos, stageA["logW"], l_mask, p_mask)

    pose_vec = torch.stack([pose_t, ref_t], dim=1)  # [B,2]
    score_pose = stageA["score_inv_geom"] + self.head_pose(pose_vec).squeeze(-1)

    dbg = {
      "pose_term": pose_t.detach(),
      "refine_term": ref_t.detach(),
      "pose_var": mean_var_pose.detach(),
      "ref_var": mean_var_ref.detach(),
    }
    return score_pose, dbg

  def score_retrieval_from_stageA_batch(
    self,
    stageA: Dict[str, torch.Tensor],
    lambda_retr: float = 0.8,
  ) -> torch.Tensor:
    """
    Screening score without pose: inv + lambda*retr_sim
    """
    return stageA["score_inv_geom"] + float(lambda_retr) * stageA["retr_sim"]
