#!/usr/bin/env python3
"""
Batch compare: one fixed pocket (item_pt) vs many other ligands (from other .pt files).

Writes a text report:
  out_dir/batch_report.txt

Example:
  python infer_batch_compare.py \
    --ckpt _runs_fast_retr_v2/best.pt \
    --item_pt _cache_pt_multiNl_full_plus_trueconf1/00000042.pt \
    --other_dir _cache_pt_multiNl_full_plus_trueconf1 \
    --other_n 10 \
    --other_seed 0 \
    --device cuda --amp \
    --n_confs 50 --optimize \
    --lambda_retr 0.8 \
    --topk 4 --edge_pairs 16 \
    --out_dir _infer_out/batch_00000042

Notes:
- Uses model settings from checkpoint args when available (dustbin, edge_scale, d_model, retr_d).
- other ligands are selected from other_dir/*.pt excluding item_pt itself.
- For each other ligand, uses record 0 from its ligand SDF as SMILES seed.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- RDKit ----------
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

RDLogger.DisableLog("rdApp.warning")

LOG_2PI = math.log(2.0 * math.pi)

# =========================
# RDKit pharm featurization
# =========================
_FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_RDKIT_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF)

FAMILY_TO_TYP = {
    "Donor": 0,
    "Acceptor": 1,
    "Aromatic": 2,
    "Hydrophobe": 3,
    "LumpedHydrophobe": 3,
    "PosIonizable": 4,
    "NegIonizable": 5,
    "Halogen": 7,
}

def _feature_center(mol: Chem.Mol, atom_ids: List[int], confId: int = -1) -> np.ndarray:
    conf = mol.GetConformer(confId)
    pts = []
    for a in atom_ids:
        p = conf.GetAtomPosition(int(a))
        pts.append([p.x, p.y, p.z])
    if not pts:
        pts = [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
               for i in range(mol.GetNumAtoms())]
    return np.mean(np.asarray(pts, dtype=np.float32), axis=0)

def lig_pharm_from_mol(mol: Chem.Mol, confId: int = -1, max_nodes: int = 48) -> Dict[str, torch.Tensor]:
    feats = _RDKIT_FACTORY.GetFeaturesForMol(mol, confId=confId)
    pos_list: List[np.ndarray] = []
    typ_list: List[int] = []

    for f in feats:
        fam = f.GetFamily()
        if fam not in FAMILY_TO_TYP:
            continue
        pos_list.append(_feature_center(mol, list(f.GetAtomIds()), confId=confId))
        typ_list.append(int(FAMILY_TO_TYP[fam]))

    if not pos_list:
        conf = mol.GetConformer(confId)
        pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())], dtype=np.float32)
        pos_list = [pts.mean(axis=0)]
        typ_list = [3]

    if max_nodes and len(pos_list) > max_nodes:
        pos_list = pos_list[:max_nodes]
        typ_list = typ_list[:max_nodes]

    pos = torch.tensor(np.stack(pos_list, axis=0), dtype=torch.float32)          # [Nl,3]
    typ = torch.tensor(np.asarray(typ_list, dtype=np.int64), dtype=torch.long)   # [Nl]
    score = torch.ones((pos.size(0),), dtype=torch.float32)                      # [Nl]

    x = torch.zeros((pos.size(0), 8), dtype=torch.float32)
    x.scatter_(1, typ.clamp(0, 7).view(-1, 1), 1.0)

    return {"pos": pos, "typ": typ, "x": x, "score": score}

def read_pose_and_smiles_from_sdf(sdf_path: Path, which_record: int = 0) -> Tuple[Chem.Mol, str]:
    sdf_path = os.path.join("artifacts/data", sdf_path)
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"Could not read any molecules from SDF: {sdf_path}")
    idx = int(which_record)
    if idx < 0 or idx >= len(mols):
        raise ValueError(f"--which_record={idx} out of range; SDF has {len(mols)} records")
    mol = mols[idx]
    if mol.GetNumConformers() < 1:
        raise ValueError(f"Selected SDF record has no conformer coords: {sdf_path}")
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return mol, smi

def generate_conformers_from_smiles(smiles: str, n_confs: int, seed: int, optimize: bool) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params)
    if not conf_ids:
        raise ValueError("Conformer generation failed")

    if optimize:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if mp is not None:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=200)
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=200)
    return mol

def extract_single_conformer_mol(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
    m = Chem.Mol(mol)
    conf = mol.GetConformer(int(conf_id))
    m.RemoveAllConformers()
    m.AddConformer(Chem.Conformer(conf), assignId=True)
    return m


# =========================
# Cache IO
# =========================
def safe_torch_load(fp: Path):
    try:
        return torch.load(fp, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(fp, map_location="cpu")

def load_item_paths(item_pt: Path) -> Tuple[Path, Path]:
    data = safe_torch_load(item_pt)
    meta = data.get("meta", {})
    lig = meta.get("ligand_file", None)
    prot = meta.get("protein_file", None)
    if lig is None or prot is None:
        raise ValueError(f"{item_pt} meta must include ligand_file and protein_file")
    return Path(prot), Path(lig)

def load_pocket_from_item(item_pt: Path) -> Dict[str, torch.Tensor]:
    data = safe_torch_load(item_pt)
    P = data["pocket_pharm"]
    return {
        "pos": P["pos"].float(),
        "x": P["x"].float(),
        "typ": P.get("typ", torch.zeros(P["pos"].shape[0], dtype=torch.long)).long(),
        "score": P.get("score", torch.ones(P["pos"].shape[0])).float(),
        "rad": P.get("rad", torch.zeros(P["pos"].shape[0])).float(),
    }

def to_device_pharm(ph: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in ph.items()}


def gaussian_logpdf(d: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp(min=1e-4)
    z = (d - mu) / sigma
    return -0.5 * (z * z) - torch.log(sigma) - 0.5 * LOG_2PI

class NodeEncoder(nn.Module):
    def __init__(self, x_dim: int, d: int, n_types: int = 256, use_score=False):
        super().__init__()
        self.use_score = use_score
        self.type_emb = nn.Embedding(n_types, d)
        self.x_mlp = nn.Sequential(nn.Linear(x_dim, d), nn.GELU(), nn.Linear(d, d))
        self.s_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d)) if use_score else None
        self.ln = nn.LayerNorm(d)

    def forward(self, x, typ, score=None):
        typ = typ.clamp(0, self.type_emb.num_embeddings - 1)
        h = self.x_mlp(x) + self.type_emb(typ)
        if self.use_score and (score is not None):
            h = h + self.s_mlp(score.unsqueeze(-1))
        return self.ln(h)

class PharmMatchNetFast(nn.Module):
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
            nn.Linear(ref_in, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

        self.retr_l = nn.Linear(d, retr_d, bias=False)
        self.retr_p = nn.Linear(d, retr_d, bias=False)

    def encode(self, l_x, l_typ, p_x, p_typ, p_score):
        hL = self.proj_l(self.l_enc(l_x, l_typ))
        hP = self.proj_p(self.p_enc(p_x, p_typ, p_score))
        return hL, hP

    def feature_logits(self, hL, hP, l_mask, p_mask):
        logits = (hL @ hP.t()) / math.sqrt(hL.shape[-1])
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~p_mask.unsqueeze(0), neg)
        logits = logits.masked_fill(~l_mask.unsqueeze(1), neg)
        return logits

    def assignment_logW(self, logits_feat, hL, l_mask, p_mask):
        neg = torch.finfo(logits_feat.dtype).min
        if not self.dustbin:
            logits = logits_feat.masked_fill(~p_mask.unsqueeze(0), neg)
            logits = logits.masked_fill(~l_mask.unsqueeze(1), neg)
            return F.log_softmax(logits, dim=1)

        dust = (hL @ self.dustbin_vec) / math.sqrt(hL.shape[-1])  # [L]
        logits_real = logits_feat.masked_fill(~p_mask.unsqueeze(0), neg)
        logits = torch.cat([logits_real, dust[:, None]], dim=1)    # [L,P+1]
        logits = logits.masked_fill(~l_mask.unsqueeze(1), neg)
        return F.log_softmax(logits, dim=1)

    def pocket_sigma_nodes(self, hP, p_rad):
        log_sig = self.p_logsig_head(hP).squeeze(-1)
        sig = F.softplus(log_sig) + 1e-3
        sig = sig + p_rad.clamp(min=0.0)
        return sig

    def invariant_feature_terms(self, logW, l_mask, p_mask):
        W = torch.exp(logW)
        denom = l_mask.float().sum().clamp(min=1.0)

        if self.dustbin and (W.shape[1] == p_mask.numel() + 1):
            W_real = W[:, :-1] * p_mask.unsqueeze(0).float()
            W_dust = W[:, -1:]
            Wm = torch.cat([W_real, W_dust], dim=1)
        else:
            Wm = W * p_mask.unsqueeze(0).float()

        sharp = (Wm.max(dim=1).values * l_mask.float()).sum() / denom
        ent = (-(Wm * logW).sum(dim=1) * l_mask.float()).sum() / denom
        return sharp, -ent

    def pooled_embeddings(self, hL, hP, l_mask, p_mask):
        lm = l_mask.float().unsqueeze(-1)
        pm = p_mask.float().unsqueeze(-1)
        l_g = (hL * lm).sum(dim=0) / lm.sum(dim=0).clamp(min=1.0)
        p_g = (hP * pm).sum(dim=0) / pm.sum(dim=0).clamp(min=1.0)
        l_z = F.normalize(self.retr_l(l_g), dim=-1)
        p_z = F.normalize(self.retr_p(p_g), dim=-1)
        return l_z, p_z

    def _sample_ligand_pairs(self, Nl_valid: int, device: torch.device, max_pairs: int):
        total_pairs = Nl_valid * (Nl_valid - 1) // 2
        if total_pairs <= max_pairs:
            return torch.combinations(torch.arange(Nl_valid, device=device), r=2)
        all_pairs = torch.combinations(torch.arange(Nl_valid, device=device), r=2)
        idx = torch.randperm(all_pairs.shape[0], device=device)[:max_pairs]
        return all_pairs[idx]

    def edge_likelihood_term(self, l_pos, p_pos, logits_feat, logW_real, l_mask, p_mask, p_sigma_nodes, topk=None, edge_pairs=None):
        K_req = int(self.topk if topk is None else topk)
        M_req = int(self.edge_pairs if edge_pairs is None else edge_pairs)

        l_idx = torch.nonzero(l_mask, as_tuple=False).squeeze(-1)
        p_idx = torch.nonzero(p_mask, as_tuple=False).squeeze(-1)
        Nl = int(l_idx.numel())
        Np = int(p_idx.numel())
        if Nl < 2 or Np < 2:
            return torch.tensor(0.0, device=l_pos.device, dtype=logW_real.dtype)

        with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
            dL = torch.cdist(l_pos[l_idx].float(), l_pos[l_idx].float())
            dP = torch.cdist(p_pos[p_idx].float(), p_pos[p_idx].float())
        dL = dL.to(logW_real.dtype)
        dP = dP.to(logW_real.dtype)

        lf = logits_feat[l_idx][:, p_idx]
        K = min(K_req, Np)
        _, topj = torch.topk(lf, k=K, dim=1)
        logW_vp = logW_real[l_idx][:, p_idx]
        logW_top = torch.gather(logW_vp, 1, topj)
        p_sel = p_idx[topj]
        sig_nodes = p_sigma_nodes[p_sel]

        inv = torch.full((p_pos.shape[0],), -1, device=p_pos.device, dtype=torch.long)
        inv[p_idx] = torch.arange(Np, device=p_pos.device)

        pairs = self._sample_ligand_pairs(Nl, device=l_pos.device, max_pairs=M_req)
        total = torch.tensor(0.0, device=l_pos.device, dtype=logW_real.dtype)

        for ab in pairs:
            a = int(ab[0].item()); b = int(ab[1].item())
            d = dL[a, b]
            ia = inv[p_sel[a]]
            ib = inv[p_sel[b]]
            mu = dP[ia][:, ib]
            siga = sig_nodes[a]; sigb = sig_nodes[b]
            sigma = torch.sqrt((siga[:, None] ** 2) + (sigb[None, :] ** 2) + 1e-6)
            logw_pair = logW_top[a][:, None] + logW_top[b][None, :]
            logp = gaussian_logpdf(d, mu, sigma)
            logmix = torch.logsumexp(logw_pair + logp, dim=(0, 1))
            total = total + logmix

        return total / float(pairs.shape[0])

    def pose_term(self, l_pos, p_pos, logW, l_mask, p_mask):
        beta = torch.exp(self.log_beta).clamp(0.01, 10.0)
        if self.dustbin and (logW.shape[1] == p_pos.shape[0] + 1):
            logW_real = logW[:, :-1]
        else:
            logW_real = logW

        with torch.amp.autocast(device_type=l_pos.device.type, enabled=False):
            d2 = torch.cdist(l_pos.float(), p_pos.float()) ** 2
        d2 = d2.to(logW_real.dtype)

        W = torch.exp(logW_real) * p_mask.unsqueeze(0).float()
        Z = W.sum(dim=1, keepdim=True).clamp(min=1e-6)
        Wn = W / Z

        exp_d2 = (Wn * d2).sum(dim=1)
        denom = l_mask.float().sum().clamp(min=1.0)
        return ((-beta * exp_d2) * l_mask.float()).sum() / denom

    def refine_term(self, hL, hP, l_pos, p_pos, logW, l_mask, p_mask):
        if self.dustbin and (logW.shape[1] == p_pos.shape[0] + 1):
            logW_real = logW[:, :-1]
        else:
            logW_real = logW

        W = torch.exp(logW_real) * p_mask.unsqueeze(0).float()
        Z = W.sum(dim=1, keepdim=True).clamp(min=1e-6)
        Wn = W / Z

        p_pos_bar = Wn @ p_pos
        hP_bar = Wn @ hP
        delta = l_pos - p_pos_bar
        dn = torch.norm(delta, dim=-1, keepdim=True)

        feat = torch.cat([hL, hP_bar, delta, dn], dim=-1)
        per = self.refine_mlp(feat).squeeze(-1)

        denom = l_mask.float().sum().clamp(min=1.0)
        return (per * l_mask.float()).sum() / denom

    def precompute_stageA(self, l_pos, l_x, l_typ, l_mask, p_pos, p_x, p_typ, p_score, p_rad, p_mask, topk=None, edge_pairs=None):
        hL, hP = self.encode(l_x, l_typ, p_x, p_typ, p_score)
        logits_feat = self.feature_logits(hL, hP, l_mask, p_mask)
        logW = self.assignment_logW(logits_feat, hL, l_mask, p_mask)

        sharp, neg_ent = self.invariant_feature_terms(logW, l_mask, p_mask)
        p_sigma = self.pocket_sigma_nodes(hP, p_rad)

        if self.dustbin and (logW.shape[1] == p_pos.shape[0] + 1):
            logW_real = logW[:, :-1]
        else:
            logW_real = logW

        edge_raw = self.edge_likelihood_term(
            l_pos=l_pos, p_pos=p_pos,
            logits_feat=logits_feat, logW_real=logW_real,
            l_mask=l_mask, p_mask=p_mask,
            p_sigma_nodes=p_sigma,
            topk=topk, edge_pairs=edge_pairs,
        )
        edge = torch.tanh(edge_raw / float(self.edge_scale))

        inv_vec = torch.stack([sharp, neg_ent, edge], dim=0)
        score_inv_geom = self.head_inv(inv_vec.view(1, -1)).view(())

        l_z, p_z = self.pooled_embeddings(hL, hP, l_mask, p_mask)
        retr_sim = (l_z * p_z).sum().view(())

        return {
            "hL": hL, "hP": hP,
            "logW": logW,
            "score_inv_geom": score_inv_geom,
            "sharp": sharp,
            "neg_ent": neg_ent,
            "edge": edge,
            "retr_sim": retr_sim,
        }

    def score_pose_from_stageA(self, stageA, l_pos, l_mask, p_pos, p_mask):
        pose_t = self.pose_term(l_pos, p_pos, stageA["logW"], l_mask, p_mask)
        ref_t = self.refine_term(stageA["hL"], stageA["hP"], l_pos, p_pos, stageA["logW"], l_mask, p_mask)
        pose_vec = torch.stack([pose_t, ref_t], dim=0)
        score_pose = stageA["score_inv_geom"] + self.head_pose(pose_vec.view(1, -1)).view(())
        return score_pose, {"pose": pose_t.detach(), "ref": ref_t.detach()}



@torch.inference_mode()
def score_gt_and_top2_confs(
    model: PharmMatchNetFast,
    pocket: Dict[str, torch.Tensor],
    pocket_cent: torch.Tensor,
    gt_mol: Chem.Mol,
    smiles: str,
    device: torch.device,
    use_amp: bool,
    topk: int,
    edge_pairs: int,
    n_confs: int,
    seed: int,
    optimize: bool,
    max_lig_nodes: int,
    lambda_retr: float,
) -> Dict[str, Any]:
    # pocket centered
    p_pos = pocket["pos"].to(device) - pocket_cent[None, :]
    p_mask = torch.ones((p_pos.shape[0],), dtype=torch.bool, device=device)

    gt_ph = lig_pharm_from_mol(gt_mol, confId=0, max_nodes=max_lig_nodes)
    l_pos = gt_ph["pos"].to(device) - pocket_cent[None, :]
    l_x = gt_ph["x"].to(device)
    l_typ = gt_ph["typ"].to(device)
    l_mask = torch.ones((l_pos.shape[0],), dtype=torch.bool, device=device)

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        stA = model.precompute_stageA(
            l_pos, l_x, l_typ, l_mask,
            p_pos, pocket["x"].to(device), pocket["typ"].to(device), pocket["score"].to(device), pocket["rad"].to(device), p_mask,
            topk=topk, edge_pairs=edge_pairs,
        )
        pose, pose_dbg = model.score_pose_from_stageA(stA, l_pos, l_mask, p_pos, p_mask)

    gt_inv = float(stA["score_inv_geom"].item())
    gt_pose = float(pose.item())
    gt_sim = float(stA["retr_sim"].item())
    gt_retr = gt_inv + lambda_retr * gt_sim
    gt_terms = {
        "sharp": float(stA["sharp"].item()),
        "neg_ent": float(stA["neg_ent"].item()),
        "edge": float(stA["edge"].item()),
        "pose": float(pose_dbg["pose"].item()),
        "ref": float(pose_dbg["ref"].item()),
    }

    conf_mol = generate_conformers_from_smiles(smiles, n_confs=n_confs, seed=seed, optimize=optimize)

    best = []
    for cid in range(conf_mol.GetNumConformers()):
        mol_c = extract_single_conformer_mol(conf_mol, cid)
        ph = lig_pharm_from_mol(mol_c, confId=0, max_nodes=max_lig_nodes)


        lpos = ph["pos"].to(device)
        lx = ph["x"].to(device)
        lty = ph["typ"].to(device)
        lm = torch.ones((lpos.shape[0],), dtype=torch.bool, device=device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            st = model.precompute_stageA(
                lpos, lx, lty, lm,
                p_pos, pocket["x"].to(device), pocket["typ"].to(device), pocket["score"].to(device), pocket["rad"].to(device), p_mask,
                topk=topk, edge_pairs=edge_pairs,
            )
        inv = float(st["score_inv_geom"].item())
        sim = float(st["retr_sim"].item())
        sretr = inv + lambda_retr * sim
        best.append((sretr, cid, inv, sim))

    best.sort(key=lambda x: x[0], reverse=True)
    top2 = best[:2]

    return {
        "gt": {
            "score_inv_geom": gt_inv,
            "score_pose": gt_pose,
            "retr_sim": gt_sim,
            "score_retr": gt_retr,
            "terms": gt_terms,
        },
        "top2": [
            {"conf_id": int(top2[0][1]), "score_inv_geom": float(top2[0][2]), "retr_sim": float(top2[0][3]), "score_retr": float(top2[0][0])},
            {"conf_id": int(top2[1][1]), "score_inv_geom": float(top2[1][2]), "retr_sim": float(top2[1][3]), "score_retr": float(top2[1][0])},
        ] if len(top2) >= 2 else [],
    }

@torch.inference_mode()
def score_best_other_ligand_on_same_pocket(
    model: PharmMatchNetFast,
    pocket: Dict[str, torch.Tensor],
    pocket_cent: torch.Tensor,
    other_sdf: Path,
    device: torch.device,
    use_amp: bool,
    topk: int,
    edge_pairs: int,
    n_confs: int,
    seed: int,
    optimize: bool,
    max_lig_nodes: int,
    lambda_retr: float,
) -> Dict[str, Any]:
    other_mol, other_smiles = read_pose_and_smiles_from_sdf(other_sdf, which_record=0)

    # pocket centered
    p_pos = pocket["pos"].to(device) - pocket_cent[None, :]
    p_mask = torch.ones((p_pos.shape[0],), dtype=torch.bool, device=device)

    conf_mol = generate_conformers_from_smiles(other_smiles, n_confs=n_confs, seed=seed, optimize=optimize)

    best = None
    for cid in range(conf_mol.GetNumConformers()):
        mol_c = extract_single_conformer_mol(conf_mol, cid)
        ph = lig_pharm_from_mol(mol_c, confId=0, max_nodes=max_lig_nodes)

        lpos = ph["pos"].to(device)
        lx = ph["x"].to(device)
        lty = ph["typ"].to(device)
        lm = torch.ones((lpos.shape[0],), dtype=torch.bool, device=device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            st = model.precompute_stageA(
                lpos, lx, lty, lm,
                p_pos, pocket["x"].to(device), pocket["typ"].to(device), pocket["score"].to(device), pocket["rad"].to(device), p_mask,
                topk=topk, edge_pairs=edge_pairs,
            )

        inv = float(st["score_inv_geom"].item())
        sim = float(st["retr_sim"].item())
        sretr = inv + lambda_retr * sim

        if best is None or sretr > best["score_retr"]:
            best = {
                "smiles": other_smiles,
                "best_conf": int(cid),
                "score_inv_geom": inv,
                "retr_sim": sim,
                "score_retr": sretr,
                "inv_terms": {
                    "sharp": float(st["sharp"].item()),
                    "neg_ent": float(st["neg_ent"].item()),
                    "edge": float(st["edge"].item()),
                }
            }

    return best



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--item_pt", required=True)
    ap.add_argument("--other_dir", required=True, help="Directory containing many other *.pt to sample from")
    ap.add_argument("--other_n", type=int, default=10)
    ap.add_argument("--other_seed", type=int, default=0)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--n_confs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--max_lig_nodes", type=int, default=48)

    ap.add_argument("--lambda_retr", type=float, default=0.8)

    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--edge_pairs", type=int, default=None)

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "batch_report.txt"

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    ckpt = safe_torch_load(Path(args.ckpt))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    d_model = int(ckpt_args.get("d_model", 128))
    retr_d = int(ckpt_args.get("retr_d", 128))
    dustbin = bool(ckpt_args.get("dustbin", False))
    edge_scale = float(ckpt_args.get("edge_scale", 5.0))
    topk_default = int(ckpt_args.get("topk", 4))
    edge_pairs_default = int(ckpt_args.get("edge_pairs", 16))

    topk = int(args.topk if args.topk is not None else topk_default)
    edge_pairs = int(args.edge_pairs if args.edge_pairs is not None else edge_pairs_default)

    model = PharmMatchNetFast(
        d=d_model,
        retr_d=retr_d,
        topk=topk,
        edge_pairs=edge_pairs,
        dustbin=dustbin,
        edge_scale=edge_scale,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    protein_pdb, ligand_sdf = load_item_paths(Path(args.item_pt))
    pocket_cpu = load_pocket_from_item(Path(args.item_pt))
    pocket = to_device_pharm(pocket_cpu, device)

    p_pos = pocket_cpu["pos"].cpu().numpy()
    cent_np = p_pos.mean(axis=0).astype(np.float32) if p_pos.size else np.zeros((3,), np.float32)
    pocket_cent = torch.tensor(cent_np, device=device, dtype=torch.float32)

    gt_mol, gt_smiles = read_pose_and_smiles_from_sdf(Path(ligand_sdf), which_record=0)

    base = score_gt_and_top2_confs(
        model=model,
        pocket=pocket_cpu,
        pocket_cent=pocket_cent,
        gt_mol=gt_mol,
        smiles=gt_smiles,
        device=device,
        use_amp=use_amp,
        topk=topk,
        edge_pairs=edge_pairs,
        n_confs=int(args.n_confs),
        seed=int(args.seed),
        optimize=bool(args.optimize),
        max_lig_nodes=int(args.max_lig_nodes),
        lambda_retr=float(args.lambda_retr),
    )

    # Choose N other pt files
    other_dir = Path(args.other_dir)
    all_pts = sorted([p for p in other_dir.glob("*.pt") if p.is_file()])
    all_pts = [p for p in all_pts if p.resolve() != Path(args.item_pt).resolve()]
    rng = random.Random(int(args.other_seed))
    rng.shuffle(all_pts)
    picks = all_pts[: int(args.other_n)]

    lines = []
    lines.append("=== Batch compare report ===")
    lines.append(f"ckpt: {args.ckpt}")
    lines.append(f"item_pt (pocket+GT): {args.item_pt}")
    lines.append(f"protein_pdb: {protein_pdb}")
    lines.append(f"gt_ligand_sdf: {ligand_sdf}")
    lines.append(f"gt_smiles: {gt_smiles}")
    lines.append("")
    lines.append(f"model: d_model={d_model} retr_d={retr_d} dustbin={dustbin} edge_scale={edge_scale} topk={topk} edge_pairs={edge_pairs}")
    lines.append(f"scoring: lambda_retr={args.lambda_retr}")
    lines.append(f"confs: n_confs={args.n_confs} optimize={args.optimize} seed={args.seed}")
    lines.append("")

    gt = base["gt"]
    lines.append("GT (centered like training):")
    lines.append(f"  score_inv_geom: {gt['score_inv_geom']:.6f}")
    lines.append(f"  score_pose    : {gt['score_pose']:.6f}")
    lines.append(f"  retr_sim      : {gt['retr_sim']:.6f}")
    lines.append(f"  score_retr    : {gt['score_retr']:.6f}")
    t = gt["terms"]
    lines.append(f"  inv terms     : sharp={t['sharp']:.4f} neg_ent={t['neg_ent']:.4f} edge={t['edge']:.4f}")
    lines.append(f"  pose terms    : pose={t['pose']:.4f} ref={t['ref']:.4f}")
    lines.append("")

    lines.append("Top conformers (by score_retr = inv + lambda*retr_sim):")
    for k, r in enumerate(base["top2"], start=1):
        lines.append(f"  conf{k} id={r['conf_id']} inv={r['score_inv_geom']:.4f} retr_sim={r['retr_sim']:.4f} score_retr={r['score_retr']:.4f}")
    lines.append("")

    lines.append(f"Other ligands: sampling {len(picks)} from {other_dir}")
    lines.append("")

    results = []
    for idx, pt in enumerate(picks, start=1):
        try:
            _pdb2, other_sdf = load_item_paths(pt)
            best = score_best_other_ligand_on_same_pocket(
                model=model,
                pocket=pocket_cpu,
                pocket_cent=pocket_cent,
                other_sdf=other_sdf,
                device=device,
                use_amp=use_amp,
                topk=topk,
                edge_pairs=edge_pairs,
                n_confs=int(args.n_confs),
                seed=int(args.seed),
                optimize=bool(args.optimize),
                max_lig_nodes=int(args.max_lig_nodes),
                lambda_retr=float(args.lambda_retr),
            )
            results.append((best["score_retr"], pt, other_sdf, best))
        except Exception as e:
            results.append((-1e9, pt, None, {"error": str(e)}))

    results.sort(key=lambda x: x[0], reverse=True)

    for rank, (sretr, pt, other_sdf, best) in enumerate(results, start=1):
        lines.append(f"[{rank:02d}] other_item_pt: {pt}")
        if other_sdf is not None:
            lines.append(f"     other_sdf: {other_sdf}")
        if "error" in best:
            lines.append(f"     ERROR: {best['error']}")
            lines.append("")
            continue
        lines.append(f"     best_conf id={best['best_conf']} score_inv_geom={best['score_inv_geom']:.6f} retr_sim={best['retr_sim']:.6f} score_retr={best['score_retr']:.6f}")
        it = best["inv_terms"]
        lines.append(f"     inv_terms: sharp={it['sharp']:.4f} neg_ent={it['neg_ent']:.4f} edge={it['edge']:.4f}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n")
    print("[done] wrote:", report_path)

if __name__ == "__main__":
    main()
