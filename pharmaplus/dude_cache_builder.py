#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build P-L style cache .pt items from a single DUD / DUD-E target folder.

Supports BOTH folder layouts:

NEW (DUD-E style):
ace/
  receptor.pdb
  crystal_ligand.mol2
  actives_final.mol2.gz
  decoys_final.mol2.gz
  (optionally actives_final.sdf.gz / decoys_final.sdf.gz / *.ism)

OLD (your previous DUD2006 example):
ace/
  databases/dud_ligands2006/<target>_ligands.mol2.gz
  databases/dud_decoys2006/<target>_decoys.mol2.gz
  targets/<target>/rec.pdb
  targets/<target>/xtal-lig.mol2

Outputs (out_dir):
  - query.pt                       (pocket_pharm + lig_pharm from crystal ligand)
  - cache/00000000.pt ...          (each has pocket_pharm + lig_pharm + meta)
  - sdf/00000000.sdf ...           (per-ligand SDF written for reuse by infer/screen)
  - manifest.csv
  - build_report.json

Notes:
- This script DOES NOT generate conformers. It requires 3D coordinates in inputs (MOL2 blocks).
- pocket_pharm is duplicated into every cache item for backward compatibility with your existing pipeline.

Example (NEW DUD-E layout):
  python build_dude_cache.py --target_dir ace --out_dir out/ace_cache --device cuda

Example (override paths explicitly):
  python build_dude_cache.py --target_dir ace --out_dir out/ace_cache \
    --receptor_pdb ace/receptor.pdb \
    --crystal_mol2 ace/crystal_ligand.mol2 \
    --actives_mol2_gz ace/actives_final.mol2.gz \
    --decoys_mol2_gz ace/decoys_final.mol2.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# ---------- PMNet imports ----------
from pmnet.api import get_pmnet_dev
from pmnet.data.parser import ProteinParser

# ---------- RDKit ----------
from rdkit import Chem, RDLogger
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

RDLogger.DisableLog("rdApp.warning")

# Build RDKit feature factory once
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


# ------------------------- helpers -------------------------

def safe_torch_save(obj: Any, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, fp)


def _feature_center(mol: Chem.Mol, atom_ids: List[int], confId: int = 0) -> np.ndarray:
    conf = mol.GetConformer(int(confId))
    pts = []
    for a in atom_ids:
        p = conf.GetAtomPosition(int(a))
        pts.append([p.x, p.y, p.z])
    if not pts:
        pts = [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ]
    return np.mean(np.asarray(pts, dtype=np.float32), axis=0)


def lig_pharm_from_mol(mol: Chem.Mol, confId: int = 0, max_nodes: int = 48) -> Dict[str, torch.Tensor]:
    """
    RDKit feature-based ligand pharmacophore from an EXISTING 3D conformer.
    No conformer generation is performed.
    """
    if mol is None:
        raise ValueError("lig_pharm_from_mol got mol=None")
    if mol.GetNumConformers() < 1:
        raise ValueError("Ligand mol has no conformers/3D coordinates (won't generate).")

    feats = _RDKIT_FACTORY.GetFeaturesForMol(mol, confId=int(confId))
    pos_list: List[np.ndarray] = []
    typ_list: List[int] = []

    for f in feats:
        fam = f.GetFamily()
        if fam not in FAMILY_TO_TYP:
            continue
        pos_list.append(_feature_center(mol, list(f.GetAtomIds()), confId=confId))
        typ_list.append(int(FAMILY_TO_TYP[fam]))

    # fallback to centroid if no features
    if not pos_list:
        conf = mol.GetConformer(int(confId))
        pts = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        pos_list = [pts.mean(axis=0)]
        typ_list = [3]  # Hydrophobe-ish placeholder

    if max_nodes and len(pos_list) > max_nodes:
        pos_list = pos_list[:max_nodes]
        typ_list = typ_list[:max_nodes]

    pos = torch.tensor(np.stack(pos_list, axis=0), dtype=torch.float32)           # [Nl,3]
    typ = torch.tensor(np.asarray(typ_list, dtype=np.int64), dtype=torch.long)    # [Nl]
    score = torch.ones((pos.size(0),), dtype=torch.float32)                       # [Nl]

    # 8-d one-hot (matches your model expectations)
    x = torch.zeros((pos.size(0), 8), dtype=torch.float32)
    x.scatter_(1, typ.clamp(0, 7).view(-1, 1), 1.0)

    return {"pos": pos, "typ": typ, "x": x, "score": score}


def iter_mol2_blocks(mol2_or_gz: Path) -> Iterator[Tuple[int, str, str]]:
    """
    Stream a multi-molecule MOL2 or MOL2.GZ.
    Yields: (record_idx, mol2_block_text, ligand_name)

    Assumes blocks start with '@<TRIPOS>MOLECULE'.
    """
    mol2_or_gz = Path(mol2_or_gz)
    if not mol2_or_gz.exists():
        raise FileNotFoundError(mol2_or_gz)

    opener = gzip.open if str(mol2_or_gz).endswith(".gz") else open

    def flush_block(buf: List[str], rec_idx: int) -> Optional[Tuple[int, str, str]]:
        if not buf:
            return None
        txt = "".join(buf)
        lig_name = "UNK"
        for i, line in enumerate(buf):
            if line.startswith("@<TRIPOS>MOLECULE"):
                if i + 1 < len(buf):
                    lig_name = buf[i + 1].strip() or "UNK"
                break
        return rec_idx, txt, lig_name

    with opener(str(mol2_or_gz), "rt", encoding="utf-8", errors="ignore") as f:
        buf: List[str] = []
        rec = -1
        started = False

        for line in f:
            if line.startswith("@<TRIPOS>MOLECULE"):
                if started:
                    out = flush_block(buf, rec)
                    if out is not None:
                        yield out
                buf = [line]
                rec += 1
                started = True
            elif started:
                buf.append(line)

        out = flush_block(buf, rec)
        if out is not None:
            yield out


def mol_from_mol2_block(mol2_block: str) -> Chem.Mol:
    """
    Parse single MOL2 block with coordinates. No conformer generation.
    """
    m = Chem.MolFromMol2Block(mol2_block, sanitize=False, removeHs=False)
    if m is None:
        raise ValueError("RDKit failed to parse MOL2 block.")
    try:
        Chem.SanitizeMol(m)
    except Exception:
        # features may still work unsanitized
        pass
    if m.GetNumConformers() < 1:
        raise ValueError("Parsed MOL2 has no conformer coords.")
    return m


def write_single_sdf(mol: Chem.Mol, out_sdf: Path):
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    w.write(mol)
    w.close()


def mol2_to_sdf(mol2_path: Path, out_sdf: Path) -> Path:
    """
    Convert a single-molecule MOL2 to SDF (used only for PMNet ProteinParser ref ligand).
    """
    m = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
    if m is None:
        raise ValueError(f"RDKit failed reading mol2: {mol2_path}")
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    if m.GetNumConformers() < 1:
        raise ValueError(f"{mol2_path} has no 3D conformer coords (won't generate).")
    write_single_sdf(m, out_sdf)
    return out_sdf


def pmnet_hotspots_to_tensors(pmnet_attr: Any, max_nodes: int = 256) -> Dict[str, torch.Tensor]:
    def _get_first_from_mapping(d: dict, names: List[str]):
        for k in names:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _as_tensor_3(v) -> Optional[torch.Tensor]:
        if v is None:
            return None
        if torch.is_tensor(v):
            v = v.detach().cpu()
            if v.numel() == 3:
                return v.reshape(3).to(torch.float32)
            if v.ndim == 2 and v.shape[-1] == 3 and v.shape[0] == 1:
                return v[0].to(torch.float32)
            return None
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return torch.tensor(v, dtype=torch.float32)
        try:
            arr = np.asarray(v)
            if arr.size == 3:
                return torch.tensor(arr.reshape(3), dtype=torch.float32)
        except Exception:
            pass
        return None

    def _as_float(v) -> Optional[float]:
        if v is None:
            return None
        if torch.is_tensor(v) and v.numel() == 1:
            return float(v.detach().cpu().item())
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _as_int(v) -> Optional[int]:
        if v is None:
            return None
        if torch.is_tensor(v) and v.numel() == 1:
            return int(v.detach().cpu().item())
        if isinstance(v, (int, bool)):
            return int(v)
        if isinstance(v, float):
            return int(v)
        return None

    def _iter_candidate_tensors(obj) -> List[torch.Tensor]:
        cands: List[torch.Tensor] = []
        if obj is None:
            return cands
        if torch.is_tensor(obj):
            return [obj]
        if isinstance(obj, (list, tuple)):
            for x in obj:
                if torch.is_tensor(x):
                    cands.append(x)
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                if torch.is_tensor(v):
                    cands.append(v)
                elif isinstance(v, (list, tuple)):
                    cands.extend([t for t in v if torch.is_tensor(t)])
        try:
            for v in obj:
                if torch.is_tensor(v):
                    cands.append(v)
        except Exception:
            pass
        return cands

    # ---- dict-like output ----
    if isinstance(pmnet_attr, dict):
        keys = list(pmnet_attr.keys())

        pos = None
        for k in ["pos", "xyz", "coords", "positions"]:
            v = pmnet_attr.get(k, None)
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] == 3:
                pos = v
                break
        if pos is None:
            for k in keys:
                v = pmnet_attr[k]
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] == 3:
                    pos = v
                    break
        if pos is None:
            raise ValueError(f"Could not find pocket positions in pmnet_attr keys={keys}")

        x = None
        for k in keys:
            v = pmnet_attr[k]
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == pos.shape[0] and v.shape[1] == 192:
                x = v
                break
        if x is None:
            for k in ["x", "h", "feat", "features", "emb", "embedding"]:
                v = pmnet_attr.get(k, None)
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == pos.shape[0]:
                    x = v
                    break
        if x is None:
            raise ValueError("Could not find pocket x/embedding (need [Np,192]).")

        typ = None
        for k in ["typ", "type", "types"]:
            v = pmnet_attr.get(k, None)
            if torch.is_tensor(v) and v.ndim == 1 and v.shape[0] == pos.shape[0]:
                typ = v
                break
        if typ is None:
            typ = torch.argmax(x[:, :11], dim=1) if x.shape[1] >= 11 else torch.zeros((pos.shape[0],), dtype=torch.long)

        score = None
        for k in ["score", "scores", "w", "weight", "weights"]:
            v = pmnet_attr.get(k, None)
            if torch.is_tensor(v) and v.ndim == 1 and v.shape[0] == pos.shape[0]:
                score = v
                break
        if score is None:
            score = torch.ones((pos.shape[0],), dtype=torch.float32)

    # ---- dataclass-like PMNetAttr ----
    elif hasattr(pmnet_attr, "hotspots") and hasattr(pmnet_attr, "multi_scale_features"):
        hotspots = list(getattr(pmnet_attr, "hotspots"))
        N = len(hotspots)
        if N == 0:
            raise ValueError("PMNetAttr.hotspots is empty; cannot build pocket tensors.")

        pos_list: List[torch.Tensor] = []
        typ_list: List[int] = []
        score_list: List[float] = []
        x_list: List[torch.Tensor] = []

        pos_keys = ["pos", "xyz", "coord", "coords", "center", "centroid", "position", "location", "point"]
        typ_keys = ["typ", "type", "cls", "class_id", "label", "family", "feature_type"]
        score_keys = ["score", "scores", "w", "weight", "weights", "confidence", "prob", "p"]
        x_keys = ["x", "h", "feat", "features", "emb", "embedding", "vector"]

        for hs in hotspots:
            hs_state = None
            if hasattr(hs, "to_state"):
                try:
                    hs_state = hs.to_state()
                except Exception:
                    hs_state = None

            if isinstance(hs_state, dict):
                d = hs_state
            else:
                d = {}
                if hasattr(hs, "__dict__"):
                    d.update(vars(hs))

            p = _as_tensor_3(_get_first_from_mapping(d, pos_keys))
            if p is None:
                for k in pos_keys:
                    p = _as_tensor_3(getattr(hs, k, None))
                    if p is not None:
                        break
            if p is None:
                raise ValueError(f"Could not extract hotspot position from fields: {list(d.keys())}")
            pos_list.append(p)

            t = _as_int(_get_first_from_mapping(d, typ_keys))
            if t is None:
                for k in typ_keys:
                    t = _as_int(getattr(hs, k, None))
                    if t is not None:
                        break
            if t is not None:
                typ_list.append(t)

            s = _as_float(_get_first_from_mapping(d, score_keys))
            if s is None:
                for k in score_keys:
                    s = _as_float(getattr(hs, k, None))
                    if s is not None:
                        break
            if s is not None:
                score_list.append(s)

            xv = _get_first_from_mapping(d, x_keys)
            if xv is None:
                for k in x_keys:
                    xv = getattr(hs, k, None)
                    if xv is not None:
                        break
            if torch.is_tensor(xv) and xv.ndim == 1:
                x_list.append(xv.detach().cpu())

        pos = torch.stack(pos_list, dim=0).to(torch.float32)

        x = None
        ms = getattr(pmnet_attr, "multi_scale_features", None)
        cands = _iter_candidate_tensors(ms)
        for t in cands:
            if t.ndim == 2 and t.shape[0] == N and t.shape[1] == 192:
                x = t
                break
        if x is None:
            for t in cands:
                if t.ndim == 2 and t.shape[0] == N:
                    x = t
                    break
        if x is None and len(x_list) == N:
            x = torch.stack([v.to(torch.float32) for v in x_list], dim=0)
        if x is None:
            raise ValueError("Could not find pocket x/embedding.")

        if x.shape[1] != 192:
            if x.shape[1] > 192:
                x = x[:, :192]
            else:
                pad = torch.zeros((N, 192 - x.shape[1]), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)

        if len(typ_list) == N:
            typ = torch.tensor(typ_list, dtype=torch.long)
        else:
            typ = torch.argmax(x[:, :11], dim=1).to(torch.long) if x.shape[1] >= 11 else torch.zeros((N,), dtype=torch.long)

        if len(score_list) == N:
            score = torch.tensor(score_list, dtype=torch.float32)
        else:
            score = torch.ones((N,), dtype=torch.float32)

    else:
        raise ValueError(f"Unsupported pmnet_attr type: {type(pmnet_attr)}")

    # cap nodes
    if max_nodes and pos.shape[0] > max_nodes:
        pos = pos[:max_nodes]
        x = x[:max_nodes]
        typ = typ[:max_nodes]
        score = score[:max_nodes]

    out = {
        "pos": pos.detach().cpu().half(),
        "x": x.detach().cpu().half(),
        "typ": typ.detach().cpu().long(),
        "score": score.detach().cpu().half(),
    }
    # logger.debug("Pocket tensors: pos=%s x=%s typ=%s score=%s",
    #              tuple(out["pos"].shape), tuple(out["x"].shape),
    #              tuple(out["typ"].shape), tuple(out["score"].shape))
    return out


@torch.inference_mode()
def extract_pmnet_pocket(
    pmnet_obj,
    parser: ProteinParser,
    protein_file: Path,
    ligand_ref_sdf: Path,
    max_nodes: int = 256,
) -> Dict[str, torch.Tensor]:
    protein_data = parser.parse(str(protein_file), str(ligand_ref_sdf), None)
    pmnet_attr = pmnet_obj.run_extraction(protein_data)
    return pmnet_hotspots_to_tensors(pmnet_attr, max_nodes=max_nodes)


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p and p.exists():
            return p
    return None


def resolve_inputs(
    target_dir: Path,
    target_name: str,
    receptor_pdb: Optional[Path],
    crystal_mol2: Optional[Path],
    actives_mol2_gz: Optional[Path],
    decoys_mol2_gz: Optional[Path],
) -> Tuple[Path, Path, Path, Path, str]:
    """
    Resolve input paths supporting both NEW and OLD layouts.
    Returns: (receptor_pdb, crystal_mol2, actives_mol2_gz, decoys_mol2_gz, resolved_target_name)
    """
    target_dir = target_dir.resolve()
    resolved_name = target_name or target_dir.name

    # receptor
    if receptor_pdb is None:
        receptor_pdb = _first_existing([
            target_dir / "receptor.pdb",              # new
            target_dir / "rec.pdb",                   # some variants
            target_dir / "targets" / resolved_name / "receptor.pdb",
            target_dir / "targets" / resolved_name / "rec.pdb",   # old
        ])
    receptor_pdb = Path(receptor_pdb) if receptor_pdb is not None else None
    if receptor_pdb is None or not receptor_pdb.exists():
        raise FileNotFoundError(
            "Could not find receptor PDB. Looked for receptor.pdb / rec.pdb in new/old layouts. "
            "Pass --receptor_pdb to override."
        )

    # crystal ligand
    if crystal_mol2 is None:
        crystal_mol2 = _first_existing([
            target_dir / "crystal_ligand.mol2",       # new
            target_dir / "xtal-lig.mol2",             # old
            target_dir / "targets" / resolved_name / "xtal-lig.mol2",
            target_dir / "targets" / resolved_name / "crystal_ligand.mol2",
        ])
    crystal_mol2 = Path(crystal_mol2) if crystal_mol2 is not None else None
    if crystal_mol2 is None or not crystal_mol2.exists():
        raise FileNotFoundError(
            "Could not find crystal ligand MOL2. Looked for crystal_ligand.mol2 / xtal-lig.mol2. "
            "Pass --crystal_mol2 to override."
        )

    # actives mol2.gz
    if actives_mol2_gz is None:
        actives_mol2_gz = _first_existing([
            target_dir / "actives_final.mol2.gz",                         # new
            target_dir / "actives.mol2.gz",
            target_dir / "databases" / "dud_ligands2006" / f"{resolved_name}_ligands.mol2.gz",  # old
            target_dir / "databases" / "dud_ligands2006" / f"{resolved_name}_actives.mol2.gz",
        ])
    actives_mol2_gz = Path(actives_mol2_gz) if actives_mol2_gz is not None else None
    if actives_mol2_gz is None or not actives_mol2_gz.exists():
        raise FileNotFoundError(
            "Could not find actives MOL2.GZ. Looked for actives_final.mol2.gz and old dud_ligands2006 paths. "
            "Pass --actives_mol2_gz to override."
        )

    # decoys mol2.gz
    if decoys_mol2_gz is None:
        decoys_mol2_gz = _first_existing([
            target_dir / "decoys_final.mol2.gz",                          # new
            target_dir / "decoys.mol2.gz",
            target_dir / "databases" / "dud_decoys2006" / f"{resolved_name}_decoys.mol2.gz",  # old
            target_dir / "databases" / "dud_decoys2006" / f"{resolved_name}_decoys2006.mol2.gz",
        ])
    decoys_mol2_gz = Path(decoys_mol2_gz) if decoys_mol2_gz is not None else None
    if decoys_mol2_gz is None or not decoys_mol2_gz.exists():
        raise FileNotFoundError(
            "Could not find decoys MOL2.GZ. Looked for decoys_final.mol2.gz and old dud_decoys2006 paths. "
            "Pass --decoys_mol2_gz to override."
        )

    return receptor_pdb, crystal_mol2, actives_mol2_gz, decoys_mol2_gz, resolved_name


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True, help="Path to the target folder (e.g. .../ace). Works for new or old layouts.")
    ap.add_argument("--target_name", default="", help="Optional name; if omitted, inferred from folder name.")

    # Overrides (optional)
    ap.add_argument("--receptor_pdb", default="", help="Override receptor pdb path")
    ap.add_argument("--crystal_mol2", default="", help="Override crystal ligand mol2 path")
    ap.add_argument("--actives_mol2_gz", default="", help="Override actives mol2.gz path")
    ap.add_argument("--decoys_mol2_gz", default="", help="Override decoys mol2.gz path")

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max_pocket_nodes", type=int, default=256)
    ap.add_argument("--max_lig_nodes", type=int, default=48)

    ap.add_argument("--limit_actives", type=int, default=0)
    ap.add_argument("--limit_decoys", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    target_dir = Path(args.target_dir)

    receptor_pdb = Path(args.receptor_pdb) if args.receptor_pdb else None
    crystal_mol2 = Path(args.crystal_mol2) if args.crystal_mol2 else None
    actives_mol2_gz = Path(args.actives_mol2_gz) if args.actives_mol2_gz else None
    decoys_mol2_gz = Path(args.decoys_mol2_gz) if args.decoys_mol2_gz else None

    rec_pdb, xtal_mol2, act_mol2gz, dec_mol2gz, target_name = resolve_inputs(
        target_dir=target_dir,
        target_name=args.target_name,
        receptor_pdb=receptor_pdb,
        crystal_mol2=crystal_mol2,
        actives_mol2_gz=actives_mol2_gz,
        decoys_mol2_gz=decoys_mol2_gz,
    )

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    sdf_dir = out_dir / "sdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert crystal MOL2 to SDF for ProteinParser ---
    xtal_sdf = out_dir / "_xtal_ref.sdf"
    if (not xtal_sdf.exists()) or args.overwrite:
        mol2_to_sdf(xtal_mol2, xtal_sdf)

    # --- Init PMNet ---
    pmnet_obj = get_pmnet_dev()
    # if args.device == "cuda":
    #     if torch.cuda.is_available() and hasattr(pmnet_obj, "to"):
    #         pmnet_obj = pmnet_obj.to("cuda")
    #     if hasattr(pmnet_obj, "eval"):
    #         pmnet_obj.eval()

    parser = ProteinParser()

    # --- Extract pocket once ---
    pocket_pharm = extract_pmnet_pocket(
        pmnet_obj=pmnet_obj,
        parser=parser,
        protein_file=rec_pdb,
        ligand_ref_sdf=xtal_sdf,
        max_nodes=int(args.max_pocket_nodes),
    )

    # --- Build query.pt from crystal ligand pose ---
    query_m = Chem.MolFromMol2File(str(xtal_mol2), sanitize=False, removeHs=False)
    if query_m is None:
        raise ValueError(f"RDKit failed reading {xtal_mol2}")
    try:
        Chem.SanitizeMol(query_m)
    except Exception:
        pass
    if query_m.GetNumConformers() < 1:
        raise ValueError(f"Crystal ligand {xtal_mol2} has no 3D coords (won't generate).")

    query_lig = lig_pharm_from_mol(query_m, confId=0, max_nodes=int(args.max_lig_nodes))
    query_item = {
        "pocket_pharm": pocket_pharm,
        "lig_pharm": {
            "pos": query_lig["pos"].half(),
            "x": query_lig["x"].half(),
            "typ": query_lig["typ"].long(),
            "score": query_lig["score"].half(),
        },
        "meta": {
            "target": target_name,
            "split": "query",
            "protein_file": str(rec_pdb),
            "ligand_file": str(xtal_sdf),
            "ligand_ref_mol2": str(xtal_mol2),
            "smiles": Chem.MolToSmiles(query_m, isomericSmiles=True),
        },
    }
    safe_torch_save(query_item, out_dir / "query.pt")

    manifest_path = out_dir / "manifest.csv"
    stats = {"ok": 0, "fail": 0, "actives": 0, "decoys": 0}

    def process_split(split: str, mol2gz: Path, limit: int, start_idx: int, writer: csv.writer) -> int:
        nonlocal stats
        idx = start_idx

        it = iter_mol2_blocks(mol2gz)
        pbar = tqdm(it, desc=f"Caching {split}", unit="mol", dynamic_ncols=True)

        for rec_idx, block, lig_name in pbar:
            if limit and limit > 0 and stats[split] >= limit:
                break

            out_pt = cache_dir / f"{idx:08d}.pt"
            out_sdf = sdf_dir / f"{idx:08d}.sdf"

            if out_pt.exists() and (not args.overwrite):
                stats[split] += 1
                idx += 1
                continue

            try:
                mol = mol_from_mol2_block(block)

                # Save per-ligand SDF (pose preserved). No conformer generation.
                write_single_sdf(mol, out_sdf)

                lig = lig_pharm_from_mol(mol, confId=0, max_nodes=int(args.max_lig_nodes))
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)

                item = {
                    "pocket_pharm": pocket_pharm,  # duplicated for compatibility
                    "lig_pharm": {
                        "pos": lig["pos"].half(),
                        "x": lig["x"].half(),
                        "typ": lig["typ"].long(),
                        "score": lig["score"].half(),
                    },
                    "meta": {
                        "target": target_name,
                        "split": split,
                        "protein_file": str(rec_pdb),
                        "ligand_file": str(out_sdf),
                        "source_mol2_gz": str(mol2gz),
                        "source_record_idx": int(rec_idx),
                        "ligand_name": lig_name,
                        "smiles": smi,
                    },
                }
                safe_torch_save(item, out_pt)

                stats["ok"] += 1
                stats[split] += 1
                idx += 1
                pbar.set_postfix(ok=stats["ok"], fail=stats["fail"])

                writer.writerow([out_pt.name, split, lig_name, rec_idx, str(out_sdf), smi])

            except Exception:
                stats["fail"] += 1
                pbar.set_postfix(ok=stats["ok"], fail=stats["fail"])
                print(f"\n[ERROR] split={split} rec_idx={rec_idx} name={lig_name}", flush=True)
                traceback.print_exc()
                idx += 1

        return idx

    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cache_file", "split", "ligand_name", "source_record_idx", "ligand_sdf", "smiles"])

        idx0 = 0
        idx0 = process_split("actives", act_mol2gz, int(args.limit_actives), idx0, w)
        idx0 = process_split("decoys", dec_mol2gz, int(args.limit_decoys), idx0, w)

    report = {
        "target": target_name,
        "target_dir": str(target_dir),
        "rec_pdb": str(rec_pdb),
        "xtal_mol2": str(xtal_mol2),
        "xtal_sdf": str(xtal_sdf),
        "actives_mol2_gz": str(act_mol2gz),
        "decoys_mol2_gz": str(dec_mol2gz),
        "out_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "sdf_dir": str(sdf_dir),
        "stats": stats,
        "pocket_nodes": int(pocket_pharm["pos"].shape[0]),
        "note": "No conformers are generated; molecules must have 3D coordinates in the input MOL2 blocks.",
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
