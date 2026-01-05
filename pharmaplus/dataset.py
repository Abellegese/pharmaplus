from pathlib import Path

import torch
from torch.utils.data import Dataset
from pharmaplus.helpers import safe_torch_load


class PharmCache(Dataset):
  def __init__(self, files):
    self.files = list(files)
    if not self.files:
      raise RuntimeError("Empty dataset")

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    fp = self.files[idx]
    d = safe_torch_load(fp)
    P = d["pocket_pharm"]
    L = d["lig_pharm"]
    meta = d.get("meta", {})

    pN = P["pos"].shape[0]
    lN = L["pos"].shape[0]

    return {
      "fp": str(fp),
      "p_pos": P["pos"].float(),  # [Np,3]
      "p_x": P["x"].float(),  # [Np,192]
      "p_typ": P.get("typ", torch.zeros(pN, dtype=torch.long)).long(),
      "p_score": P.get("score", torch.zeros(pN)).float(),
      "p_rad": P.get("rad", torch.zeros(pN)).float(),
      "l_pos": L["pos"].float(),  # [Nl,3]
      "l_x": L["x"].float(),  # [Nl,8]
      "l_typ": L.get("typ", torch.zeros(lN, dtype=torch.long)).long(),
      "pdbid": meta.get("pdbid", Path(fp).stem),
    }


def collate_pad(batch):
  B = len(batch)
  pN = max(x["p_pos"].shape[0] for x in batch)
  lN = max(x["l_pos"].shape[0] for x in batch)
  pxd = batch[0]["p_x"].shape[1]
  lxd = batch[0]["l_x"].shape[1]

  def pad2(t, N, D):
    out = torch.zeros((N, D), dtype=t.dtype)
    out[: t.shape[0]] = t
    return out

  def pad1(t, N):
    out = torch.zeros((N,), dtype=t.dtype)
    out[: t.shape[0]] = t
    return out

  p_pos = torch.stack([pad2(x["p_pos"], pN, 3) for x in batch], 0)
  p_x = torch.stack([pad2(x["p_x"], pN, pxd) for x in batch], 0)
  p_typ = torch.stack([pad1(x["p_typ"], pN) for x in batch], 0)
  p_sc = torch.stack([pad1(x["p_score"], pN) for x in batch], 0)
  p_rad = torch.stack([pad1(x["p_rad"], pN) for x in batch], 0)

  l_pos = torch.stack([pad2(x["l_pos"], lN, 3) for x in batch], 0)
  l_x = torch.stack([pad2(x["l_x"], lN, lxd) for x in batch], 0)
  l_typ = torch.stack([pad1(x["l_typ"], lN) for x in batch], 0)

  p_mask = torch.stack(
    [torch.arange(pN) < x["p_pos"].shape[0] for x in batch], 0
  )  # [B,pN]
  l_mask = torch.stack(
    [torch.arange(lN) < x["l_pos"].shape[0] for x in batch], 0
  )  # [B,lN]

  return {
    "p_pos": p_pos,
    "p_x": p_x,
    "p_typ": p_typ,
    "p_score": p_sc,
    "p_rad": p_rad,
    "p_mask": p_mask,
    "l_pos": l_pos,
    "l_x": l_x,
    "l_typ": l_typ,
    "l_mask": l_mask,
    "pdbid": [x["pdbid"] for x in batch],
    "fps": [x["fp"] for x in batch],
  }
