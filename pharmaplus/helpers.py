import argparse, math, random, torch
from pathlib import Path

LOG_2PI = math.log(2.0 * math.pi)


def safe_torch_load(fp: Path):
  try:
    return torch.load(fp, map_location="cpu", weights_only=False)  # torch>=2.4
  except TypeError:
    return torch.load(fp, map_location="cpu")


def list_pt(cache_dir: Path):
  return sorted([p for p in cache_dir.glob("*.pt") if p.is_file()])


def pocket_centroid(p_pos, p_mask):
  pm = p_mask.float().unsqueeze(-1)  # [B,N,1]
  denom = pm.sum(dim=1).clamp(min=1.0)  # [B,1]
  return (p_pos * pm).sum(dim=1) / denom  # [B,3]


def rand_rotmat(B, device):
  A = torch.randn(B, 3, 3, device=device)
  Q, _ = torch.linalg.qr(A)
  det = torch.det(Q)
  Q = torch.where(
    det[:, None, None] < 0,
    Q * torch.tensor([[-1, 1, 1]], device=device).view(1, 1, 3),
    Q,
  )
  return Q


def apply_se3(x, R, t, mask):
  # x: [B,N,3], R:[B,3,3], t:[B,3]
  y = torch.einsum("bnc,bcj->bnj", x, R.transpose(1, 2)) + t[:, None, :]
  return torch.where(mask[..., None], y, x)


def make_pose_negative(l_pos, l_mask, max_trans=4.0):
  B = l_pos.shape[0]
  device = l_pos.device
  R = rand_rotmat(B, device)
  t = (torch.rand(B, 3, device=device) * 2 - 1.0) * float(max_trans)
  return apply_se3(l_pos, R, t, l_mask)


def augment_se3_pair(p_pos, l_pos, p_mask, l_mask, max_trans=2.0):
  B = p_pos.shape[0]
  device = p_pos.device
  R = rand_rotmat(B, device)
  t = (torch.rand(B, 3, device=device) * 2 - 1.0) * float(max_trans)
  return apply_se3(p_pos, R, t, p_mask), apply_se3(l_pos, R, t, l_mask)


def gaussian_logpdf(
  d: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
  sigma = sigma.clamp(min=1e-4)
  z = (d - mu) / sigma
  return -0.5 * (z * z) - torch.log(sigma) - 0.5 * LOG_2PI


@torch.no_grad()
def auc_from_pos_neg(pos, neg):
  pos = pos.detach()
  neg = neg.detach()
  gt = (pos[:, None] > neg[None, :]).float().mean().item()
  eq = (pos[:, None] == neg[None, :]).float().mean().item()
  return gt + 0.5 * eq


@torch.no_grad()
def retrieval_metrics(score_mat):
  B = score_mat.shape[0]
  sorted_idx = torch.argsort(score_mat, dim=1, descending=True)
  target = torch.arange(B, device=score_mat.device)[:, None]
  rank = (sorted_idx == target).nonzero()[:, 1].float() + 1.0
  top1 = float((rank == 1).float().mean().item())
  mrr = float((1.0 / rank).mean().item())
  mean_rank = float(rank.mean().item())
  diag = torch.diag(score_mat)
  off = score_mat[~torch.eye(B, dtype=torch.bool, device=score_mat.device)]
  auc = auc_from_pos_neg(diag, off)
  return top1, mrr, mean_rank, auc


def split_files(files, val_frac, seed):
  rng = random.Random(seed)
  files = list(files)
  rng.shuffle(files)
  n_val = max(1, int(len(files) * val_frac))
  return files[n_val:], files[:n_val]


def move_batch(batch, device):
  for k in [
    "p_pos",
    "p_x",
    "p_typ",
    "p_score",
    "p_rad",
    "p_mask",
    "l_pos",
    "l_x",
    "l_typ",
    "l_mask",
  ]:
    batch[k] = batch[k].to(device, non_blocking=True)
  return batch


def derangement_shift(B, device):
  return (torch.arange(B, device=device) + 1) % B


def _fmt(x: object, nd: int = 4) -> str:
  if x is None:
    return "-"
  try:
    return f"{float(x):.{nd}f}"
  except Exception:
    return str(x)


def _render_viz_table(summary: dict) -> str:
  """
  Build a Markdown table comparing GT vs top conformers (and optional other).
  """
  lam = summary.get("ranking", {}).get("lambda_retr", None)

  gt = summary.get("GT", {}) or {}
  tops = summary.get("top_conformers", []) or []
  other = summary.get("other", None)

  def row(name: str, conf_id: object, block: dict) -> dict:
    inv_terms = block.get("inv_terms", {}) or {}
    pose_terms = block.get("pose_terms", {}) or {}
    return {
      "name": name,
      "conf_id": conf_id,
      "inv": block.get("score_inv_geom", None),
      "retr": block.get("retr_sim", None),
      "sretr": block.get("score_retr", None),
      "pose": block.get("pose_aligned", None),
      "sharp": inv_terms.get("sharp", None),
      "neg_ent": inv_terms.get("neg_ent", None),
      "edge": inv_terms.get("edge", None),
      "pose_t": pose_terms.get("pose", None),
      "ref_t": pose_terms.get("ref", None),
    }

  rows = []
  rows.append(row("GT", "-", gt))
  if len(tops) >= 1:
    rows.append(row("Conf1", tops[0].get("conf_id", "-"), tops[0]))
  if len(tops) >= 2:
    rows.append(row("Conf2", tops[1].get("conf_id", "-"), tops[1]))

  if other and isinstance(other, dict):
    best = other.get("best", {}) or {}
    rows.append(row("Other(best)", other.get("best_conf", "-"), best))

  # find best score_retr among rows (ignore None)
  sretrs = [r["sretr"] for r in rows if isinstance(r.get("sretr", None), (int, float))]
  best_sretr = max(sretrs) if sretrs else None

  def maybe_bold(val: object) -> str:
    s = _fmt(val, 4)
    if (
      best_sretr is not None
      and isinstance(val, (int, float))
      and abs(val - best_sretr) < 1e-12
    ):
      return f"**{s}**"
    return s

  header = []
  header.append(
    f"**lambda_retr** = {_fmt(lam, 3)}  (score_retr = inv + lambda_retr * retr_sim)"
  )
  header.append(f"item_pt: `{summary.get('item_pt', '')}`")
  header.append(f"ckpt: `{summary.get('ckpt_args', {})}`")

  md = []
  md.append("\n".join(header))
  md.append("")
  md.append(
    "| Candidate | conf_id | score_inv_geom | retr_sim | score_retr | pose_aligned | sharp | -ent | edge | pose_term | ref_term |"
  )
  md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

  for r in rows:
    md.append(
      "| {name} | {conf_id} | {inv} | {retr} | {sretr} | {pose} | {sharp} | {neg_ent} | {edge} | {pose_t} | {ref_t} |".format(
        name=r["name"],
        conf_id=r["conf_id"],
        inv=_fmt(r["inv"], 4),
        retr=_fmt(r["retr"], 4),
        sretr=maybe_bold(r["sretr"]),
        pose=_fmt(r["pose"], 4),
        sharp=_fmt(r["sharp"], 4),
        neg_ent=_fmt(r["neg_ent"], 4),
        edge=_fmt(r["edge"], 4),
        pose_t=_fmt(r["pose_t"], 4),
        ref_t=_fmt(r["ref_t"], 4),
      )
    )

  return "\n".join(md)


def _fmt(x: object, nd: int = 4) -> str:
  if x is None:
    return "-"
  try:
    return f"{float(x):.{nd}f}"
  except Exception:
    return str(x)


def _render_viz_table(summary: dict) -> str:
  """
  Build a Markdown table comparing GT vs top conformers (and optional other).
  """
  lam = summary.get("ranking", {}).get("lambda_retr", None)

  gt = summary.get("GT", {}) or {}
  tops = summary.get("top_conformers", []) or []
  other = summary.get("other", None)

  def row(name: str, conf_id: object, block: dict) -> dict:
    inv_terms = block.get("inv_terms", {}) or {}
    pose_terms = block.get("pose_terms", {}) or {}
    return {
      "name": name,
      "conf_id": conf_id,
      "inv": block.get("score_inv_geom", None),
      "retr": block.get("retr_sim", None),
      "sretr": block.get("score_retr", None),
      "pose": block.get("pose_aligned", None),
      "sharp": inv_terms.get("sharp", None),
      "neg_ent": inv_terms.get("neg_ent", None),
      "edge": inv_terms.get("edge", None),
      "pose_t": pose_terms.get("pose", None),
      "ref_t": pose_terms.get("ref", None),
    }

  rows = []
  rows.append(row("GT", "-", gt))
  if len(tops) >= 1:
    rows.append(row("Conf1", tops[0].get("conf_id", "-"), tops[0]))
  if len(tops) >= 2:
    rows.append(row("Conf2", tops[1].get("conf_id", "-"), tops[1]))

  if other and isinstance(other, dict):
    best = other.get("best", {}) or {}
    rows.append(row("Other(best)", other.get("best_conf", "-"), best))

  sretrs = [r["sretr"] for r in rows if isinstance(r.get("sretr", None), (int, float))]
  best_sretr = max(sretrs) if sretrs else None

  def maybe_bold(val: object) -> str:
    s = _fmt(val, 4)
    if (
      best_sretr is not None
      and isinstance(val, (int, float))
      and abs(val - best_sretr) < 1e-12
    ):
      return f"**{s}**"
    return s

  header = []
  header.append(
    f"**lambda_retr** = {_fmt(lam, 3)}  (score_retr = inv + lambda_retr * retr_sim)"
  )
  header.append(f"item_pt: `{summary.get('item_pt', '')}`")
  header.append(f"ckpt: `{summary.get('ckpt_args', {})}`")

  md = []
  md.append("\n".join(header))
  md.append("")
  md.append(
    "| Candidate | conf_id | score_inv_geom | retr_sim | score_retr | pose_aligned | sharp | -ent | edge | pose_term | ref_term |"
  )
  md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

  for r in rows:
    md.append(
      "| {name} | {conf_id} | {inv} | {retr} | {sretr} | {pose} | {sharp} | {neg_ent} | {edge} | {pose_t} | {ref_t} |".format(
        name=r["name"],
        conf_id=r["conf_id"],
        inv=_fmt(r["inv"], 4),
        retr=_fmt(r["retr"], 4),
        sretr=maybe_bold(r["sretr"]),
        pose=_fmt(r["pose"], 4),
        sharp=_fmt(r["sharp"], 4),
        neg_ent=_fmt(r["neg_ent"], 4),
        edge=_fmt(r["edge"], 4),
        pose_t=_fmt(r["pose_t"], 4),
        ref_t=_fmt(r["ref_t"], 4),
      )
    )

  return "\n".join(md)


def _viz_args_to_argv(args: argparse.Namespace) -> list[str]:
  argv: list[str] = []
  argv += ["--ckpt", str(args.ckpt)]
  argv += ["--item_pt", str(args.item_pt)]
  argv += ["--other_dir", str(args.other_dir)]

  argv += ["--device", str(args.device)]
  if args.amp:
    argv += ["--amp"]

  argv += ["--n_confs", str(args.n_confs)]
  argv += ["--seed", str(args.seed)]
  if args.optimize:
    argv += ["--optimize"]
  argv += ["--max_lig_nodes", str(args.max_lig_nodes)]

  if args.topk is not None:
    argv += ["--topk", str(args.topk)]
  if args.edge_pairs is not None:
    argv += ["--edge_pairs", str(args.edge_pairs)]
  argv += ["--lambda_retr", str(args.lambda_retr)]
  argv += ["--out_dir", str(args.out_dir)]
  return argv
