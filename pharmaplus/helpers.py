import argparse, math, os, random, torch
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

LOG_2PI = math.log(2.0 * math.pi)


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


def _fmt(x: object, nd: int = 4) -> str:
  if x is None:
    return "-"
  try:
    return f"{float(x):.{nd}f}"
  except Exception:
    return str(x)


def viz_args_to_argv(args: argparse.Namespace) -> list[str]:
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


def print_batch_results_rich_table(
  results: List[Tuple[float, Path, Optional[Path], Dict[str, Any]]],
  *,
  top_n: Optional[int] = None,
  title: str = "Batch compare (other ligands on same pocket)",
) -> None:
  from rich.console import Console
  from rich.table import Table
  from rich.text import Text

  console = Console()

  if not results:
    console.print("[yellow]No results to display.[/yellow]")
    return

  rows = results[:top_n] if (top_n is not None and top_n > 0) else results

  scores = []
  for sretr, _pt, _sdf, best in rows:
    if (
      isinstance(best, dict) and "error" not in best and isinstance(sretr, (int, float))
    ):
      scores.append(float(sretr))
  best_score = max(scores) if scores else None

  def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
      return "-"
    try:
      return f"{float(x):.{nd}f}"
    except Exception:
      return str(x)

  table = Table(title=title, header_style="bold", show_lines=False)
  table.add_column("Rank", justify="right")
  table.add_column("Ligand path")
  table.add_column("best_conf", justify="right")
  table.add_column("score_inv_geom", justify="right")
  table.add_column("retr_sim", justify="right")
  table.add_column("score_retr", justify="right")
  table.add_column("sharp", justify="right")
  table.add_column("-ent", justify="right")
  table.add_column("edge", justify="right")
  table.add_column("status")

  for i, (sretr, pt, _sdf, best) in enumerate(rows, start=1):
    if isinstance(best, dict) and "error" in best:
      table.add_row(
        str(i),
        str(pt),
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        Text("ERROR", style="bold red"),
      )
      continue

    inv_terms = (best.get("inv_terms", {}) or {}) if isinstance(best, dict) else {}
    sretr_text = Text(fmt(best.get("score_retr", sretr), 4))

    if (
      best_score is not None
      and isinstance(sretr, (int, float))
      and abs(float(sretr) - float(best_score)) < 1e-12
    ):
      sretr_text.stylize("bold bright_green")

    table.add_row(
      str(i),
      str(pt),
      str(best.get("best_conf", "-")),
      fmt(best.get("score_inv_geom", None), 4),
      fmt(best.get("retr_sim", None), 4),
      sretr_text,
      fmt(inv_terms.get("sharp", None), 4),
      fmt(inv_terms.get("neg_ent", None), 4),
      fmt(inv_terms.get("edge", None), 4),
      Text("OK", style="green"),
    )

  console.print(table)
