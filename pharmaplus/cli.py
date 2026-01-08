from __future__ import annotations

import argparse, json, random, sys, rich, time, torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from torch.utils.data import DataLoader

from pharmaplus.helpers import (
  list_pt,
  split_files,
  viz_args_to_argv,
)
from pharmaplus.dataset import PharmCache, collate_pad
from pharmaplus.model import PharmMatchNetFast
from pharmaplus.train import train_one_epoch, evaluate
from pharmaplus import visualzie


def _now() -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S")


def seed_everything(seed: int, deterministic: bool = False) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> torch.device:
  if device_str == "auto":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
  return torch.device(device_str)


def make_loaders(
  args, cache_dir: Path
) -> Tuple[DataLoader, DataLoader, List[Path], List[Path]]:
  files = list_pt(cache_dir)
  tr_files, va_files = split_files(files, args.val_frac, args.seed)

  tr_ds = PharmCache(tr_files)
  va_ds = PharmCache(va_files)

  tr_dl = DataLoader(
    tr_ds,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.workers,
    pin_memory=not args.no_pin_memory,
    collate_fn=collate_pad,
  )
  va_dl = DataLoader(
    va_ds,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=args.workers,
    pin_memory=not args.no_pin_memory,
    collate_fn=collate_pad,
  )
  return tr_dl, va_dl, tr_files, va_files


def build_model(args, device: torch.device) -> PharmMatchNetFast:
  model = PharmMatchNetFast(
    d=args.d_model,
    retr_d=args.retr_d,
    topk=args.topk,
    edge_pairs=args.edge_pairs,
    dustbin=args.dustbin,
    edge_scale=args.edge_scale,
  ).to(device)
  if args.compile and hasattr(torch, "compile"):
    model = torch.compile(model)
  return model


def make_scaler(enabled: bool) -> Any:
  try:
    return torch.amp.GradScaler("cuda", enabled=enabled)
  except TypeError:
    return torch.cuda.amp.GradScaler(enabled=enabled)


def save_ckpt(
  path: Path,
  *,
  model,
  opt,
  scaler,
  args,
  epoch: int,
  step: int,
  best_score: float,
  val_metrics: Dict[str, float],
  tr_files: Optional[List[Path]] = None,
  va_files: Optional[List[Path]] = None,
) -> None:
  ckpt = {
    "model": model.state_dict(),
    "opt": opt.state_dict() if opt is not None else None,
    "scaler": scaler.state_dict() if scaler is not None else None,
    "args": vars(args),
    "epoch": epoch,
    "step": step,
    "best_score": best_score,
    "val_metrics": val_metrics,
    "tr_files": [str(p) for p in tr_files] if tr_files is not None else None,
    "va_files": [str(p) for p in va_files] if va_files is not None else None,
    "saved_at": _now(),
  }
  torch.save(ckpt, path)


def load_ckpt(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
  return torch.load(path, map_location=map_location)


def _render_viz_rich_table(summary: dict):
  from rich.table import Table
  from rich.text import Text

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

  rows = [row("GT", "-", gt)]
  if len(tops) >= 1:
    rows.append(row("Conf1", tops[0].get("conf_id", "-"), tops[0]))
  if len(tops) >= 2:
    rows.append(row("Conf2", tops[1].get("conf_id", "-"), tops[1]))
  if other and isinstance(other, dict):
    best = other.get("best", {}) or {}
    rows.append(row("Other(best)", other.get("best_conf", "-"), best))

  sretrs = [r["sretr"] for r in rows if isinstance(r.get("sretr"), (int, float))]
  best_sretr = max(sretrs) if sretrs else None

  def fmt(x: object, nd: int = 4) -> str:
    if x is None:
      return "-"
    try:
      return f"{float(x):.{nd}f}"
    except Exception:
      return str(x)

  title = f"Visualization Summary (lambda_retr={fmt(lam, 3)})"

  table = Table(title=title, show_lines=False, header_style="bold")
  table.add_column("Candidate", style="bold")
  table.add_column("conf_id", justify="right")
  table.add_column("score_inv_geom", justify="right")
  table.add_column("retr_sim", justify="right")
  table.add_column("score_retr", justify="right")
  table.add_column("pose_aligned", justify="right")
  table.add_column("sharp", justify="right")
  table.add_column("-ent", justify="right")
  table.add_column("edge", justify="right")
  table.add_column("pose_term", justify="right")
  table.add_column("ref_term", justify="right")

  for r in rows:
    sretr_val = r["sretr"]
    sretr_text = Text(fmt(sretr_val, 4))

    if (
      best_sretr is not None
      and isinstance(sretr_val, (int, float))
      and abs(sretr_val - best_sretr) < 1e-12
    ):
      sretr_text.stylize("bold bright_green")
    elif isinstance(sretr_val, (int, float)):
      sretr_text.stylize("bright_white")

    table.add_row(
      r["name"],
      str(r["conf_id"]),
      fmt(r["inv"], 4),
      fmt(r["retr"], 4),
      sretr_text,
      fmt(r["pose"], 4),
      fmt(r["sharp"], 4),
      fmt(r["neg_ent"], 4),
      fmt(r["edge"], 4),
      fmt(r["pose_t"], 4),
      fmt(r["ref_t"], 4),
    )

  return table


def cmd_train(args: argparse.Namespace) -> int:
  cache_dir = Path(args.cache_dir)
  if not cache_dir.exists():
    raise SystemExit(f"[error] cache-dir does not exist: {cache_dir}")

  seed_everything(args.seed, deterministic=args.deterministic)

  try:
    torch.set_float32_matmul_precision("high")
  except Exception:
    pass

  device = resolve_device(args.device)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  tr_dl, va_dl, tr_files, va_files = make_loaders(args, cache_dir)

  model = build_model(args, device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
  scaler = make_scaler(enabled=bool(args.amp and device.type == "cuda"))

  step = 0
  best_score = -1e9
  start_epoch = 0

  if args.resume:
    ckpt = load_ckpt(Path(args.resume), map_location="cpu")
    print(f"[{_now()}] [resume] loading {args.resume}")
    if "model" in ckpt:
      model.load_state_dict(ckpt["model"], strict=True)
    if ckpt.get("opt") and args.resume_opt:
      try:
        opt.load_state_dict(ckpt["opt"])
      except Exception as e:
        print(f"[{_now()}] [resume] warning: could not load optimizer state: {e}")
    if ckpt.get("scaler") and args.resume_opt:
      try:
        scaler.load_state_dict(ckpt["scaler"])
      except Exception as e:
        print(f"[{_now()}] [resume] warning: could not load scaler state: {e}")

    step = int(ckpt.get("step", 0))
    best_score = float(ckpt.get("best_score", best_score))
    start_epoch = int(ckpt.get("epoch", -1)) + 1

    model.to(device)

  print(f"[{_now()}] device={device} out_dir={out_dir}")
  if args.print_args:
    print(json.dumps(vars(args), indent=2, sort_keys=True))

  for epoch in range(start_epoch, args.epochs):
    if args.max_train_steps and step >= args.max_train_steps:
      print(f"[{_now()}] reached --max-train-steps={args.max_train_steps}, stopping.")
      break

    step = train_one_epoch(model, tr_dl, opt, scaler, device, args, step0=step)

    metrics = evaluate(model, va_dl, device, args, max_batches=args.eval_batches)
    score = metrics["retr_top1"] + metrics["pose_acc"] + metrics["xpose_acc"]

    print(
      f"[{_now()}] [val] epoch {epoch:02d} "
      f"pose_acc {metrics['pose_acc'] * 100:.1f}% "
      f"xpose_acc {metrics['xpose_acc'] * 100:.1f}% | "
      f"retr_top1 {metrics['retr_top1'] * 100:.1f}% "
      f"mrr {metrics['retr_mrr']:.3f} mean_rank {metrics['retr_mean_rank']:.2f} "
      f"retr_auc {metrics['retr_auc']:.3f}"
    )

    if not args.no_save:
      save_ckpt(
        out_dir / f"epoch{epoch:02d}.pt",
        model=model,
        opt=opt,
        scaler=scaler,
        args=args,
        epoch=epoch,
        step=step,
        best_score=best_score,
        val_metrics=metrics,
        tr_files=tr_files,
        va_files=va_files,
      )

    if score > best_score:
      best_score = score
      if not args.no_save:
        save_ckpt(
          out_dir / "best.pt",
          model=model,
          opt=opt,
          scaler=scaler,
          args=args,
          epoch=epoch,
          step=step,
          best_score=best_score,
          val_metrics=metrics,
          tr_files=tr_files,
          va_files=va_files,
        )
      print(
        f"[{_now()}] [ckpt] new best -> {out_dir / 'best.pt'} (score={best_score:.4f})"
      )

  print(f"[{_now()}] done")
  return 0


def cmd_eval(args: argparse.Namespace) -> int:
  cache_dir = Path(args.cache_dir)
  ckpt_path = Path(args.ckpt)

  if not cache_dir.exists():
    raise SystemExit(f"[error] cache-dir does not exist: {cache_dir}")
  if not ckpt_path.exists():
    raise SystemExit(f"[error] ckpt does not exist: {ckpt_path}")

  device = resolve_device(args.device)

  ckpt = load_ckpt(ckpt_path, map_location="cpu")

  ckpt_args = ckpt.get("args", {}) or {}
  for k in ["d_model", "retr_d", "topk", "edge_pairs", "dustbin", "edge_scale"]:
    if getattr(args, k, None) is None and k in ckpt_args:
      setattr(args, k, ckpt_args[k])

  model = build_model(args, device)
  model.load_state_dict(ckpt["model"], strict=True)
  model.to(device)

  seed_everything(args.seed, deterministic=args.deterministic)
  try:
    torch.set_float32_matmul_precision("high")
  except Exception:
    pass

  _tr_dl, va_dl, _tr_files, _va_files = make_loaders(args, cache_dir)

  metrics = evaluate(model, va_dl, device, args, max_batches=args.eval_batches)

  if args.json:
    print(json.dumps(metrics, indent=2, sort_keys=True))
  else:
    print(
      f"[{_now()}] eval: "
      f"pose_acc {metrics['pose_acc'] * 100:.1f}% "
      f"xpose_acc {metrics['xpose_acc'] * 100:.1f}% | "
      f"retr_top1 {metrics['retr_top1'] * 100:.1f}% "
      f"mrr {metrics['retr_mrr']:.3f} mean_rank {metrics['retr_mean_rank']:.2f} "
      f"retr_auc {metrics['retr_auc']:.3f}"
    )
  return 0


def cmd_visualize(args: argparse.Namespace) -> int:
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

  old_argv = sys.argv[:]
  try:
    sys.argv = ["pharmaplus-visualize"] + viz_args_to_argv(args)
    visualzie.main()
  finally:
    sys.argv = old_argv

  return 0


def add_common_data_args(p: argparse.ArgumentParser) -> None:
  g = p.add_argument_group("data")
  g.add_argument(
    "--cache-dir", required=True, help="Directory containing cached .pt files"
  )
  g.add_argument("--val-frac", type=float, default=0.15)
  g.add_argument("--seed", type=int, default=0)

  g2 = p.add_argument_group("dataloader")
  g2.add_argument("--batch-size", type=int, default=64)
  g2.add_argument("--workers", type=int, default=6)
  g2.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory")


def add_common_runtime_args(p: argparse.ArgumentParser) -> None:
  g = p.add_argument_group("runtime")
  g.add_argument(
    "--device", default="auto", help='Device like "auto", "cpu", "cuda", "cuda:0"'
  )
  g.add_argument("--amp", action="store_true")
  g.add_argument("--compile", action="store_true")
  g.add_argument("--enable-pose", action="store_true")
  g.add_argument(
    "--deterministic", action="store_true", help="Enable deterministic CuDNN (slower)"
  )


def add_model_args(p: argparse.ArgumentParser, allow_none: bool = False) -> None:
  g = p.add_argument_group("model")
  t_int = int
  t_float = float

  default_or_none = (lambda v: None) if allow_none else (lambda v: v)

  g.add_argument("--d-model", dest="d_model", type=t_int, default=default_or_none(128))
  g.add_argument("--retr-d", dest="retr_d", type=t_int, default=default_or_none(128))

  g.add_argument("--topk", type=t_int, default=default_or_none(4))
  g.add_argument("--edge-pairs", type=t_int, default=default_or_none(16))
  g.add_argument("--edge-scale", type=t_float, default=default_or_none(5.0))
  g.add_argument("--dustbin", action="store_true")


def add_aug_and_loss_args(p: argparse.ArgumentParser) -> None:
  g = p.add_argument_group("augmentation / negatives")
  g.add_argument("--aug-se3", action="store_true")
  g.add_argument("--eval-aug-se3", action="store_true")
  g.add_argument("--aug-trans", type=float, default=2.0)
  g.add_argument("--pose-neg-trans", type=float, default=4.0)

  g2 = p.add_argument_group("loss weights")
  g2.add_argument("--pose-w", type=float, default=1.0)
  g2.add_argument("--xpose-w", type=float, default=0.3)
  g2.add_argument("--retr-w", type=float, default=1.0)
  g2.add_argument("--invneg-w", type=float, default=0.2)
  g2.add_argument("--temp", type=float, default=0.07)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="pharmaplus",
    description="PharmaPlus training/evaluation CLI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_tr = sub.add_parser(
    "train",
    help="Train a model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  add_common_data_args(p_tr)
  add_common_runtime_args(p_tr)
  add_model_args(p_tr, allow_none=False)
  add_aug_and_loss_args(p_tr)

  g = p_tr.add_argument_group("optimizer")
  g.add_argument("--lr", type=float, default=1e-3)
  g.add_argument("--wd", type=float, default=1e-4)

  g2 = p_tr.add_argument_group("training")
  g2.add_argument("--epochs", type=int, default=10)
  g2.add_argument("--log-every", type=int, default=20)
  g2.add_argument("--eval-batches", type=int, default=10)
  g2.add_argument(
    "--max-train-steps", type=int, default=0, help="Stop after N steps (0=disabled)"
  )

  g3 = p_tr.add_argument_group("checkpointing")
  g3.add_argument("--out-dir", default="_runs_fast_retr")
  g3.add_argument(
    "--resume", type=str, default="", help="Path to checkpoint to resume from"
  )
  g3.add_argument(
    "--resume-opt", action="store_true", help="Also resume optimizer/scaler if present"
  )
  g3.add_argument("--no-save", action="store_true", help="Disable saving checkpoints")
  g3.add_argument(
    "--print-args", action="store_true", help="Print resolved args as JSON at start"
  )

  p_ev = sub.add_parser(
    "eval",
    help="Evaluate a checkpoint",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  add_common_data_args(p_ev)
  add_common_runtime_args(p_ev)
  add_model_args(p_ev, allow_none=True)
  add_aug_and_loss_args(p_ev)

  g = p_ev.add_argument_group("evaluation")
  g.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
  g.add_argument("--eval-batches", type=int, default=10)
  g.add_argument("--json", action="store_true", help="Print metrics as JSON")

  p_viz = sub.add_parser(
    "visualize",
    help="Run inference + write PyMOL assets, then print a comparison table",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  g = p_viz.add_argument_group("inputs")
  g.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
  g.add_argument(
    "--item-pt", dest="item_pt", required=True, help="Cache item .pt to visualize"
  )
  g.add_argument(
    "--other-dir",
    dest="other_dir",
    default=None,
    help="Optional other cache item to take a different ligand (SMILES)",
  )

  g = p_viz.add_argument_group("runtime")
  g.add_argument(
    "--device", default="auto", help='Device like "auto", "cpu", "cuda", "cuda:0"'
  )
  g.add_argument("--amp", action="store_true")

  g = p_viz.add_argument_group("conformers")
  g.add_argument("--which-record", dest="which_record", type=int, default=0)
  g.add_argument("--n-confs", dest="n_confs", type=int, default=50)
  g.add_argument("--seed", type=int, default=0)
  g.add_argument("--optimize", action="store_true")
  g.add_argument("--max-lig-nodes", dest="max_lig_nodes", type=int, default=48)

  g = p_viz.add_argument_group("ranking / overrides")
  g.add_argument(
    "--topk", type=int, default=None, help="Override topk (default: ckpt args)"
  )
  g.add_argument(
    "--edge-pairs",
    dest="edge_pairs",
    type=int,
    default=None,
    help="Override edge_pairs (default: ckpt args)",
  )
  g.add_argument(
    "--lambda-retr",
    dest="lambda_retr",
    type=float,
    default=0.8,
    help="score_retr = score_inv_geom + lambda_retr * retr_sim",
  )

  g = p_viz.add_argument_group("match lines")
  g.add_argument("--match-topn", dest="match_topn", type=int, default=50)
  g.add_argument("--match-thresh", dest="match_thresh", type=float, default=0.01)

  g = p_viz.add_argument_group("outputs")
  g.add_argument("--out-dir", dest="out_dir", required=True)
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  args = build_parser().parse_args(argv)
  if args.cmd == "visualize":
    if args.device == "auto":
      args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cmd_visualize(args)

  if args.cmd == "train":
    return cmd_train(args)
  if args.cmd == "eval":
    return cmd_eval(args)

  raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
  raise SystemExit(main())
