import torch
import torch.nn.functional as F
from typing import Dict

from pharmaplus.helpers import (
  move_batch,
  pocket_centroid,
  augment_se3_pair,
  make_pose_negative,
  derangement_shift,
  retrieval_metrics,
)


def train_one_epoch(model, dl, opt, scaler, device, args, step0: int = 0) -> int:
  model.train()
  step = step0

  enable_pose = bool(getattr(args, "enable_pose", True))
  enable_edge = bool(getattr(args, "enable_edge", True))

  for batch in dl:
    batch = move_batch(batch, device)

    # center by pocket centroid (like inference)
    cent = pocket_centroid(batch["p_pos"], batch["p_mask"])
    batch["p_pos"] = batch["p_pos"] - cent[:, None, :]
    batch["l_pos"] = batch["l_pos"] - cent[:, None, :]

    # shared SE3 augmentation (keeps relative pose)
    if getattr(args, "aug_se3", False):
      batch["p_pos"], batch["l_pos"] = augment_se3_pair(
        batch["p_pos"],
        batch["l_pos"],
        batch["p_mask"],
        batch["l_mask"],
        max_trans=getattr(args, "aug_trans", 0.0),
      )

    # pose negatives (ligand only)
    l_pos_bad = make_pose_negative(
      batch["l_pos"], batch["l_mask"], max_trans=getattr(args, "pose_neg_trans", 3.0)
    )

    opt.zero_grad(set_to_none=True)
    use_amp = bool(getattr(args, "amp", False) and device.type == "cuda")

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):

      stageA = model.precompute_stageA_batch(
        batch["l_pos"],
        batch["l_x"],
        batch["l_typ"],
        batch["l_mask"],
        batch["p_pos"],
        batch["p_x"],
        batch["p_typ"],
        batch["p_score"],
        batch["p_rad"],
        batch["p_mask"],
        topk=getattr(args, "topk", None),
        edge_pairs=getattr(args, "edge_pairs", None),
        enable_edge=enable_edge,
      )

      # Pose losses (optional)
      pose_loss = torch.tensor(0.0, device=device)
      xpose_loss = torch.tensor(0.0, device=device)
      s_pos = stageA["score_inv_geom"]
      s_neg_pose = stageA["score_inv_geom"]
      s_cross = stageA["score_inv_geom"]

      if enable_pose and (getattr(args, "pose_w", 0.0) > 0.0 or getattr(args, "xpose_w", 0.0) > 0.0):
        s_pos, _dbg = model.score_pose_from_stageA_batch(
          stageA, batch["l_pos"], batch["l_mask"], batch["p_pos"], batch["p_mask"], enable_pose=True
        )
        s_neg_pose, _ = model.score_pose_from_stageA_batch(
          stageA, l_pos_bad, batch["l_mask"], batch["p_pos"], batch["p_mask"], enable_pose=True
        )
        pose_loss = F.softplus(-(s_pos - s_neg_pose)).mean()

        # Cross-pocket negatives: ligand from other sample with this pocket
        perm = derangement_shift(batch["p_pos"].shape[0], device=device)
        stageA_cross = model.precompute_stageA_batch(
          batch["l_pos"][perm],
          batch["l_x"][perm],
          batch["l_typ"][perm],
          batch["l_mask"][perm],
          batch["p_pos"],
          batch["p_x"],
          batch["p_typ"],
          batch["p_score"],
          batch["p_rad"],
          batch["p_mask"],
          topk=getattr(args, "topk", None),
          edge_pairs=getattr(args, "edge_pairs", None),
          enable_edge=enable_edge,
        )
        s_cross, _ = model.score_pose_from_stageA_batch(
          stageA_cross,
          batch["l_pos"][perm],
          batch["l_mask"][perm],
          batch["p_pos"],
          batch["p_mask"],
          enable_pose=True,
        )
        xpose_loss = F.softplus(-(s_pos - s_cross)).mean()
      else:
        # still need stageA_cross for invneg/retrieval
        perm = derangement_shift(batch["p_pos"].shape[0], device=device)
        stageA_cross = model.precompute_stageA_batch(
          batch["l_pos"][perm],
          batch["l_x"][perm],
          batch["l_typ"][perm],
          batch["l_mask"][perm],
          batch["p_pos"],
          batch["p_x"],
          batch["p_typ"],
          batch["p_score"],
          batch["p_rad"],
          batch["p_mask"],
          topk=getattr(args, "topk", None),
          edge_pairs=getattr(args, "edge_pairs", None),
          enable_edge=enable_edge,
        )

      # Retrieval loss (InfoNCE) on pooled embeddings
      l_z = stageA["l_z"]
      p_z = stageA["p_z"]
      sim = (l_z @ p_z.t()) / float(getattr(args, "temp", 0.07))  # [B,B]
      y = torch.arange(sim.shape[0], device=device)
      retr_loss = 0.5 * (F.cross_entropy(sim, y) + F.cross_entropy(sim.t(), y))

      # Optional inv-margin using cross-pocket negatives (cheap)
      inv_pos = stageA["score_inv_geom"]
      inv_neg = stageA_cross["score_inv_geom"]
      invneg_loss = F.softplus(-(inv_pos - inv_neg)).mean()

      loss = (
        float(getattr(args, "pose_w", 1.0)) * pose_loss
        + float(getattr(args, "xpose_w", 1.0)) * xpose_loss
        + float(getattr(args, "retr_w", 1.0)) * retr_loss
        + float(getattr(args, "invneg_w", 1.0)) * invneg_loss
      )

    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % int(getattr(args, "log_every", 50)) == 0:
      with torch.no_grad():
        beta = float(torch.exp(model.log_beta))
        wa = float(F.softplus(model.sig_wa).detach())
        wb = float(F.softplus(model.sig_wb).detach())
        tau = float(F.softplus(model.log_tau).detach())
        mu_a = float(model.mu_a.detach())
        mu_b = float(model.mu_b.detach())

        if enable_pose and (getattr(args, "pose_w", 0.0) > 0.0 or getattr(args, "xpose_w", 0.0) > 0.0):
          acc_pose = (s_pos > s_neg_pose).float().mean().item()
          acc_xpose = (s_pos > s_cross).float().mean().item()
        else:
          acc_pose = float("nan")
          acc_xpose = float("nan")

        print(
          f"step {step:06d} "
          f"loss {loss.item():.4f} "
          f"pose {pose_loss.item():.4f} xpose {xpose_loss.item():.4f} "
          f"retr {retr_loss.item():.4f} invneg {invneg_loss.item():.4f} | "
          f"acc_pose {acc_pose if acc_pose==acc_pose else -1:.3f} "
          f"acc_xpose {acc_xpose if acc_xpose==acc_xpose else -1:.3f} "
          f"beta {beta:.3f} "
          f"edge_raw mean {stageA['edge_raw'].mean().item():.3f} "
          f"edge(tanh) mean {stageA['edge'].mean().item():.3f} | "
          f"mu_a {mu_a:.3f} mu_b {mu_b:.3f} wa {wa:.3f} wb {wb:.3f} tau {tau:.3f} "
          f"(enable_pose={enable_pose} enable_edge={enable_edge})"
        )

    step += 1
    if getattr(args, "max_train_steps", 0) and step >= args.max_train_steps:
      break

  return step


@torch.no_grad()
def evaluate(model, dl, device, args, max_batches: int = 10) -> Dict[str, float]:
  model.eval()

  enable_pose = bool(getattr(args, "enable_pose", True))
  enable_edge = bool(getattr(args, "enable_edge", True))

  pose_accs, xpose_accs = [], []
  top1s, mrrs, meanranks, aucs = [], [], [], []

  for bi, batch in enumerate(dl):
    if bi >= max_batches:
      break

    batch = move_batch(batch, device)

    cent = pocket_centroid(batch["p_pos"], batch["p_mask"])
    batch["p_pos"] = batch["p_pos"] - cent[:, None, :]
    batch["l_pos"] = batch["l_pos"] - cent[:, None, :]

    if getattr(args, "eval_aug_se3", False):
      batch["p_pos"], batch["l_pos"] = augment_se3_pair(
        batch["p_pos"],
        batch["l_pos"],
        batch["p_mask"],
        batch["l_mask"],
        max_trans=getattr(args, "aug_trans", 0.0),
      )

    l_pos_bad = make_pose_negative(
      batch["l_pos"], batch["l_mask"], max_trans=getattr(args, "pose_neg_trans", 3.0)
    )

    with torch.amp.autocast(
      device_type=device.type,
      enabled=bool(getattr(args, "amp", False) and device.type == "cuda"),
    ):
      stageA = model.precompute_stageA_batch(
        batch["l_pos"],
        batch["l_x"],
        batch["l_typ"],
        batch["l_mask"],
        batch["p_pos"],
        batch["p_x"],
        batch["p_typ"],
        batch["p_score"],
        batch["p_rad"],
        batch["p_mask"],
        topk=getattr(args, "topk", None),
        edge_pairs=getattr(args, "edge_pairs", None),
        enable_edge=enable_edge,
      )

      # pose eval optional
      if enable_pose:
        s_pos, _ = model.score_pose_from_stageA_batch(
          stageA, batch["l_pos"], batch["l_mask"], batch["p_pos"], batch["p_mask"], enable_pose=True
        )
        s_neg, _ = model.score_pose_from_stageA_batch(
          stageA, l_pos_bad, batch["l_mask"], batch["p_pos"], batch["p_mask"], enable_pose=True
        )

        perm = derangement_shift(batch["p_pos"].shape[0], device=device)
        stageA_cross = model.precompute_stageA_batch(
          batch["l_pos"][perm],
          batch["l_x"][perm],
          batch["l_typ"][perm],
          batch["l_mask"][perm],
          batch["p_pos"],
          batch["p_x"],
          batch["p_typ"],
          batch["p_score"],
          batch["p_rad"],
          batch["p_mask"],
          topk=getattr(args, "topk", None),
          edge_pairs=getattr(args, "edge_pairs", None),
          enable_edge=enable_edge,
        )
        s_cross, _ = model.score_pose_from_stageA_batch(
          stageA_cross,
          batch["l_pos"][perm],
          batch["l_mask"][perm],
          batch["p_pos"],
          batch["p_mask"],
          enable_pose=True,
        )

        pose_accs.append((s_pos > s_neg).float().mean().item())
        xpose_accs.append((s_pos > s_cross).float().mean().item())

      # retrieval metrics
      sim = stageA["l_z"] @ stageA["p_z"].t()
      top1, mrr, mean_rank, auc = retrieval_metrics(sim)
      top1s.append(top1)
      mrrs.append(mrr)
      meanranks.append(mean_rank)
      aucs.append(auc)

  def _mean(xs):
    return float(sum(xs) / max(1, len(xs)))

  return {
    "pose_acc": _mean(pose_accs) if enable_pose else float("nan"),
    "xpose_acc": _mean(xpose_accs) if enable_pose else float("nan"),
    "retr_top1": _mean(top1s),
    "retr_mrr": _mean(mrrs),
    "retr_mean_rank": _mean(meanranks),
    "retr_auc": _mean(aucs),
  }
