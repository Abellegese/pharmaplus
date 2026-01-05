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

  for batch in dl:
    batch = move_batch(batch, device)

    # center by pocket centroid (like inference)
    cent = pocket_centroid(batch["p_pos"], batch["p_mask"])
    batch["p_pos"] = batch["p_pos"] - cent[:, None, :]
    batch["l_pos"] = batch["l_pos"] - cent[:, None, :]

    # shared SE3 augmentation (keeps relative pose)
    if args.aug_se3:
      batch["p_pos"], batch["l_pos"] = augment_se3_pair(
        batch["p_pos"],
        batch["l_pos"],
        batch["p_mask"],
        batch["l_mask"],
        max_trans=args.aug_trans,
      )

    # pose negatives (ligand only)
    l_pos_bad = make_pose_negative(
      batch["l_pos"], batch["l_mask"], max_trans=args.pose_neg_trans
    )

    opt.zero_grad(set_to_none=True)
    use_amp = bool(args.amp and device.type == "cuda")

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
      # Stage A for correct pairs (batched)
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
        topk=args.topk,
        edge_pairs=args.edge_pairs,
      )

      # Pose scores for pos and pose-negative (reuse W)
      s_pos, _dbg = model.score_pose_from_stageA_batch(
        stageA, batch["l_pos"], batch["l_mask"], batch["p_pos"], batch["p_mask"]
      )
      s_neg_pose, _ = model.score_pose_from_stageA_batch(
        stageA, l_pos_bad, batch["l_mask"], batch["p_pos"], batch["p_mask"]
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
        topk=args.topk,
        edge_pairs=args.edge_pairs,
      )
      s_cross, _ = model.score_pose_from_stageA_batch(
        stageA_cross,
        batch["l_pos"][perm],
        batch["l_mask"][perm],
        batch["p_pos"],
        batch["p_mask"],
      )
      xpose_loss = F.softplus(-(s_pos - s_cross)).mean()

      # Retrieval loss (InfoNCE) on pooled embeddings (FAST)
      l_z = stageA["l_z"]
      p_z = stageA["p_z"]
      sim = (l_z @ p_z.t()) / float(args.temp)  # [B,B]
      y = torch.arange(sim.shape[0], device=device)
      retr_loss = 0.5 * (F.cross_entropy(sim, y) + F.cross_entropy(sim.t(), y))

      # Optional inv-margin using cross-pocket negatives (cheap)
      inv_pos = stageA["score_inv_geom"]
      inv_neg = stageA_cross["score_inv_geom"]
      invneg_loss = F.softplus(-(inv_pos - inv_neg)).mean()

      loss = (
        args.pose_w * pose_loss
        + args.xpose_w * xpose_loss
        + args.retr_w * retr_loss
        + args.invneg_w * invneg_loss
      )

    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if step % args.log_every == 0:
      with torch.no_grad():
        acc_pose = (s_pos > s_neg_pose).float().mean().item()
        acc_xpose = (s_pos > s_cross).float().mean().item()
        beta = float(torch.exp(model.log_beta))
        print(
          f"step {step:06d} "
          f"loss {loss.item():.4f} "
          f"pose {pose_loss.item():.4f} xpose {xpose_loss.item():.4f} "
          f"retr {retr_loss.item():.4f} invneg {invneg_loss.item():.4f} | "
          f"acc_pose {acc_pose * 100:.1f}% acc_xpose {acc_xpose * 100:.1f}% "
          f"beta {beta:.3f} "
          f"edge_raw mean {stageA['edge_raw'].mean().item():.3f} "
          f"edge(tanh) mean {stageA['edge'].mean().item():.3f}"
        )

    step += 1
    if getattr(args, "max_train_steps", 0) and step >= args.max_train_steps:
      break

  return step


@torch.no_grad()
def evaluate(model, dl, device, args, max_batches: int = 10) -> Dict[str, float]:
  model.eval()
  pose_accs, xpose_accs = [], []
  top1s, mrrs, meanranks, aucs = [], [], [], []

  for bi, batch in enumerate(dl):
    if bi >= max_batches:
      break
    batch = move_batch(batch, device)

    cent = pocket_centroid(batch["p_pos"], batch["p_mask"])
    batch["p_pos"] = batch["p_pos"] - cent[:, None, :]
    batch["l_pos"] = batch["l_pos"] - cent[:, None, :]

    if args.eval_aug_se3:
      batch["p_pos"], batch["l_pos"] = augment_se3_pair(
        batch["p_pos"],
        batch["l_pos"],
        batch["p_mask"],
        batch["l_mask"],
        max_trans=args.aug_trans,
      )

    l_pos_bad = make_pose_negative(
      batch["l_pos"], batch["l_mask"], max_trans=args.pose_neg_trans
    )

    with torch.amp.autocast(
      device_type=device.type,
      enabled=bool(args.amp and device.type == "cuda"),
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
        topk=args.topk,
        edge_pairs=args.edge_pairs,
      )
      s_pos, _ = model.score_pose_from_stageA_batch(
        stageA, batch["l_pos"], batch["l_mask"], batch["p_pos"], batch["p_mask"]
      )
      s_neg, _ = model.score_pose_from_stageA_batch(
        stageA, l_pos_bad, batch["l_mask"], batch["p_pos"], batch["p_mask"]
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
        topk=args.topk,
        edge_pairs=args.edge_pairs,
      )
      s_cross, _ = model.score_pose_from_stageA_batch(
        stageA_cross,
        batch["l_pos"][perm],
        batch["l_mask"][perm],
        batch["p_pos"],
        batch["p_mask"],
      )

      pose_accs.append((s_pos > s_neg).float().mean().item())
      xpose_accs.append((s_pos > s_cross).float().mean().item())

      # retrieval metrics (retr_sim only; fast)
      sim = stageA["l_z"] @ stageA["p_z"].t()
      top1, mrr, mean_rank, auc = retrieval_metrics(sim)
      top1s.append(top1)
      mrrs.append(mrr)
      meanranks.append(mean_rank)
      aucs.append(auc)

  return {
    "pose_acc": float(sum(pose_accs) / max(1, len(pose_accs))),
    "xpose_acc": float(sum(xpose_accs) / max(1, len(xpose_accs))),
    "retr_top1": float(sum(top1s) / max(1, len(top1s))),
    "retr_mrr": float(sum(mrrs) / max(1, len(mrrs))),
    "retr_mean_rank": float(sum(meanranks) / max(1, len(meanranks))),
    "retr_auc": float(sum(aucs) / max(1, len(aucs))),
  }
