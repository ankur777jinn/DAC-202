"""
Stage 2 training: tumor segmentation with cascaded hybrid model.

Launch (auto-detects GPU count):
    torchrun --nproc_per_node=$(python -c "import torch; print(max(1,torch.cuda.device_count()))") train_stage2.py \
        --ablation D_full \
        --crops-manifest ./crops_manifest.json

  Single GPU / CPU:
    python train_stage2.py --ablation D_full --crops-manifest ./crops_manifest.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import DATA, TRAIN, MODEL, LOSS, ABLATIONS
from dataset import (LiTSStage2Dataset, load_split_dataframes, apply_filters)
from models.hybrid_stage2 import HybridSegmenter
from losses import build_stage2_loss
from metrics import batch_metrics
from ddp_utils import (setup_ddp, cleanup_ddp, is_main_process, all_reduce_mean,
                       set_seed, cosine_warmup_lr, set_lr)


def build_loaders(ablation, crops_manifest, world_size, rank):
    splits = load_split_dataframes(
        use_provided=DATA.use_provided_splits,
        csv_path=DATA.csv_path, data_root=DATA.data_root,
        train_csv=DATA.train_csv, val_csv=DATA.probe_csv, test_csv=DATA.test_csv,
        train_ratio=DATA.train_ratio, val_ratio=DATA.val_ratio, seed=DATA.split_seed,
    )

    # Stage 2: drop slices without liver (no ROI to crop). Optionally drop
    # tumor-empty slices during training only (NEVER for eval).
    train_df = apply_filters(splits["train"],
                             drop_empty_liver=DATA.drop_empty_liver_for_stage2,
                             drop_empty_tumor=DATA.drop_empty_tumor_for_stage2_train)
    val_df = apply_filters(splits["val"],
                           drop_empty_liver=DATA.drop_empty_liver_for_stage2,
                           drop_empty_tumor=False)

    if is_main_process():
        print(f"[stage2:{ablation.name}] train slices: {len(train_df)}, val slices: {len(val_df)}")
        print(f"[stage2:{ablation.name}] use_cascade={ablation.use_cascade} "
              f"use_transformer={ablation.use_transformer} "
              f"use_focal_tversky={ablation.use_focal_tversky} "
              f"use_boundary={ablation.use_boundary_loss}")

    use_full_image = not ablation.use_cascade
    # When cascade is on, use_gt_liver=False -> Stage 1 predictions are used.
    use_gt_liver = False

    train_ds = LiTSStage2Dataset(
        train_df, crop_size=DATA.lesion_crop_size, augment=True,
        compute_boundary=True, boundary_k=LOSS.boundary_k,
        use_gt_liver=use_gt_liver, crops_manifest=crops_manifest,
        use_full_image=use_full_image, seed=TRAIN.seed + rank,
    )
    val_ds = LiTSStage2Dataset(
        val_df, crop_size=DATA.lesion_crop_size, augment=False,
        compute_boundary=True, boundary_k=LOSS.boundary_k,
        use_gt_liver=use_gt_liver, crops_manifest=crops_manifest,
        use_full_image=use_full_image,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN.stage2_batch_size,
                              sampler=train_sampler, num_workers=TRAIN.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=TRAIN.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=TRAIN.stage2_batch_size,
                            sampler=val_sampler, num_workers=TRAIN.num_workers,
                            pin_memory=True, drop_last=False,
                            persistent_workers=TRAIN.num_workers > 0)
    return train_loader, val_loader, train_sampler


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch):
    model.train()
    losses, dices, ft_terms, bd_terms = [], [], [], []
    pbar = tqdm(loader, disable=not is_main_process(),
                desc=f"[stage2][train] ep{epoch}")
    for imgs, masks, weights, _, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=TRAIN.use_amp):
            logits = model(imgs)
            loss, ft, bd = loss_fn(logits, masks, weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        ft_terms.append(ft.item() if torch.is_tensor(ft) else float(ft))
        bd_terms.append(bd.item() if torch.is_tensor(bd) else float(bd))
        with torch.no_grad():
            m = batch_metrics(logits.detach(), masks)
        dices.append(m["dice"])
        if is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.3f}")
    return (sum(losses) / len(losses), sum(dices) / len(dices),
            sum(ft_terms) / len(ft_terms), sum(bd_terms) / len(bd_terms))


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch):
    model.eval()
    losses, dices, ious, hd95s = [], [], [], []
    pbar = tqdm(loader, disable=not is_main_process(),
                desc=f"[stage2][val] ep{epoch}")
    for imgs, masks, weights, _, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        with autocast(enabled=TRAIN.use_amp):
            logits = model(imgs)
            loss, _, _ = loss_fn(logits, masks, weights)
        losses.append(loss.item())
        m = batch_metrics(logits, masks)
        dices.append(m["dice"])
        ious.append(m["iou"])
        if not torch.isnan(torch.tensor(m["hd95"])):
            hd95s.append(m["hd95"])
    val_hd95 = sum(hd95s) / len(hd95s) if hd95s else float("nan")
    return (sum(losses) / len(losses),
            sum(dices) / len(dices),
            sum(ious) / len(ious),
            val_hd95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=list(ABLATIONS.keys()), required=True)
    parser.add_argument("--crops-manifest", default="./crops_manifest.json")
    args = parser.parse_args()

    ablation = ABLATIONS[args.ablation]
    local_rank, rank, world_size = setup_ddp()
    set_seed(TRAIN.seed, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if ablation.use_cascade:
        with open(args.crops_manifest) as fp:
            crops_manifest = json.load(fp)
    else:
        crops_manifest = {}

    train_loader, val_loader, train_sampler = build_loaders(
        ablation, crops_manifest, world_size, rank)

    model = HybridSegmenter(
        encoder_name=MODEL.stage2_encoder,
        pretrained=MODEL.stage2_pretrained,
        use_transformer=ablation.use_transformer,
        transformer_layers=MODEL.transformer_layers,
        transformer_heads=MODEL.transformer_heads,
        transformer_dim_feedforward=MODEL.transformer_dim_feedforward,
        transformer_dropout=MODEL.transformer_dropout,
        input_size=DATA.lesion_crop_size,
    ).to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False)

    loss_fn = build_stage2_loss(
        use_focal_tversky=ablation.use_focal_tversky,
        use_boundary=ablation.use_boundary_loss,
        loss_cfg=LOSS,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.stage2_lr,
                                  weight_decay=TRAIN.weight_decay)
    scaler = GradScaler(enabled=TRAIN.use_amp)

    ckpt_dir = Path(TRAIN.ckpt_dir) / args.ablation
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(TRAIN.log_dir) / f"stage2_{args.ablation}") \
             if is_main_process() else None
    best_dice = 0.0

    for epoch in range(TRAIN.stage2_epochs):
        train_sampler.set_epoch(epoch)
        lr = cosine_warmup_lr(epoch, TRAIN.stage2_epochs, TRAIN.warmup_epochs,
                              TRAIN.stage2_lr, TRAIN.min_lr)
        set_lr(optimizer, lr)

        t0 = time.time()
        tr_loss, tr_dice, tr_ft, tr_bd = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch)
        val_loss, val_dice, val_iou, val_hd95 = validate(
            model, val_loader, loss_fn, device, epoch)
        dt = time.time() - t0

        tr_loss = all_reduce_mean(tr_loss, world_size)
        tr_dice = all_reduce_mean(tr_dice, world_size)
        val_loss = all_reduce_mean(val_loss, world_size)
        val_dice = all_reduce_mean(val_dice, world_size)
        val_iou = all_reduce_mean(val_iou, world_size)

        if is_main_process():
            print(f"[stage2:{args.ablation}] ep{epoch:03d} lr={lr:.2e} "
                  f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} "
                  f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
                  f"val_hd95={val_hd95:.2f} time={dt:.0f}s")
            writer.add_scalar("train/loss", tr_loss, epoch)
            writer.add_scalar("train/dice", tr_dice, epoch)
            writer.add_scalar("train/focal_tversky_term", tr_ft, epoch)
            writer.add_scalar("train/boundary_term", tr_bd, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            writer.add_scalar("val/iou", val_iou, epoch)
            writer.add_scalar("val/hd95", val_hd95, epoch)
            writer.add_scalar("lr", lr, epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.module.state_dict(),
                            "epoch": epoch, "val_dice": val_dice,
                            "ablation": args.ablation},
                           ckpt_dir / "best.pt")
                print(f"[stage2:{args.ablation}] new best val_dice={val_dice:.4f}, saved.")

            if (epoch + 1) % TRAIN.save_every == 0:
                torch.save({"model": model.module.state_dict(),
                            "epoch": epoch, "val_dice": val_dice,
                            "ablation": args.ablation},
                           ckpt_dir / f"ep{epoch:03d}.pt")

    if is_main_process():
        writer.close()
        print(f"[stage2:{args.ablation}] done. best val_dice={best_dice:.4f}")
    cleanup_ddp()


if __name__ == "__main__":
    main()