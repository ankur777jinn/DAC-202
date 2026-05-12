"""
Stage 1 training: liver segmentation with U-Net + ResNet-34.

Launch (auto-detects GPU count):
    torchrun --nproc_per_node=$(python -c "import torch; print(max(1,torch.cuda.device_count()))") train_stage1.py

  Single GPU / CPU:
    python train_stage1.py
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import DATA, TRAIN, MODEL, LOSS
from dataset import (LiTSStage1Dataset, load_split_dataframes, apply_filters)
from models.unet_stage1 import ResNetUNet
from losses import DiceBCELoss
from metrics import batch_metrics
from ddp_utils import (setup_ddp, cleanup_ddp, is_main_process, all_reduce_mean,
                       set_seed, cosine_warmup_lr, set_lr)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def build_loaders(world_size, rank):
    splits = load_split_dataframes(
        use_provided=DATA.use_provided_splits,
        csv_path=DATA.csv_path, data_root=DATA.data_root,
        train_csv=DATA.train_csv, val_csv=DATA.probe_csv, test_csv=DATA.test_csv,
        train_ratio=DATA.train_ratio, val_ratio=DATA.val_ratio, seed=DATA.split_seed,
    )

    # Stage 1: drop slices that have NO liver -- they're pure background
    train_df = apply_filters(splits["train"],
                             drop_empty_liver=DATA.drop_empty_liver_for_stage1)
    val_df = apply_filters(splits["val"],
                           drop_empty_liver=DATA.drop_empty_liver_for_stage1)
    if is_main_process():
        print(f"[stage1] train slices: {len(train_df)}, val slices: {len(val_df)}")

    train_ds = LiTSStage1Dataset(train_df, image_size=DATA.image_size,
                                  augment=True, seed=TRAIN.seed + rank)
    val_ds = LiTSStage1Dataset(val_df, image_size=DATA.image_size, augment=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN.stage1_batch_size,
                              sampler=train_sampler, num_workers=TRAIN.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=TRAIN.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=TRAIN.stage1_batch_size,
                            sampler=val_sampler, num_workers=TRAIN.num_workers,
                            pin_memory=True, drop_last=False,
                            persistent_workers=TRAIN.num_workers > 0)
    return train_loader, val_loader, train_sampler


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    losses, dices = [], []
    pbar = tqdm(loader, disable=not is_main_process(),
                desc=f"[stage1][train] ep{epoch}")
    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=TRAIN.use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        with torch.no_grad():
            m = batch_metrics(logits.detach(), masks)
        dices.append(m["dice"])
        if is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.3f}")
    return sum(losses) / len(losses), sum(dices) / len(dices)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    losses, dices, ious = [], [], []
    pbar = tqdm(loader, disable=not is_main_process(),
                desc=f"[stage1][val] ep{epoch}")
    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=TRAIN.use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        losses.append(loss.item())
        m = batch_metrics(logits, masks)
        dices.append(m["dice"])
        ious.append(m["iou"])
    return (sum(losses) / len(losses),
            sum(dices) / len(dices),
            sum(ious) / len(ious))


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    local_rank, rank, world_size = setup_ddp()
    set_seed(TRAIN.seed, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_sampler = build_loaders(world_size, rank)

    model = ResNetUNet(encoder_name=MODEL.stage1_encoder,
                       pretrained=MODEL.stage1_pretrained).to(device)
    model = torch.compile(model)        
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False)

    criterion = DiceBCELoss(LOSS.stage1_dice_weight, LOSS.stage1_bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.stage1_lr,
                                  weight_decay=TRAIN.weight_decay)
    scaler = GradScaler(enabled=TRAIN.use_amp)

    ckpt_dir = Path(TRAIN.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(Path(TRAIN.log_dir) / "stage1") if is_main_process() else None
    best_dice = 0.0

    for epoch in range(TRAIN.stage1_epochs):
        train_sampler.set_epoch(epoch)
        lr = cosine_warmup_lr(epoch, TRAIN.stage1_epochs, TRAIN.warmup_epochs,
                              TRAIN.stage1_lr, TRAIN.min_lr)
        set_lr(optimizer, lr)

        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer,
                                            criterion, scaler, device, epoch)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion,
                                                device, epoch)
        dt = time.time() - t0

        tr_loss = all_reduce_mean(tr_loss, world_size)
        tr_dice = all_reduce_mean(tr_dice, world_size)
        val_loss = all_reduce_mean(val_loss, world_size)
        val_dice = all_reduce_mean(val_dice, world_size)
        val_iou = all_reduce_mean(val_iou, world_size)

        if is_main_process():
            print(f"[stage1] ep{epoch:03d} lr={lr:.2e} "
                  f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} "
                  f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} "
                  f"time={dt:.0f}s")
            writer.add_scalar("stage1/train_loss", tr_loss, epoch)
            writer.add_scalar("stage1/val_loss", val_loss, epoch)
            writer.add_scalar("stage1/val_dice", val_dice, epoch)
            writer.add_scalar("stage1/val_iou", val_iou, epoch)
            writer.add_scalar("stage1/lr", lr, epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.module.state_dict(),
                            "epoch": epoch, "val_dice": val_dice},
                           ckpt_dir / "stage1_best.pt")
                print(f"[stage1] new best val_dice={val_dice:.4f}, saved.")
            if (epoch + 1) % TRAIN.save_every == 0:
                torch.save({"model": model.module.state_dict(),
                            "epoch": epoch, "val_dice": val_dice},
                           ckpt_dir / f"stage1_ep{epoch:03d}.pt")

    if is_main_process():
        writer.close()
        print(f"[stage1] training done. best val_dice={best_dice:.4f}")
    cleanup_ddp()


if __name__ == "__main__":
    main()