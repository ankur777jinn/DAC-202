"""
Preprocessing utilities for the CSV-based LiTS-PNG layout.

Two modes:

  --mode split
      Build (or copy) train/val/test DataFrames and write them as CSVs.
      If config.DATA.use_provided_splits is True, just verifies the provided
      CSVs exist and prints a summary. Otherwise generates a fresh patient-
      level split from lits_df.csv.

  --mode crops
      Run trained Stage 1 inference on every slice in the dataset and dump a
      JSON manifest mapping identifier 'vol{S}_s{I}' -> [y0, x0, y1, x1] of the
      predicted-liver bounding box. Stage 2 uses these for cascaded cropping.

Identifier convention: 'vol{study_number}_s{instance_number}'.
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from config import DATA, MODEL
from dataset import (LiTSCsvIndex, load_split_dataframes, LiTSStage1Dataset,
                     apply_filters)


def make_split(args):
    if DATA.use_provided_splits:
        # Just verify and summarize
        splits = load_split_dataframes(
            use_provided=True,
            csv_path=DATA.csv_path, data_root=DATA.data_root,
            train_csv=DATA.train_csv, val_csv=DATA.probe_csv, test_csv=DATA.test_csv,
        )
        print("[split] using provided CSVs:")
        for k, v in splits.items():
            print(f"  {k}: rows={len(v)}, studies={v['study_number'].nunique()}")
    else:
        idx = LiTSCsvIndex(DATA.csv_path, DATA.data_root)
        splits = idx.patient_split(train_ratio=DATA.train_ratio,
                                    val_ratio=DATA.val_ratio,
                                    seed=DATA.split_seed)
        out_dir = Path(args.split_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in splits.items():
            path = out_dir / f"{k}.csv"
            v.to_csv(path, index=False)
            print(f"  saved {path}  rows={len(v)}  studies={v['study_number'].nunique()}")


def inspect_data(args):
    """
    Open a handful of slices and print mask statistics so you can verify the
    'liver_maskpath' and 'tumor_maskpath' columns mean what we think.

    Things to look for in the printed output:
      - 'liver mask' should have THOUSANDS of pixels in slices through the
        middle of the liver (the liver is a big organ).
      - 'tumor mask' should be SMALLER and usually inside the liver mask.
      - If 'tumor mask' is consistently as big as 'liver mask', the columns
        may be swapped in this dataset variant.
    """
    idx = LiTSCsvIndex(DATA.csv_path, DATA.data_root)
    df = idx.df

    # Pick mid-volume slices from a few studies (more likely to contain liver)
    mid = df.groupby("study_number").apply(
        lambda g: g.iloc[len(g) // 2]).reset_index(drop=True)

    print(f"Inspecting {min(args.n_inspect, len(mid))} mid-volume slices:")
    print(f"{'study':>6} {'inst':>5} {'liver_px':>10} {'tumor_px':>10} "
          f"{'tumor_in_liver%':>16} {'tumor_in_tumor%':>16}")
    print("-" * 75)

    n_tumor_inside = 0
    n_tumor_outside = 0
    for _, row in mid.head(args.n_inspect).iterrows():
        from dataset import load_binary_mask
        try:
            liver = load_binary_mask(row["liver_maskpath"])
            tumor = load_binary_mask(row["tumor_maskpath"])
        except FileNotFoundError as e:
            print(f"  MISSING FILE: {e}")
            continue
        liver_n = int(liver.sum())
        tumor_n = int(tumor.sum())
        if tumor_n == 0:
            inside_pct = float("nan")
        else:
            inside_pct = float((tumor & liver).sum()) / tumor_n * 100
        if liver_n == 0:
            tumor_pct = float("nan")
        else:
            tumor_pct = tumor_n / liver_n * 100
        print(f"{int(row['study_number']):>6} "
              f"{int(row['instance_number']):>5} "
              f"{liver_n:>10} {tumor_n:>10} "
              f"{inside_pct:>15.1f}% {tumor_pct:>15.1f}%")
        if tumor_n > 0 and inside_pct > 80:
            n_tumor_inside += 1
        elif tumor_n > 0:
            n_tumor_outside += 1

    print()
    print(f"Tumor mostly inside liver mask:   {n_tumor_inside} slices")
    print(f"Tumor mostly outside liver mask:  {n_tumor_outside} slices")
    print()
    print("Expected for correct column labels:")
    print("  - liver_px >> tumor_px in slices that contain both (liver is bigger)")
    print("  - tumor pixels should be >80% inside the liver mask")
    print("If liver_px and tumor_px look swapped, set the column mapping in "
          "config.DataConfig (manual fix).")


@torch.no_grad()
def make_crops(args):
    from models.unet_stage1 import ResNetUNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(encoder_name=MODEL.stage1_encoder, pretrained=False).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Run inference on ALL slices (train + val + test) so Stage 2 can use cascade
    # crops uniformly. We DO NOT drop liver-empty slices here -- predictions on
    # them just give an empty bbox -> fall back to full image.
    idx = LiTSCsvIndex(DATA.csv_path, DATA.data_root)
    full_df = idx.df

    ds = LiTSStage1Dataset(full_df, image_size=DATA.image_size, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=True)

    manifest = {}
    pad = 8

    for imgs, _, idents in tqdm(loader, desc="stage1 inference"):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Map idents back to rows -> get original image size
        for prob, ident in zip(probs, idents):
            # find the original image size from the CSV row
            row = full_df[
                (full_df["study_number"].astype(str) + "_" +
                 full_df["instance_number"].astype(str))
                == ident.replace("vol", "").replace("_s", "_")
            ]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            orig = np.array(Image.open(row["filepath"]).convert("L"))
            oh, ow = orig.shape

            mask = (prob[0] > 0.5).astype(np.uint8)
            if mask.sum() == 0:
                manifest[ident] = [0, 0, int(oh), int(ow)]
                continue
            ys, xs = np.where(mask > 0)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            # Rescale bbox from DATA.image_size to original image dims
            sy = oh / DATA.image_size
            sx = ow / DATA.image_size
            y0 = max(0, int(y0 * sy) - pad)
            x0 = max(0, int(x0 * sx) - pad)
            y1 = min(int(oh), int(y1 * sy) + pad)
            x1 = min(int(ow), int(x1 * sx) + pad)
            manifest[ident] = [y0, x0, y1, x1]

    Path(args.manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_path, "w") as fp:
        json.dump(manifest, fp)
    print(f"[crops] manifest saved to {args.manifest_path}  ({len(manifest)} entries)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["split", "crops", "inspect"], required=True)
    p.add_argument("--split-dir", default="./splits",
                   help="(split mode, no provided splits) output dir for csvs")
    p.add_argument("--stage1-ckpt", default="./checkpoints/stage1_best.pt")
    p.add_argument("--manifest-path", default="./crops_manifest.json")
    p.add_argument("--n-inspect", type=int, default=5,
                   help="(inspect mode) number of slices to dump info for")
    args = p.parse_args()

    if args.mode == "split":
        make_split(args)
    elif args.mode == "inspect":
        inspect_data(args)
    else:
        make_crops(args)
