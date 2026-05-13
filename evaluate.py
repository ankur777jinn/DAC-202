"""
Evaluation: headline ablation table + concept-stratified analysis.

Produces:
  results/per_slice_<ablation>.csv  -> raw per-slice metrics + concepts
  results/per_slice.csv              -> concatenated
  results/headline_table.csv         -> mean Dice/IoU/HD95 per ablation
  results/stratified.csv             -> mean Dice per (ablation x concept bucket)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from tqdm import tqdm

from config import DATA, MODEL, ABLATIONS
from dataset import (LiTSStage2Dataset, load_split_dataframes, load_binary_mask,
                     apply_filters)
from models.hybrid_stage2 import HybridSegmenter
from metrics import dice_score, iou_score, hd95_score
from concepts import extract_all_concepts


def load_model(ckpt_path, ablation, device):
    model = HybridSegmenter(
        encoder_name=MODEL.stage2_encoder, pretrained=False,
        use_transformer=ablation.use_transformer,
        transformer_layers=MODEL.transformer_layers,
        transformer_heads=MODEL.transformer_heads,
        transformer_dim_feedforward=MODEL.transformer_dim_feedforward,
        transformer_dropout=MODEL.transformer_dropout,
        input_size=DATA.lesion_crop_size,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_one(ablation_name, ablation, test_df, crops_manifest,
                 ckpt_root, device, out_csv):
    ckpt_path = Path(ckpt_root) / ablation_name / "best.pt"
    if not ckpt_path.exists():
        print(f"[eval] missing checkpoint for {ablation_name}: {ckpt_path}")
        return None
    model = load_model(str(ckpt_path), ablation, device)

    use_full = not ablation.use_cascade
    # Eval keeps ALL slices that have liver (so distribution is realistic).
    # Tumor-empty slices are kept too -- model has to produce empty masks.
    eval_df = apply_filters(test_df,
                            drop_empty_liver=DATA.drop_empty_liver_for_stage2,
                            drop_empty_tumor=False)

    ds = LiTSStage2Dataset(
        eval_df, crop_size=DATA.lesion_crop_size, augment=False,
        compute_boundary=False,
        use_gt_liver=False, crops_manifest=crops_manifest,
        use_full_image=use_full,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    # Build lookup from identifier -> row (for path resolution during concept extraction)
    eval_df_indexed = eval_df.copy()
    eval_df_indexed["ident"] = ("vol" + eval_df_indexed["study_number"].astype(str)
                                + "_s" + eval_df_indexed["instance_number"].astype(str))
    ident_to_row = {r["ident"]: r for _, r in eval_df_indexed.iterrows()}

    rows = []
    for imgs, masks, _w, idents, bboxes in tqdm(loader, desc=f"eval {ablation_name}"):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
        gts = (masks > 0.5).long().numpy()

        for i in range(preds.shape[0]):
            p = preds[i, 0]
            t = gts[i, 0]
            ident = idents[i]
            row = ident_to_row[ident]

            d = dice_score(p, t)
            iou = iou_score(p, t)
            hd = hd95_score(p, t)

            # Reload originals at the SAME crop+resize so concept stats align spatially
            img_full = np.array(Image.open(row["filepath"]).convert("L"))
            liver_full_raw = load_binary_mask(row["liver_maskpath"])
            tumor_full = load_binary_mask(row["tumor_maskpath"])
            # See dataset.apply_filters note: 'liver region' = liver | tumor
            liver_full = ((liver_full_raw > 0) | (tumor_full > 0)).astype(np.uint8)
            y0, x0, y1, x1 = bboxes[i].tolist()
            img_crop = cv2.resize(img_full[y0:y1, x0:x1],
                                  (DATA.lesion_crop_size, DATA.lesion_crop_size),
                                  interpolation=cv2.INTER_LINEAR)
            liver_crop = cv2.resize(liver_full[y0:y1, x0:x1],
                                    (DATA.lesion_crop_size, DATA.lesion_crop_size),
                                    interpolation=cv2.INTER_NEAREST)
            tumor_crop = cv2.resize(tumor_full[y0:y1, x0:x1],
                                    (DATA.lesion_crop_size, DATA.lesion_crop_size),
                                    interpolation=cv2.INTER_NEAREST)
            concepts = extract_all_concepts(img_crop, tumor_crop, liver_crop)

            rows.append({
                "ablation": ablation_name,
                "ident": ident,
                "study_number": int(row["study_number"]),
                "instance_number": int(row["instance_number"]),
                "dice": d, "iou": iou, "hd95": hd,
                "has_tumor": int(tumor_crop.sum() > 0),
                "has_liver": int(liver_crop.sum() > 0),
                **concepts,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[eval] wrote {out_csv} ({len(df)} rows)")
    return df


def build_summary_tables(all_dfs, out_dir: Path):
    big = pd.concat(all_dfs, ignore_index=True)
    big.to_csv(out_dir / "per_slice.csv", index=False)

    # Headline: only slices that ACTUALLY contain tumor in GT
    # (Dice on empty GT is degenerate -- a model predicting empty gets dice=1,
    #  any prediction gets dice=0; including these would dominate the average.)
    with_tumor = big[big["has_tumor"] == 1]

    headline = with_tumor.groupby("ablation").agg(
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        iou_mean=("iou", "mean"),
        hd95_mean=("hd95", lambda s: s.dropna().mean()),
        n=("dice", "count"),
    ).reset_index()
    headline.to_csv(out_dir / "headline_table.csv", index=False)
    print("\n=== Headline table (tumor-present slices) ===")
    print(headline.to_string(index=False))

    # Stratified analysis
    concepts_buckets = ["size_bucket", "boundary_bucket",
                        "contrast_bucket", "compactness_bucket"]
    strat_rows = []
    for c in concepts_buckets:
        g = with_tumor.groupby(["ablation", c]).agg(
            dice_mean=("dice", "mean"),
            hd95_mean=("hd95", lambda s: s.dropna().mean()),
            n=("dice", "count"),
        ).reset_index().rename(columns={c: "bucket"})
        g["concept"] = c.replace("_bucket", "")
        strat_rows.append(g[["concept", "ablation", "bucket",
                              "dice_mean", "hd95_mean", "n"]])
    strat = pd.concat(strat_rows, ignore_index=True)
    strat.to_csv(out_dir / "stratified.csv", index=False)
    print("\n=== Stratified analysis (Dice per concept bucket) ===")
    print(strat.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-manifest", default="./crops_manifest.json")
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_split_dataframes(
        use_provided=DATA.use_provided_splits,
        csv_path=DATA.csv_path, data_root=DATA.data_root,
        train_csv=DATA.train_csv, val_csv=DATA.probe_csv, test_csv=DATA.test_csv,
        train_ratio=DATA.train_ratio, val_ratio=DATA.val_ratio, seed=DATA.split_seed,
    )
    test_df = splits["test"]

    with open(args.crops_manifest) as fp:
        crops_manifest = json.load(fp)

    dfs = []
    for ab_name in args.ablations:
        ab = ABLATIONS[ab_name]
        df = evaluate_one(ab_name, ab, test_df, crops_manifest,
                          args.ckpt_root, device,
                          out_dir / f"per_slice_{ab_name}.csv")
        if df is not None:
            dfs.append(df)

    if dfs:
        build_summary_tables(dfs, out_dir)


if __name__ == "__main__":
    main()
