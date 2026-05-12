"""
Qualitative figures for the report.

For a few cherry-picked test slices, render:
    image | GT mask | each ablation's prediction
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from config import DATA, MODEL, ABLATIONS
from dataset import load_binary_mask, load_split_dataframes, apply_filters
from models.hybrid_stage2 import HybridSegmenter


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


def prepare_input(img_crop):
    img = torch.from_numpy(img_crop.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((img - mean) / std).unsqueeze(0)


@torch.no_grad()
def render_panel(row, crops_manifest, models_by_ablation, device, out_path):
    img_full = np.array(Image.open(row["filepath"]).convert("L"))
    tumor_full = load_binary_mask(row["tumor_maskpath"])
    ident = f"vol{int(row['study_number'])}_s{int(row['instance_number'])}"

    h, w = img_full.shape
    bbox = crops_manifest.get(ident, [0, 0, h, w])
    y0, x0, y1, x1 = bbox
    img_crop = cv2.resize(img_full[y0:y1, x0:x1],
                          (DATA.lesion_crop_size, DATA.lesion_crop_size),
                          interpolation=cv2.INTER_LINEAR)
    tum_crop = cv2.resize(tumor_full[y0:y1, x0:x1],
                          (DATA.lesion_crop_size, DATA.lesion_crop_size),
                          interpolation=cv2.INTER_NEAREST)

    preds = {}
    inp = prepare_input(img_crop).to(device)
    for name, m in models_by_ablation.items():
        logits = m(inp)
        preds[name] = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    n_cols = 2 + len(preds)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.0))
    axes[0].imshow(img_crop, cmap="gray"); axes[0].set_title("CT (liver crop)"); axes[0].axis("off")
    axes[1].imshow(img_crop, cmap="gray")
    axes[1].imshow(np.ma.masked_where(tum_crop == 0, tum_crop), cmap="autumn", alpha=0.5)
    axes[1].set_title("Ground Truth"); axes[1].axis("off")
    for i, (name, p) in enumerate(preds.items()):
        axes[2 + i].imshow(img_crop, cmap="gray")
        axes[2 + i].imshow(np.ma.masked_where(p == 0, p), cmap="autumn", alpha=0.5)
        axes[2 + i].set_title(name); axes[2 + i].axis("off")

    fig.suptitle(ident, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-manifest", default="./crops_manifest.json")
    parser.add_argument("--ckpt-root", default="./checkpoints")
    parser.add_argument("--out-dir", default="./results/figures")
    parser.add_argument("--per-slice-csv", default="./results/per_slice.csv")
    parser.add_argument("--n-best", type=int, default=4)
    parser.add_argument("--n-worst", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_split_dataframes(
        use_provided=DATA.use_provided_splits,
        csv_path=DATA.csv_path, data_root=DATA.data_root,
        train_csv=DATA.train_csv, val_csv=DATA.probe_csv, test_csv=DATA.test_csv,
        train_ratio=DATA.train_ratio, val_ratio=DATA.val_ratio, seed=DATA.split_seed,
    )
    test_df = apply_filters(splits["test"],
                            drop_empty_liver=True, drop_empty_tumor=True)
    test_df["ident"] = ("vol" + test_df["study_number"].astype(str)
                        + "_s" + test_df["instance_number"].astype(str))

    with open(args.crops_manifest) as fp:
        crops_manifest = json.load(fp)

    models = {}
    for name, ab in ABLATIONS.items():
        ckpt = Path(args.ckpt_root) / name / "best.pt"
        if ckpt.exists():
            models[name] = load_model(str(ckpt), ab, device)

    # Choose slices to render
    if Path(args.per_slice_csv).exists():
        import pandas as pd
        df = pd.read_csv(args.per_slice_csv)
        df = df[(df["ablation"] == "D_full") & (df["has_tumor"] == 1)]
        df_sorted = df.sort_values("dice")
        worst_idents = df_sorted.head(args.n_worst)["ident"].tolist()
        best_idents = df_sorted.tail(args.n_best)["ident"].tolist()
        chosen_idents = list(set(best_idents + worst_idents))
    else:
        chosen_idents = test_df["ident"].iloc[:8].tolist()

    chosen_rows = test_df[test_df["ident"].isin(chosen_idents)]
    for _, row in chosen_rows.iterrows():
        safe = row["ident"]
        render_panel(row, crops_manifest, models, device, out_dir / f"{safe}.png")
    print(f"[viz] wrote {len(chosen_rows)} panels to {out_dir}")


if __name__ == "__main__":
    main()
