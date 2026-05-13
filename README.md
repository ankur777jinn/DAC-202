# LiTS Cascaded Hybrid Segmentation — Course Project

End-to-end pipeline for liver-tumor segmentation on the
[LiTS-PNG Kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/lits-png).

## Pipeline

- **Stage 1**: U-Net + ResNet-34 → liver mask
- **Stage 2**: ResNet-34 encoder + transformer bottleneck + U-Net decoder → tumor mask
- **Loss (config D)**: Boundary-Aware Focal Tversky
- **Evaluation**: 4-way ablation + concept-stratified failure analysis

Trained with DDP across 2 GPUs.

---

## 0. Dataset layout (what we expect)

The Kaggle dump organizes everything in a single `dataset_6/` folder, and
ships with these CSVs:

```
dataset_6/                          (all PNGs flat in this directory)
  volume-0_0.png                    <- CT slice for study 0, instance 0
  segmentation-0_livermask_0.png    <- liver mask for the same
  segmentation-0_lesionmask_0.png   <- tumor mask for the same
  ... (~58k slices across 131 studies)
lits_df.csv                          <- master CSV (every slice)
lits_train.csv  lits_test.csv  lits_probe.csv  <- provided splits
```

`lits_df.csv` columns:
`filepath, liver_maskpath, tumor_maskpath, study_number, instance_number, liver_mask_empty, tumor_mask_empty`

Paths inside the CSVs look like `../input/lits-png/dataset_6/volume-0_0.png`.
Our code rewrites these to point at your local `DATA.data_root` automatically.

## 1. Setup

```bash
pip install -r requirements.txt
```

Edit `config.py` → `DataConfig`:

```python
csv_path: str = "./lits_df.csv"                       # path to the master CSV
data_root: str = "./data/lits-png/dataset_6"          # local PNG folder

use_provided_splits: bool = True                       # use the shipped CSVs
train_csv: str = "./lits_train.csv"
test_csv: str = "./lits_test.csv"
probe_csv: str = "./lits_probe.csv"                    # we use probe as validation
```

If you don't want the provided splits, set `use_provided_splits = False` and we
build a patient-level split (no leakage across studies).

## 2. Verify split

```bash
python preprocess.py --mode split
```

This just prints a summary so you know the CSVs are read correctly.

## 2.5. Verify mask semantics (IMPORTANT — do this once)

The provided CSV columns `liver_maskpath` and `tumor_maskpath` should mean what
they say, but the dataset has some quirks worth verifying before training:

```bash
python preprocess.py --mode inspect --n-inspect 10
```

This prints mid-volume slice statistics: liver pixel count, tumor pixel count,
and what fraction of tumor pixels fall inside the liver mask. **Expected**:
liver is much bigger than tumor, and tumor is mostly (>80%) inside the liver
mask. If the numbers look swapped (tumor consistently larger than liver), the
CSV columns may be flipped — open `config.DataConfig` and add a manual fix.

**Filtering note**: the CSV has ~12k rows where `liver_mask_empty=True` but
`tumor_mask_empty=False`. Biologically a slice can't have tumor without liver,
so our filter treats "liver present" as `liver_mask OR tumor_mask is non-empty`
to be safe. Stage 1's training target is also `liver | tumor` for the same
reason.

## 3. Stage 1 — liver U-Net

```bash
torchrun --nproc_per_node=2 train_stage1.py
```

Best checkpoint at `./checkpoints/stage1_best.pt`. Expect liver Dice > 0.93.

Stage 1 automatically drops slices where `liver_mask_empty == True` (~33% of
the data) — those slices are pure background and offer nothing to learn.

## 4. Generate liver-crop manifest

```bash
python preprocess.py --mode crops \
    --stage1-ckpt ./checkpoints/stage1_best.pt \
    --manifest-path ./crops_manifest.json
```

For every slice (train + val + test), runs Stage 1 inference and dumps the
predicted-liver bbox as `{"vol{S}_s{I}": [y0, x0, y1, x1]}`. Stage 2 reads
this for cascaded cropping — using predicted (not GT) liver bboxes makes
the cascade realistic.

## 5. Stage 2 — train all 4 ablations

```bash
for A in A_baseline B_cascade C_cascade_transformer D_full; do
  torchrun --nproc_per_node=2 train_stage2.py --ablation $A
done
```

- **A_baseline** — no cascade (full slice → tumor), no transformer, Dice+BCE
- **B_cascade** — predicted-liver crop, no transformer, Dice+BCE
- **C_cascade_transformer** — + transformer bottleneck, Dice+BCE
- **D_full** — + Boundary-Aware Focal Tversky loss

Checkpoints land at `./checkpoints/<ablation>/best.pt`.

## 6. Evaluate

```bash
python evaluate.py
```

Outputs in `./results/`:
- `per_slice_<ablation>.csv` — per slice metrics + 6 concept properties
- `per_slice.csv` — concatenated
- `headline_table.csv` — your headline 4-row ablation table
- `stratified.csv` — Dice per (ablation × concept bucket)

**Note**: the headline metrics are computed on **tumor-present slices only**
(`has_tumor == 1`). On tumor-empty slices, Dice is undefined (0/0) or
degenerate (1 if model also predicts empty, 0 if anything else). Reporting on
the realistic subset is the honest convention.

## 7. Qualitative figures

```bash
python visualize.py
```

Saves to `./results/figures/`. Picks best and worst test slices according to
the D-config model's Dice — gives you both success cases and failure modes for
the critical-analysis section.

---

## Loss intuition

```
L = lambda_ft * FocalTversky(logits, target)
  + lambda_bd * BoundaryWeightedBCE(logits, target, weight_map)
```

- **Focal Tversky**: Tversky generalizes Dice with asymmetric FP/FN weights.
  β=0.7 > α=0.3 → missing a tumor costs more than a false alarm. Focal
  exponent γ=4/3 down-weights easy examples.
- **Boundary-weighted BCE**: per-pixel weight = `exp(-dist_to_boundary / k)`
  precomputed from the GT tumor mask. Pixels near the boundary contribute
  exponentially more — directly attacks low-contrast edges.

Defaults in `config.LossConfig`:
- α=0.3, β=0.7, γ=4/3 (Abraham & Khan 2019)
- boundary_k=5.0
- λ_ft=1.0, λ_bd=0.5

## Compute notes

- ResNet-34 + 3-layer transformer fits comfortably on A10/L4/A100.
- AMP enabled by default.
- Per-GPU batch sizes: Stage 1 = 16, Stage 2 = 24 (effective 32 / 48 on 2 GPUs).
- Stage 1 ≈ 40 epochs; Stage 2 ≈ 80 epochs.

## Filtering policy summary

| Stage | Drop liver-empty? | Drop tumor-empty? |
|---|---|---|
| Stage 1 train + val | Yes | n/a |
| Stage 2 train | Yes | configurable (default: keep, as hard negatives) |
| Stage 2 val | Yes | No |
| Stage 2 test | Yes | No |

Reporting on `has_tumor == 1` slices in the headline table reflects realistic
clinical use — performance on slices that actually contain something to find.
