"""
Dataset for the Kaggle LiTS-PNG dump organized as:

  dataset_6/
    volume-N_K.png                          <- CT slice
    segmentation-N_livermask_K.png          <- liver binary mask
    segmentation-N_lesionmask_K.png         <- tumor binary mask

Indexed by lits_df.csv with columns:
  filepath, liver_maskpath, tumor_maskpath,
  study_number, instance_number, liver_mask_empty, tumor_mask_empty

Masks are typically saved as {0, 255} 8-bit PNG; we binarize on load.

Two Dataset classes:
  - LiTSStage1Dataset: CT slice  -> liver mask (binary)
  - LiTSStage2Dataset: liver-cropped CT slice -> tumor mask (binary, with boundary weights)
"""
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt

from config import DATA


# ===================== CSV index =====================

class LiTSCsvIndex:
    """
    Loads lits_df.csv, rewrites the relative paths in the CSV (which look like
    "../input/lits-png/dataset_6/...") so they point at your local dataset_6
    directory, and exposes helpers for filtering + patient-level splitting.
    """
    def __init__(self, csv_path: str, data_root: str):
        self.df = pd.read_csv(csv_path)
        self.df = self._fix_paths(self.df, data_root)

    @staticmethod
    def _fix_paths(df: pd.DataFrame, data_root: str) -> pd.DataFrame:
        data_root = str(data_root)
        for col in ["filepath", "liver_maskpath", "tumor_maskpath"]:
            # Replace everything up to and including ".../dataset_6/" with data_root + "/"
            df[col] = df[col].apply(lambda p: os.path.join(data_root, os.path.basename(p)))
        return df

    def filter(self, drop_empty_liver: bool = False,
               drop_empty_tumor: bool = False) -> pd.DataFrame:
        d = self.df
        if drop_empty_liver:
            d = d[~d["liver_mask_empty"].astype(bool)]
        if drop_empty_tumor:
            d = d[~d["tumor_mask_empty"].astype(bool)]
        return d.reset_index(drop=True)

    def patient_split(self, train_ratio: float, val_ratio: float,
                      seed: int = 42) -> Dict[str, pd.DataFrame]:
        studies = sorted(self.df["study_number"].unique().tolist())
        rng = np.random.default_rng(seed)
        rng.shuffle(studies)
        n = len(studies)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_s = set(studies[:n_train])
        val_s = set(studies[n_train:n_train + n_val])
        test_s = set(studies[n_train + n_val:])

        def take(s):
            return self.df[self.df["study_number"].isin(s)].reset_index(drop=True)

        return {"train": take(train_s), "val": take(val_s), "test": take(test_s)}


def load_split_dataframes(use_provided: bool,
                          csv_path: str, data_root: str,
                          train_csv: Optional[str] = None,
                          val_csv: Optional[str] = None,
                          test_csv: Optional[str] = None,
                          train_ratio: float = 0.70,
                          val_ratio: float = 0.15,
                          seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Returns dict {train, val, test} of DataFrames with fixed local paths.

    If use_provided is True, loads the provided CSVs (lits_train.csv, lits_test.csv,
    lits_probe.csv as validation) and fixes their paths the same way.

    Otherwise builds a fresh patient-level split from lits_df.csv.
    """
    if use_provided:
        out = {}
        for name, path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(f"Provided split CSV missing for {name}: {path}")
            df = pd.read_csv(path)
            df = LiTSCsvIndex._fix_paths(df, data_root)
            out[name] = df
        return out
    idx = LiTSCsvIndex(csv_path, data_root)
    return idx.patient_split(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


# ===================== mask loading =====================

def load_binary_mask(path: str) -> np.ndarray:
    """
    Load a PNG mask and binarize. Handles 0/255 storage and 0/1 storage.
    Returns uint8 array with values in {0, 1}.
    """
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


# ===================== boundary weights =====================

def compute_boundary_weights(binary_mask: np.ndarray, k: float = 5.0) -> np.ndarray:
    mask = binary_mask.astype(np.uint8)
    if mask.sum() == 0 or mask.sum() == mask.size:
        return np.full_like(mask, 1e-3, dtype=np.float32)
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    dist_to_boundary = dist_outside + dist_inside
    weights = np.exp(-dist_to_boundary / k).astype(np.float32)
    return weights


# ===================== augmentations =====================

def _random_flip_rot(image: np.ndarray, mask: np.ndarray,
                      rng: np.random.Generator,
                      extra_mask: Optional[np.ndarray] = None):
    if rng.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        if extra_mask is not None:
            extra_mask = np.fliplr(extra_mask).copy()
    if rng.random() < 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        if extra_mask is not None:
            extra_mask = np.flipud(extra_mask).copy()
    k = int(rng.integers(0, 4))
    if k > 0:
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
        if extra_mask is not None:
            extra_mask = np.rot90(extra_mask, k).copy()
    return image, mask, extra_mask


def _normalize_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """uint8 HxW -> normalized 3xHxW tensor."""
    t = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


# ===================== Stage 1 dataset =====================

class LiTSStage1Dataset(Dataset):
    """CT slice -> liver mask. Operates on a DataFrame (filtered & path-fixed)."""

    def __init__(self, df: pd.DataFrame, image_size: int = 384,
                 augment: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["filepath"]).convert("L"), dtype=np.uint8)
        liver_mask = load_binary_mask(row["liver_maskpath"])
        # Tumor pixels ARE liver pixels biologically. ~12k rows have
        # liver_mask=empty but tumor_mask=non-empty; the liver IS present there.
        # Train Stage 1 on (liver | tumor).
        tumor_mask = load_binary_mask(row["tumor_maskpath"])
        liver_mask = ((liver_mask > 0) | (tumor_mask > 0)).astype(np.uint8)

        if img.shape != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
            liver_mask = cv2.resize(liver_mask, (self.image_size, self.image_size),
                                    interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, liver_mask, _ = _random_flip_rot(img, liver_mask, self.rng)

        img_t = _normalize_to_tensor(img)
        mask_t = torch.from_numpy(liver_mask).unsqueeze(0).float()
        # Identifier for logging / later joining
        ident = f"vol{int(row['study_number'])}_s{int(row['instance_number'])}"
        return img_t, mask_t, ident


# ===================== Stage 2 dataset =====================

class LiTSStage2Dataset(Dataset):
    """
    Liver-cropped CT slice -> tumor mask + boundary weight map.

    Crop bbox sources:
      - use_gt_liver=True  -> derive bbox from the GT liver mask (upper-bound experiments)
      - use_gt_liver=False -> use a bbox from crops_manifest (keyed by 'vol{S}_s{I}')
                              produced by preprocess.py from Stage 1 predictions.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 crop_size: int = 256,
                 augment: bool = False,
                 compute_boundary: bool = True,
                 boundary_k: float = 5.0,
                 use_gt_liver: bool = True,
                 crops_manifest: Optional[Dict[str, List[int]]] = None,
                 use_full_image: bool = False,
                 seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.crop_size = crop_size
        self.augment = augment
        self.compute_boundary = compute_boundary
        self.boundary_k = boundary_k
        self.use_gt_liver = use_gt_liver
        self.crops_manifest = crops_manifest or {}
        self.use_full_image = use_full_image   # ablation A: skip the cascade entirely
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            h, w = mask.shape
            return 0, 0, h, w
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        h, w = mask.shape
        return (max(0, int(y0) - pad), max(0, int(x0) - pad),
                min(h, int(y1) + pad), min(w, int(x1) + pad))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["filepath"]).convert("L"), dtype=np.uint8)
        liver_mask = load_binary_mask(row["liver_maskpath"])
        tumor_mask = load_binary_mask(row["tumor_maskpath"])
        # Combined "liver-present" region for bbox derivation (see Stage 1 note)
        liver_or_tumor = ((liver_mask > 0) | (tumor_mask > 0)).astype(np.uint8)

        h, w = img.shape
        ident = f"vol{int(row['study_number'])}_s{int(row['instance_number'])}"

        # ---- decide bbox ----
        if self.use_full_image:
            bbox = (0, 0, h, w)
        elif self.use_gt_liver:
            bbox = self._bbox_from_mask(liver_or_tumor, pad=8)
        else:
            bbox = tuple(self.crops_manifest.get(ident, (0, 0, h, w)))
        y0, x0, y1, x1 = bbox

        # ---- crop and resize ----
        img_c = img[y0:y1, x0:x1]
        tum_c = tumor_mask[y0:y1, x0:x1]
        # Some borderline cases (empty bbox) -> fall back to full image
        if img_c.size == 0:
            img_c = img
            tum_c = tumor_mask
            y0, x0, y1, x1 = 0, 0, h, w
        img_c = cv2.resize(img_c, (self.crop_size, self.crop_size),
                           interpolation=cv2.INTER_LINEAR)
        tum_c = cv2.resize(tum_c, (self.crop_size, self.crop_size),
                           interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img_c, tum_c, _ = _random_flip_rot(img_c, tum_c, self.rng)

        if self.compute_boundary:
            weight = compute_boundary_weights(tum_c, k=self.boundary_k)
        else:
            weight = np.ones_like(tum_c, dtype=np.float32)

        img_t = _normalize_to_tensor(img_c)
        mask_t = torch.from_numpy(tum_c).unsqueeze(0).float()
        weight_t = torch.from_numpy(weight).unsqueeze(0)
        bbox_t = torch.tensor([y0, x0, y1, x1], dtype=torch.long)
        return img_t, mask_t, weight_t, ident, bbox_t


# ===================== filtering helper =====================

def apply_filters(df: pd.DataFrame,
                  drop_empty_liver: bool = False,
                  drop_empty_tumor: bool = False) -> pd.DataFrame:
    """
    IMPORTANT: in lits_df.csv, `liver_mask_empty=True` does NOT always mean
    "no liver in this slice". There are ~12k rows where liver_mask_empty=True
    but tumor_mask_empty=False -- biologically the liver IS present (tumors
    live inside livers); the liver-mask PNG is just empty because the lesion
    mask captures that region. So we treat a slice as 'has liver' if EITHER
    the liver mask OR the tumor mask is non-empty.
    """
    out = df
    if drop_empty_liver:
        # "no liver" == "both liver mask AND tumor mask are empty"
        has_anything = (~out["liver_mask_empty"].astype(bool)) | \
                       (~out["tumor_mask_empty"].astype(bool))
        out = out[has_anything]
    if drop_empty_tumor and "tumor_mask_empty" in out.columns:
        out = out[~out["tumor_mask_empty"].astype(bool)]
    return out.reset_index(drop=True)
