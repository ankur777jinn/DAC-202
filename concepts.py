"""
Concept extraction for stratified failure-mode analysis.

Given a CT image, ground-truth tumor mask, and ground-truth liver mask, compute
6 properties used for stratified evaluation (NOT fed into the model):
  1. size_pixels           -> raw pixel count of the tumor
  2. boundary_sharpness    -> mean image-gradient magnitude on the tumor boundary
  3. contrast              -> |mean(tumor intensity) - mean(surrounding ring)|
  4. compactness           -> 4*pi*area / perimeter^2  (1 = circle, < 1 elongated)
  5. dist_to_liver_boundary-> min distance from tumor pixel to liver edge
  6. texture_heterogeneity -> GLCM entropy inside the tumor
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.measure import label as cc_label, regionprops
from skimage.feature import graycomatrix


SIZE_BUCKETS = [(0, 100, "tiny"), (100, 900, "small"),
                (900, 3600, "medium"), (3600, 10**9, "large")]
COMPACT_BUCKETS = [(0.0, 0.4, "elongated"), (0.4, 0.7, "irregular"),
                   (0.7, 1.01, "round")]
CONTRAST_BUCKETS = [(0, 15, "very_low"), (15, 30, "low"),
                    (30, 60, "moderate"), (60, 256, "high")]
BOUNDARY_BUCKETS = [(0, 5, "diffuse"), (5, 15, "moderate"), (15, 1e9, "sharp")]


def bucketize(value: float, buckets) -> str:
    for lo, hi, name in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][2]


def size_pixels(tumor_mask: np.ndarray) -> float:
    return float(tumor_mask.sum())


def boundary_sharpness(image: np.ndarray, tumor_mask: np.ndarray) -> float:
    """Mean image-gradient magnitude on the tumor boundary."""
    if tumor_mask.sum() == 0:
        return float("nan")
    img = image.astype(np.float32)
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    dil = binary_dilation(tumor_mask, iterations=1)
    boundary = dil & (~tumor_mask.astype(bool))
    vals = gmag[boundary]
    return float(vals.mean()) if vals.size > 0 else float("nan")


def contrast(image: np.ndarray, tumor_mask: np.ndarray,
             liver_mask: np.ndarray, ring_width: int = 5) -> float:
    """|mean(tumor) - mean(surrounding liver ring)| in intensity units."""
    if tumor_mask.sum() == 0:
        return float("nan")
    img = image.astype(np.float32)
    inside_mean = img[tumor_mask.astype(bool)].mean()
    dilated = binary_dilation(tumor_mask, iterations=ring_width)
    ring = dilated & (~tumor_mask.astype(bool)) & liver_mask.astype(bool)
    if ring.sum() == 0:
        ring = liver_mask.astype(bool) & (~tumor_mask.astype(bool))
    if ring.sum() == 0:
        return float("nan")
    outside_mean = img[ring].mean()
    return float(abs(inside_mean - outside_mean))


def compactness(tumor_mask: np.ndarray) -> float:
    """Area-weighted compactness across connected components."""
    labeled = cc_label(tumor_mask > 0)
    props = regionprops(labeled)
    if not props:
        return float("nan")
    total_area = sum(p.area for p in props)
    if total_area == 0:
        return float("nan")
    score = 0.0
    for p in props:
        if p.perimeter == 0:
            continue
        c = 4.0 * np.pi * p.area / (p.perimeter ** 2)
        score += c * (p.area / total_area)
    return float(min(score, 1.0))


def dist_to_liver_boundary(tumor_mask: np.ndarray,
                            liver_mask: np.ndarray) -> float:
    """Min distance from any tumor pixel to the liver boundary, in pixels."""
    if tumor_mask.sum() == 0 or liver_mask.sum() == 0:
        return float("nan")
    dt = distance_transform_edt(liver_mask > 0)
    return float(dt[tumor_mask.astype(bool)].min())


def texture_heterogeneity(image: np.ndarray, tumor_mask: np.ndarray) -> float:
    """GLCM entropy inside the tumor bbox (non-tumor pixels zeroed out)."""
    if tumor_mask.sum() < 20:
        return float("nan")
    ys, xs = np.where(tumor_mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch = image[y0:y1, x0:x1].copy()
    sub_mask = tumor_mask[y0:y1, x0:x1]
    patch[sub_mask == 0] = 0
    patch_q = (patch.astype(np.float32) / 256.0 * 32).clip(0, 31).astype(np.uint8)
    glcm = graycomatrix(patch_q, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    p = glcm + 1e-12
    ent = -np.sum(p * np.log(p), axis=(0, 1))
    return float(ent.mean())


def extract_all_concepts(image: np.ndarray,
                          tumor_mask: np.ndarray,
                          liver_mask: np.ndarray) -> dict:
    """Compute all 6 concept values + bucketed versions."""
    sz = size_pixels(tumor_mask)
    bs = boundary_sharpness(image, tumor_mask)
    co = contrast(image, tumor_mask, liver_mask)
    cm = compactness(tumor_mask)
    db = dist_to_liver_boundary(tumor_mask, liver_mask)
    th = texture_heterogeneity(image, tumor_mask)

    return {
        "size_pixels": sz,
        "size_bucket": bucketize(sz, SIZE_BUCKETS),
        "boundary_sharpness": bs,
        "boundary_bucket": bucketize(bs if not np.isnan(bs) else 0.0, BOUNDARY_BUCKETS),
        "contrast": co,
        "contrast_bucket": bucketize(co if not np.isnan(co) else 0.0, CONTRAST_BUCKETS),
        "compactness": cm,
        "compactness_bucket": bucketize(cm if not np.isnan(cm) else 0.0, COMPACT_BUCKETS),
        "dist_to_liver_boundary": db,
        "texture_heterogeneity": th,
    }
