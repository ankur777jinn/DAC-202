"""Metrics: Dice, IoU, HD95."""
import numpy as np
import torch

try:
    from medpy.metric.binary import hd95 as _medpy_hd95
    _HAVE_MEDPY = True
except ImportError:
    _HAVE_MEDPY = False


def _to_numpy_bin(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (x > 0).astype(np.uint8)


def dice_score(pred, target, eps: float = 1e-6) -> float:
    p = _to_numpy_bin(pred).reshape(-1)
    t = _to_numpy_bin(target).reshape(-1)
    inter = (p * t).sum()
    return float((2.0 * inter + eps) / (p.sum() + t.sum() + eps))


def iou_score(pred, target, eps: float = 1e-6) -> float:
    p = _to_numpy_bin(pred).reshape(-1)
    t = _to_numpy_bin(target).reshape(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + eps) / (union + eps))


def hd95_score(pred, target, spacing=None) -> float:
    p = _to_numpy_bin(pred)
    t = _to_numpy_bin(target)
    if p.sum() == 0 or t.sum() == 0:
        return float("nan")
    if _HAVE_MEDPY:
        try:
            return float(_medpy_hd95(p, t, voxelspacing=spacing))
        except Exception:
            return float("nan")
    from scipy.ndimage import distance_transform_edt
    dt_t = distance_transform_edt(1 - t)
    dt_p = distance_transform_edt(1 - p)
    d_pt = dt_t[p.astype(bool)]
    d_tp = dt_p[t.astype(bool)]
    if d_pt.size == 0 or d_tp.size == 0:
        return float("nan")
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


def batch_metrics(logits: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.5) -> dict:
    p = (torch.sigmoid(logits) > threshold).long()
    t = (target > 0.5).long()
    dices, ious, hd95s = [], [], []
    for i in range(p.size(0)):
        pi = p[i, 0]; ti = t[i, 0]
        dices.append(dice_score(pi, ti))
        ious.append(iou_score(pi, ti))
        h = hd95_score(pi, ti)
        if not np.isnan(h):
            hd95s.append(h)
    return {
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "hd95": float(np.mean(hd95s)) if hd95s else float("nan"),
        "n_with_hd95": len(hd95s),
    }
