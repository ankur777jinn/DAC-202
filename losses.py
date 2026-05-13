"""
Loss functions.

  - DiceBCELoss              -> Stage 1 (liver) and ablations A, B, C
  - FocalTverskyLoss         -> asymmetric, focal version of Dice
  - BoundaryWeightedBCELoss  -> per-pixel BCE weighted by distance-to-boundary
  - BoundaryAwareFocalTverskyLoss -> our final loss (config D)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        p = p.view(p.size(0), -1)
        y = target.view(target.size(0), -1).float()
        num = 2.0 * (p * y).sum(dim=1) + self.eps
        den = p.sum(dim=1) + y.sum(dim=1) + self.eps
        return (1.0 - num / den).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self, logits, target):
        d = self.dice(logits, target)
        b = F.binary_cross_entropy_with_logits(logits, target.float())
        return self.dice_w * d + self.bce_w * b


class FocalTverskyLoss(nn.Module):
    """
    TI  = TP / (TP + alpha * FP + beta * FN)
    L   = (1 - TI) ** (1 / gamma)
    With beta > alpha, false negatives (missed lesions) cost more than false
    positives -- right tradeoff when lesions are small.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 4.0 / 3.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        p = p.view(p.size(0), -1)
        y = target.view(target.size(0), -1).float()

        tp = (p * y).sum(dim=1)
        fp = (p * (1 - y)).sum(dim=1)
        fn = ((1 - p) * y).sum(dim=1)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1.0 - tversky).clamp(min=self.eps).pow(1.0 / self.gamma)
        return loss.mean()


class BoundaryWeightedBCELoss(nn.Module):
    """Weighted BCE with per-pixel weights precomputed from distance transform."""
    def forward(self, logits, target, weight_map):
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(), reduction="none"
        )
        denom = weight_map.sum().clamp(min=1.0)
        return (bce * weight_map).sum() / denom


class BoundaryAwareFocalTverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.3, beta: float = 0.7, gamma: float = 4.0 / 3.0,
                 lambda_ft: float = 1.0, lambda_bd: float = 0.5):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha, beta, gamma)
        self.bd = BoundaryWeightedBCELoss()
        self.lambda_ft = lambda_ft
        self.lambda_bd = lambda_bd

    def forward(self, logits, target, weight_map):
        ft = self.ft(logits, target)
        bd = self.bd(logits, target, weight_map)
        loss = self.lambda_ft * ft + self.lambda_bd * bd
        return loss, ft.detach(), bd.detach()


def build_stage2_loss(use_focal_tversky: bool, use_boundary: bool, loss_cfg):
    """Pick the loss based on ablation flags."""
    if use_focal_tversky and use_boundary:
        full = BoundaryAwareFocalTverskyLoss(
            alpha=loss_cfg.alpha, beta=loss_cfg.beta, gamma=loss_cfg.gamma,
            lambda_ft=loss_cfg.lambda_focal_tversky,
            lambda_bd=loss_cfg.lambda_boundary,
        )
        def fwd(logits, target, weight_map):
            return full(logits, target, weight_map)
        return fwd

    if use_focal_tversky and not use_boundary:
        ft = FocalTverskyLoss(loss_cfg.alpha, loss_cfg.beta, loss_cfg.gamma)
        def fwd(logits, target, weight_map):
            l = ft(logits, target)
            return l, l.detach(), torch.tensor(0.0, device=logits.device)
        return fwd

    dice_bce = DiceBCELoss()
    def fwd(logits, target, weight_map):
        l = dice_bce(logits, target)
        return l, l.detach(), torch.tensor(0.0, device=logits.device)
    return fwd
