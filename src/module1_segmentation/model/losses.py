"""Adapter from the team's integer masks to MONAI DiceCELoss."""
import torch
from torch import nn
from monai.losses import DiceCELoss


class SegmentationLoss(nn.Module):
    """Equal-weight foreground soft Dice + all-class cross entropy.

    Targets: int64 [B,D,H,W] (also accepts [B,1,D,H,W]).
    Class 0 is excluded from Dice only; CE still learns background.
    Missing classes use MONAI's smoothed Dice convention, not an evaluation
    metric. Negative/unknown labels are rejected, not silently remapped.
    """
    def __init__(self):
        super().__init__()
        self.criterion = DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True,
            lambda_dice=1.0, lambda_ce=1.0,
        )

    def forward(self, logits, masks):
        if logits.ndim != 5 or logits.shape[1] != 9:
            raise ValueError("logits must have shape [B,9,D,H,W]")
        if masks.ndim == 4:
            masks = masks.unsqueeze(1)  # MONAI needs a singleton channel.
        expected = (logits.shape[0], 1, *logits.shape[2:])
        if tuple(masks.shape) != expected:
            raise ValueError(f"mask shape must be {expected}, got {tuple(masks.shape)}")
        if masks.dtype != torch.int64:
            raise TypeError("masks must be torch.int64 integer class indices")
        if masks.device != logits.device:
            raise ValueError("masks and logits must be on the same device")
        if masks.numel() == 0 or masks.min().item() < 0 or masks.max().item() > 8:
            raise ValueError("mask values must be 0..8; resolve unknown labels upstream")
        # softmax=True affects Dice; the CE component receives raw logits.
        return self.criterion(logits, masks)
