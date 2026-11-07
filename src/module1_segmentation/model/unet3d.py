"""Member 3: randomly initialized MONAI 3D U-Net. No preprocessing here."""
import torch
from torch import nn
from monai.networks.nets import UNet

LABEL_MAP = {"background": 0, "T10": 1, "T11": 2, "T12": 3,
             "L1": 4, "L2": 5, "L3": 6, "L4": 7, "L5": 8}


class VertebraUNet(nn.Module):
    """Input [B,1,D,H,W] -> raw logits [B,9,D,H,W].

    Four stride-2 levels require spatial sizes divisible by 16. Minimum 32
    per axis leaves more than one spatial element for InstanceNorm.
    MONAI uses strided convolutions, skip concatenations and transposed
    convolutions. This is an existing U-Net implementation, not a novel model.
    """
    def __init__(self, channels=(16, 32, 64, 128, 256)):
        super().__init__()
        if len(channels) != 5 or any(c <= 0 for c in channels):
            raise ValueError("channels must contain five positive integers")
        self.channels = tuple(channels)
        self.net = UNet(
            spatial_dims=3, in_channels=1, out_channels=9,
            channels=self.channels, strides=(2, 2, 2, 2),
            num_res_units=0, norm=("INSTANCE", {"affine": True}),
            act="PRELU", dropout=0.0,
        )

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 1 or images.shape[0] < 1:
            raise ValueError("images must have shape [B,1,D,H,W], B >= 1")
        if not images.is_floating_point():
            raise TypeError("images must be floating point; DataLoader uses float32")
        if any(s < 32 or s % 16 for s in images.shape[2:]):
            raise ValueError("D,H,W must each be >= 32 and divisible by 16")
        return self.net(images)  # No softmax: loss expects raw logits.
