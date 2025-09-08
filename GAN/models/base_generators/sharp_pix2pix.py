import torch
import torch.nn as nn
import torch.nn.functional as F
from domain_wall_generation.gan.models.unet_blocks import UNetDown, UNetUp


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_classes=6):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512, n_classes, dropout=0.5)
        self.up2 = UNetUp(1024, 512, n_classes, dropout=0.5)
        self.up3 = UNetUp(1024, 256, n_classes)
        self.up4 = UNetUp(512, 128, n_classes)
        self.up5 = UNetUp(256, 64, n_classes)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5, labels)
        u2 = self.up2(u1, d4, labels)
        u3 = self.up3(u2, d3, labels)
        u4 = self.up4(u3, d2, labels)
        u5 = self.up5(u4, d1, labels)

        return self.final(u5)

class Pix2PixGenerator_gradient_based(nn.Module):
    """
    A gradient-aware variant of the Pix2Pix generator designed to focus on domain wall structures 
    in displacement maps. This model perturbs the input using gradient-based attention to 
    encourage learning of physically meaningful variations in wall geometry.

    This architecture:
    - Adds noise modulated by estimated gradient magnitude (wall mask) to the input displacement map.
    - Uses a standard U-Net-like encoder-decoder architecture with FiLM conditioning via UNetUp blocks.
    - Is particularly useful for training on synthetic or real data where non-uniform wall features 
      (e.g., curved or stepped walls) are important for realism.

    Args:
        in_channels (int): Number of input channels (default: 1, e.g., vertical displacement map).
        out_channels (int): Number of output image channels (default: 1, e.g., grayscale STEM image).
        n_classes (int): Number of domain classes for FiLM modulation in the decoder.
    
    Forward Args:
        x (Tensor): Input displacement map of shape [B, 1, H, W].
        labels (Tensor): Class labels of shape [B], used for FiLM conditioning in decoder.

    Returns:
        Tensor: Generated image of shape [B, out_channels, H, W] with pixel values in [-1, 1].
    """
    def __init__(self, in_channels=1, out_channels=1, n_classes=2):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512, n_classes, dropout=0.5)
        self.up2 = UNetUp(1024, 512, n_classes, dropout=0.5)
        self.up3 = UNetUp(1024, 256, n_classes)
        self.up4 = UNetUp(512, 128, n_classes)
        self.up5 = UNetUp(256, 64, n_classes)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, labels):
        wall_mask = estimate_wall_mask(x)
        noise = torch.randn_like(x) * 0.2
        x_noisy = x + noise * wall_mask
        d1 = self.down1(x_noisy)

        #d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5, labels)
        u2 = self.up2(u1, d4, labels)
        u3 = self.up3(u2, d3, labels)
        u4 = self.up4(u3, d2, labels)
        u5 = self.up5(u4, d1, labels)

        return self.final(u5)



