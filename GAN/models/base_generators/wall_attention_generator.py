
import torch
import torch.nn as nn
import torch.nn.functional as F
from domain_wall_generation.gan.models.unet_blocks import UNetDown_latent, UNetUp_latent

# Generator to Use Latent vector and wall attention togetehr

class Pix2PixGenerator_wall_attention(nn.Module):
    """
    Pix2PixGenerator_wall_attention integrates learned spatial attention and stochastic latent modulation 
    to generate more realistic domain wall structures in synthetic STEM images, especially for complex 
    domain classes like typeC-DW.
    It returns two tensors when return_attn=True, fake_img and attn_mask otherwise returns only fake_img.
    This generator architecture extends the standard U-Net-based Pix2Pix generator with the following enhancements:

    1. **WallAttention Module**:
       - Learns a soft spatial attention map from the displacement map.
       - Highlights probable domain wall regions.
       - Injects spatially localized noise to these regions during training to increase structural diversity 
         without disrupting the base polarization pattern.

    2. **LatentFiLM Conditioning**:
       - Injects an 8-dimensional Gaussian latent vector `z` into both encoder and decoder layers via 
         FiLM (Feature-wise Linear Modulation).
       - Allows stochastic variation in output for classes with domain wall variability.
       - Latent modulation is class-aware: for `UP` and `DN` classes (no domain wall), `z` is zeroed out 
         to disable variability.

    3. **Flexible Output**:
       - Optionally returns the learned attention map alongside the generated image.

    Args:
        in_channels (int): Number of input channels (default: 1 for displacement maps).
        out_channels (int): Number of output channels (default: 1 for grayscale STEM-like images).
        n_classes (int): Number of class labels used for FiLM conditioning.

    Forward Args:
        x (torch.Tensor): Input displacement map of shape (B, 1, H, W).
        labels (torch.Tensor): Class labels of shape (B,) for FiLM conditioning.
        z (torch.Tensor, optional): Latent noise vector of shape (B, 8). If None, defaults to zeros.
        return_attn (bool): If True, returns the attention map along with the generated image.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
            - Generated image of shape (B, 1, H, W).
            - Optionally, the attention map of shape (B, 1, H, W) if return_attn=True.
    """
    def __init__(self, in_channels=1, out_channels=1, n_classes= 2):
        super().__init__()
        self.use_learned_attention = True  # Optional toggle
        self.noise_scale = 0.2             # Can be learned or tuned
        self.wall_attention = WallAttention(in_channels=1)
        
        self.down1 = UNetDown_latent(1, 64, normalize=False, z_dim=8)
        self.down2 = UNetDown_latent(64, 128, z_dim=8)
        self.down3 = UNetDown_latent(128, 256)
        self.down4 = UNetDown_latent(256, 512)
        self.down5 = UNetDown_latent(512, 512)
        self.down6 = UNetDown_latent(512, 512)

        self.up1 = UNetUp_latent(512, 512, n_classes, dropout=0.5)
        self.up2 = UNetUp_latent(1024, 512, n_classes, dropout=0.5)
        self.up3 = UNetUp_latent(1024, 256, n_classes)
        self.up4 = UNetUp_latent(512, 128, n_classes)
        self.up5 = UNetUp_latent(256, 64, n_classes)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, labels, z=None, return_attn=False):
        if self.use_learned_attention:
            attn_mask = self.wall_attention(x)
            #Attention Noise Application
            
            noise = torch.randn_like(x) * self.noise_scale
            #x = x * (1 + attn_mask * noise)    #might unintentionally amplify non-wall regions, especially early in training.
            x = x + (attn_mask * noise)   #Keeps the displacement signal closer to original form, encourages local, spatially-controlled perturbation
            
        else:
            attn_mask = torch.zeros_like(x[:, :1, :, :])
    
        # Default z = zeros for UP/DN
        if z is None:
            z = torch.zeros(x.size(0), 8, device=x.device)
    
        d1 = self.down1(x, z)
        d2 = self.down2(d1, z)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
    
        u1 = self.up1(d6, d5, labels, z)
        u2 = self.up2(u1, d4, labels, z)
        u3 = self.up3(u2, d3, labels, z)
        u4 = self.up4(u3, d2, labels, z)
        u5 = self.up5(u4, d1, labels, z)
    
        output = self.final(u5)
    
        if return_attn:
            return output, attn_mask
        else:
            return output
        

def estimate_wall_mask(disp_map, threshold=0.1, smooth=True):
    """
    Estimate a smooth wall mask from a displacement map using gradient magnitude
    and optional local averaging for smoothing.

    Args:
        disp_map (torch.Tensor): Displacement tensor of shape [B, 1, H, W]
        threshold (float): Threshold for normalized gradient magnitude to define wall presence
        smooth (bool): If True, apply local average pooling to smooth the mask

    Returns:
        torch.Tensor: Soft wall mask of shape [B, 1, H, W], with values in [0, 1]
    """
    # Sobel-like horizontal and vertical gradient kernels
    sobel_x = torch.tensor([[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)  # [1, 1, 3, 3]
    sobel_y = torch.tensor([[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0)

    sobel_x = sobel_x.to(disp_map.device)
    sobel_y = sobel_y.to(disp_map.device)

    dx = F.conv2d(disp_map, sobel_x, padding=1)
    dy = F.conv2d(disp_map, sobel_y, padding=1)

    grad_mag = torch.sqrt(dx**2 + dy**2)  # [B, 1, H, W]

    # Normalize per sample
    B, _, H, W = grad_mag.shape
    grad_mag_flat = grad_mag.view(B, -1)
    min_val = grad_mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_val = grad_mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    grad_mag = (grad_mag - min_val) / (max_val - min_val + 1e-8)

    if smooth:
        grad_mag = F.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)

    return grad_mag  # Soft attention target instead of binary mask Soft wall mask in [0, 1]


class WallAttention(nn.Module):
    """
    WallAttention computes a spatial attention map over the input displacement map
    to highlight likely domain wall regions. The attention map is used to modulate
    the input via noise injection, encouraging the generator to focus more on
    structurally significant areas such as domain walls.

    This module uses a small convolutional network with a sigmoid activation to
    output a soft attention mask in the range [0, 1]. The architecture can be
    extended for deeper or multi-scale attention if needed.

    Attributes:
        attn_net (nn.Sequential): A convolutional neural network that takes a
            single-channel input (displacement map) and outputs a single-channel
            attention map highlighting probable wall regions.

    Forward Args:
        disp_map (torch.Tensor): Tensor of shape (B, 1, H, W), representing
            vertical displacement maps used as a proxy for polarization.

    Returns:
        torch.Tensor: A soft attention map of shape (B, 1, H, W) with values
            in [0, 1], indicating spatial importance for domain wall structures.
    """
    
    def __init__(self, in_channels=1):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # output attention mask
            #nn.Softplus()  # Values > 0, unbounded. Softplus can output values >1, which makes MSE-based loss_attn ineffective unless you normalize.
            nn.Sigmoid()  # more aligned with loss supervision targets
        )

    def forward(self, disp_map):
        return self.attn_net(disp_map)