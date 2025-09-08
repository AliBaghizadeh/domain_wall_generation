# Import Libraries
import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient

# Import Finished

spatial_grad = SpatialGradient(mode='sobel', order=1, normalized=True)

def curvature_loss(img):
    """
    Curvature Loss for Structural Realism of Domain Walls.

    This loss encourages the presence of curved, non-linear domain wall features
    by maximizing the variance of gradient magnitudes across the image.
    A higher gradient variance implies more local structural diversity, 
    which is characteristic of realistic stepped or jagged domain walls.

    Args:
        img (torch.Tensor): Generated image or predicted displacement map. 
            Expected shape is one of:
                - [B, 1, H, W] for batch of grayscale images
                - [B, H, W] for unbatched input
                - [H, W] for a single image (will be reshaped)

    Returns:
        torch.Tensor: A scalar tensor representing the negative gradient magnitude variance.
            This negative value is added to the total generator loss to promote curvature.

    Raises:
        ValueError: If the input tensor is not single-channel.

    Notes:
        - Uses Kornia's Sobel-based SpatialGradient to compute edge magnitude.
        - Output is negative since higher curvature (variance) should reduce total loss.
    """
    if img.dim() == 3:  # [B, H, W]
        img = img.unsqueeze(1)
    elif img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)

    if img.size(1) != 1:
        raise ValueError(f"Expected 1-channel image. Got shape: {img.shape}")

    grad = spatial_grad(img)  # ✅ don't call spatial_grad(), just spatial_grad(img)
    # grad shape: [B, 1, 2, H, W]
    grad_x = grad[:, 0, 0]  # [B, H, W]
    grad_y = grad[:, 0, 1]  # [B, H, W]

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    grad_var = torch.var(grad_mag)

    return -grad_var  # Encourage wall curvature


def latent_diversity_loss(generator, disp_map, labels, z_dim):
    """
    Latent Diversity Loss for Encouraging Output Variability.

    Measures how sensitive the generator is to different latent codes (z) by 
    generating two images from the same input displacement map and label,
    but with different z vectors. A higher difference between these outputs 
    indicates better utilization of the latent space.

    This loss helps prevent the generator from ignoring the stochastic latent 
    code and encourages it to model intra-class variability (e.g., wall roughness, 
    local distortions).

    Args:
        generator (nn.Module): The Pix2Pix-based generator model that accepts latent input z.
        disp_map (torch.Tensor): Input displacement maps, shape [B, 1, H, W].
        labels (torch.Tensor): Class labels for each input sample, shape [B].
        z_dim (int): Dimensionality of the latent vector z.

    Returns:
        torch.Tensor: A scalar tensor representing the negative L1 distance between
            images generated using two different latent codes. The more different the
            outputs, the better the diversity, so the L1 distance is negated to
            reward diversity.
    Notes:
        - Assumes the generator accepts arguments: (disp_map, labels, z, return_attn=True)
    """
    
    z1 = torch.randn(disp_map.size(0), z_dim, device=disp_map.device)
    z2 = torch.randn_like(z1)
    img1, _ = generator(disp_map, labels, z=z1, return_attn=True)
    img2, _ = generator(disp_map, labels, z=z2, return_attn=True)
    return -F.l1_loss(img1, img2)

