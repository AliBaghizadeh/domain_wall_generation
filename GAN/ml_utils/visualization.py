# Import Libraries
import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from domain_wall_generation.config_labels import POLAR_LABELS
from torchvision.utils import make_grid

# Import Finished

def show_sample_displacements(dataset, n_per_class=3, label_names=None):
    """
    Display STEM image and displacement map pairs for each class separately.

    Each row shows:
        - Left: STEM image
        - Right: Displacement map
        - Class label as text

    Args:
        dataset (Dataset): Dataset instance with (disp, img, label[, ...]).
        n_per_class (int): Number of samples to show per class.
        label_names (dict): Optional label-to-name mapping (e.g. {3: 'typeC-DW'}).
    """
    label_to_indices = defaultdict(list)

    # Dynamically build label index mapping
    for idx in range(len(dataset)):
        _, _, label, *_ = dataset[idx]  # Support for extra returned fields
        label = int(label)
        label_to_indices[label].append(idx)

    total_classes = len(label_to_indices)
    fig, axs = plt.subplots(n_per_class, total_classes * 2, figsize=(4 * total_classes, 3 * n_per_class))

    if n_per_class == 1:
        axs = np.expand_dims(axs, axis=0)

    for col, label in enumerate(sorted(label_to_indices.keys())):
        label_name = label_names[label] if label_names and label in label_names else f"Class {label}"
        indices = random.sample(label_to_indices[label], min(n_per_class, len(label_to_indices[label])))

        for row, idx in enumerate(indices):
            disp, img, *_ = dataset[idx]
            axs[row, col * 2].imshow((img.squeeze(0) * 0.5 + 0.5).numpy(), cmap='gray')
            axs[row, col * 2].set_title(f"{label_name} - STEM")
            axs[row, col * 2].axis('off')

            axs[row, col * 2 + 1].imshow((disp.squeeze(0) * 0.5 + 0.5).numpy(), cmap='viridis')
            axs[row, col * 2 + 1].set_title(f"{label_name} - Disp")
            axs[row, col * 2 + 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_attention(generator, disp_map, save_dir="./outputs"):
    """
    Visualizes the raw learned attention map from the generator's WallAttention module.

    This function extracts the attention mask generated from the input displacement map
    and displays it as a heatmap. It is useful for inspecting whether the model's attention
    mechanism is activating meaningfully in wall-relevant regions during or after training.

    Args:
        generator (nn.Module): The generator model containing a `wall_attention` module.
        disp_map (torch.Tensor): A single displacement map of shape [1, 1, H, W], on any device.
        save_dir (str): Directory where the generated attention heatmap will be saved.

    Returns:
        None. Saves the attention map as a heatmap and displays it.

    Note:
        This function does not overlay the attention on the input image or compare it with
        wall masks. For richer visualization, use `visualize_attention_overlay(...)`.
    """
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "learned_wall_attention.png")

    with torch.no_grad():
        device = next(generator.parameters()).device
        attn_mask = generator.wall_attention(disp_map.to(device))

        # Convert to CPU for plotting
        mask_np = attn_mask[0, 0].cpu().numpy()

        # Plot and save
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_np, cmap='hot')
        plt.title("Learned Wall Attention")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

        
def visualize_attention_overlay(sample_disp, generator, save_dir, epoch=None, index=0):
    """
    Visualize attention map overlayed on displacement map with wall mask contour.

    Args:
        sample_disp (torch.Tensor): [1, 1, H, W] single displacement map on correct device.
        generator (nn.Module): Your Pix2PixGenerator_wall_attention model.
        save_path (str): Directory to save the plot.
        epoch (int, optional): Epoch number for filename.
        index (int): Index of the sample in batch for naming (if applicable).

    Returns:
        None. Saves image to save_path.
    """
    generator.eval()
    with torch.no_grad():
        attn_map = generator.wall_attention(sample_disp)

    disp_np = sample_disp[0, 0].cpu().numpy()
    attn_np = attn_map[0, 0].cpu().numpy()

    plt.figure(figsize=(6, 5))
    base = plt.imshow(disp_np, cmap='gray')
    overlay = plt.imshow(attn_np, cmap='hot', alpha=0.5)

    title = f"Attention Overlay — Epoch {epoch}" if epoch is not None else "Attention Overlay"
    plt.title(title)

    # Now use the overlay as the mappable for the colorbar
    plt.colorbar(overlay, label='Attention')
    plt.tight_layout()

    fname = f"attn_epoch_{epoch:03d}_idx{index}.png" if epoch is not None else "attn_overlay.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


def visualize_latent_effect(generator, disp_map, label, z_dim=8, n_variants=6, save_dir="./outputs"):
    """
    Generates and visualizes the effect of different latent vectors (z) on the generator output
    for a single input displacement map and label. Outputs a horizontal image grid showing
    how the generator's outputs vary with different z values.

    Args:
        generator (nn.Module): Trained generator model.
        disp_map (Tensor): Input displacement map of shape [1, H, W].
        label (int): Class label corresponding to the input.
        z_dim (int): Dimensionality of the latent vector z.
        n_variants (int): Number of different z samples to generate outputs for.
        save_dir (str): Directory where the visualization image will be saved.

    Returns:
        None. Saves a visualization image named 'z_variation.png' in the specified save_dir.
    """
    generator.eval()
    device = next(generator.parameters()).device
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "z_variation.png")

    # Prepare input batch
    disp_map = disp_map.unsqueeze(0).to(device)         # [1, 1, H, W]
    label = torch.tensor([label], dtype=torch.long).to(device)

    variants = []
    with torch.no_grad():
        for _ in range(n_variants):
            z = torch.randn(1, z_dim, device=device)
            fake_img = generator(disp_map, label, z=z)
            variants.append(fake_img.squeeze(0).cpu())  # [1, H, W]

    # Stack & visualize
    grid = make_grid(variants, nrow=n_variants, normalize=True, scale_each=True)
    plt.figure(figsize=(2 * n_variants, 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Variation with Different z (Same Input)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()