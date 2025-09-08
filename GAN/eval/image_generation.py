# Import Libraries
import os
import csv
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from domain_wall_generation.config_labels import POLAR_LABELS
# Import Finished


def generate_gan_images(
    generator, gan_loader, remapped_label_names, POLAR_LABELS, 
    save_dir, device, 
    use_z=True, num_variants=5, z_scale=0.4, save_z=False
):
    """
    Generate and save synthetic images from a GAN generator, with or without stochastic latent input z.

    Args:
        generator: Your trained generator model (in eval mode, on correct device)
        gan_loader: DataLoader yielding (x, y, label, filename) or similar batches
        remapped_label_names: dict mapping label idx to class name
        POLAR_LABELS: dict for fallback class names
        save_dir: root directory for saving results
        device: torch device
        use_z: bool, whether to use stochastic z input
        num_variants: number of z samples per input (if use_z), otherwise 1
        z_scale: scaling for latent z
        save_z: bool, whether to save the z vector alongside image (as .npy)
        writes a metadata.csv for each class folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(gan_loader):
            x = batch[0].to(device)
            labels = batch[2].to(device)
            for z_idx in range(num_variants if use_z else 1):
                if use_z:
                    z = torch.randn(x.shape[0], 8).to(device) * z_scale
                else:
                    z = torch.zeros(x.shape[0], 8).to(device)
                output = generator(x, labels, z=z, return_attn=False)
                for i in range(output.shape[0]):
                    label_idx = labels[i].item()
                    class_name = remapped_label_names[label_idx] if remapped_label_names else \
                                 [k for k, v in POLAR_LABELS.items() if v == label_idx][0]
                    class_save_dir = os.path.join(save_dir, class_name)
                    os.makedirs(class_save_dir, exist_ok=True)

                    img_tensor = output[i].cpu().detach().squeeze()
                    img_array = np.clip(((img_tensor + 1) / 2 * 255).cpu().numpy(), 0, 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array)
                    
                    # Set filename
                    if use_z:
                        img_fn = f"gen_{batch_idx:03d}_{i:02d}_z{z_idx}.png"
                    else:
                        img_fn = f"gen_{batch_idx:03d}_{i:02d}_det.png"
                    img_path = os.path.join(class_save_dir, img_fn)
                    img_pil.save(img_path)

                    # Prepare z info
                    z_vec = z[i].cpu().numpy()
                    if save_z and use_z:
                        z_fn = f"gen_{batch_idx:03d}_{i:02d}_z{z_idx}.npy"
                        z_path = os.path.join(class_save_dir, z_fn)
                        np.save(z_path, z_vec)
                        z_str = z_fn
                    else:
                        # Save as comma-separated string
                        z_str = ",".join([f"{v:.5f}" for v in z_vec])

                    # Metadata csv path
                    metadata_path = os.path.join(class_save_dir, "metadata.csv")
                    file_exists = os.path.isfile(metadata_path)
                    # Write metadata row
                    with open(metadata_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            # Write header
                            writer.writerow([
                                "filename", "label", "batch_idx", "variant_idx", "z_vector"
                            ])
                        writer.writerow([
                            img_fn, class_name, batch_idx, z_idx, z_str
                        ])

def visualize_stochastic_outputs(
    generator, dataloader, device,
    n_classes=7, num_samples=3, num_z=5, save_dir="stochastic_vis"
):
    """
    For each class, show (num_samples) displacement maps.
    For each, generate (num_z) images with different z.
    """
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    class_counts = {cls: 0 for cls in range(n_classes)}
    sample_indices = {cls: [] for cls in range(n_classes)}
    disp_maps, labels_all = [], []

    # Gather up to num_samples indices per class
    with torch.no_grad():
        for disp, _, labels, _ in dataloader:
            for i in range(labels.size(0)):
                cls = labels[i].item()
                if class_counts[cls] < num_samples:
                    disp_maps.append(disp[i])
                    labels_all.append(cls)
                    sample_indices[cls].append(len(disp_maps) - 1)
                    class_counts[cls] += 1
            if all([class_counts[c] >= num_samples for c in class_counts]):
                break

    for cls in range(n_classes):
        for idx_in_class, idx in enumerate(sample_indices[cls]):
            disp_tensor = disp_maps[idx].unsqueeze(0).to(device)
            label_tensor = torch.tensor([cls]).to(device)
            fig, axs = plt.subplots(1, num_z + 1, figsize=(3 * (num_z + 1), 3))
            axs[0].imshow(disp_tensor.cpu().detach().squeeze().numpy(), cmap='viridis')
            axs[0].set_title(f"Disp (Label {cls})")
            axs[0].axis('off')

            for z_idx in range(num_z):
                z = torch.randn(1, 8).to(device) * 0.4
                fake = generator(disp_tensor, label_tensor, z=z)
                axs[z_idx + 1].imshow(fake.cpu().detach().squeeze().numpy(), cmap='gray')
                axs[z_idx + 1].set_title(f"z#{z_idx}")
                axs[z_idx + 1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"class{cls}_sample{idx_in_class}_stochastic.png"))
            plt.show()
            plt.close()
