# Import Libraries
import os
import torch
from torch.utils.data import DataLoader

# Project
from domain_wall_generation.gan.models.base_generators.sharp_pix2pix import Pix2PixGenerator
from domain_wall_generation.gan.models.base_generators.wall_attention_generator import Pix2PixGenerator_wall_attention
from domain_wall_generation.gan.models.discriminator import PatchGANDiscriminator
from domain_wall_generation.gan.train.train_pix2pix import train_pix2pix
from domain_wall_generation.config_labels import POLAR_LABELS
from domain_wall_generation.gan.ml_utils.dataset import (
    DisplacementToImageDataset,
    RemappedDatasetWithFilenames,
    filter_dataset_by_classnames,
)
from domain_wall_generation.gan.ml_utils.transforms import JointTransform
# Import Finished

# === Main Training Function ===
def main(
    disp_root=None,
    image_root=None,
    target_classes=["typeC-DW"],
    save_dir="./output",
    epochs=30,
    batch_size=32,
    lambda_l1=30.0,
    lambda_edge=15.0,
    lambda_ssim=10.0,
    lambda_attn=3.0,
    lambda_div=5.0,
    lambda_curv=10.0,
    noise_scale=0.4,
    dataset_override=None,
):

    # === Setup ===
    model_type = "complex" if any(cls in ["typeC-DW"] for cls in target_classes) else "sharp"
    n_classes = len(target_classes)
    #save_dir = os.path.join(save_dir, f"{model_type}_pix2pix")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset + Transform ===
    if dataset_override is not None:
        print("⚡ Using prebuilt dataset (override mode)")
        dataset = dataset_override
    else:
        transform = JointTransform(p_flip=0.0, rotation=10)
        full_dataset = DisplacementToImageDataset(image_root=image_root, disp_root=disp_root, transform=transform)

        label_indices = {k: v for k, v in POLAR_LABELS.items()}
        selected_indices = {label_indices[name] for name in target_classes}
        filtered_subset, _ = filter_dataset_by_classnames(full_dataset, target_classes, label_indices)

        dataset = RemappedDatasetWithFilenames(filtered_subset, full_dataset, selected_indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === Models ===
    if model_type == "complex":
        generator = Pix2PixGenerator_wall_attention(in_channels=1, out_channels=1, n_classes=n_classes).to(device)
        generator.use_learned_attention = True
        generator.noise_scale = 0.4
    else:
        generator = Pix2PixGenerator(in_channels=1, out_channels=1, n_classes=n_classes).to(device)

    discriminator = PatchGANDiscriminator(in_channels=2, n_classes=n_classes).to(device)

    # === Save config ===
    use_learned_attention = getattr(generator, "use_learned_attention", False)
    noise_scale = getattr(generator, "noise_scale", 0.0)

    config_text = f"""
    Training Configuration
    =========================
    epochs:               {epochs}
    lambda_l1:            {lambda_l1}
    lambda_edge:          {lambda_edge}
    lambda_ssim:          {lambda_ssim}
    lambda_attn:          {lambda_attn}
    lambda_div:           {lambda_div}
    lambda_curv:          {lambda_curv}
    target_classes:       {target_classes}
    n_classes:            {n_classes}
    use_learned_attention:{use_learned_attention}
    noise_scale:          {noise_scale}
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write(config_text)
    print(f"✅ Saved training configuration to {config_path}")

    # === Train ===
    train_pix2pix(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        n_classes=n_classes,
        device=device,
        epochs=epochs,
        save_dir=save_dir,
        lambda_l1=lambda_l1,
        lambda_edge=lambda_edge,
        lambda_ssim=lambda_ssim,
        lambda_attn=lambda_attn,
        lambda_div=lambda_div,
        lambda_curv=lambda_curv
    )

# Entry point for command line or notebook
if __name__ == "__main__":
    main()
