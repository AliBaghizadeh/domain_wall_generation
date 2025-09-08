# Import Libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from kornia.filters import Sobel
from pytorch_msssim import ssim

# Project - Models
from domain_wall_generation.gan.models.base_generators.wall_attention_generator import (
    Pix2PixGenerator_wall_attention,
    estimate_wall_mask,
)

# Project - Losses and utilities
from domain_wall_generation.gan.ml_utils.losses import curvature_loss, latent_diversity_loss
from domain_wall_generation.gan.ml_utils.visualization import visualize_attention_overlay

# Project - Plotting utilities
from domain_wall_generation.gan.eval.plotting import (
    plot_loss,
    plot_loss_components,
    plot_ssim_over_epochs,
)

# Import Finished

def train_pix2pix(
    generator,
    discriminator,
    dataloader,
    n_classes,
    device,
    epochs,
    lambda_l1,
    lambda_edge,    
    lambda_ssim,
    lambda_attn,
    lambda_div,  
    lambda_curv,  
    save_dir
    ):
    
    """
    Trains a Pix2Pix or Pix2Pix-Attention GAN on a given dataset, supporting a variety of loss terms.

    The function handles both classic and attention-based Pix2Pix generators. It tracks all loss components, 
    handles saving checkpoints and sample images, and generates loss plots at the end.

    Args:
        generator (nn.Module): The generator network (Pix2Pix or Pix2PixGenerator_wall_attention).
        discriminator (nn.Module): The discriminator network.
        dataloader (torch.utils.data.DataLoader): DataLoader yielding (disp_map, real_img, labels, filenames) tuples.
        n_classes (int): Number of distinct label classes in the dataset.
        device (torch.device): PyTorch device (e.g. "cuda" or "cpu").
        epochs (int): Number of epochs to train for.
        lambda_l1 (float): Weight for the L1 reconstruction loss.
        lambda_edge (float): Weight for the edge loss (Sobel-based).
        lambda_ssim (float): Weight for the SSIM loss.
        lambda_attn (float): Weight for the attention mask loss (only for attention model).
        lambda_div (float): Weight for the latent diversity loss (only for attention model).
        lambda_curv (float): Weight for the curvature loss (only for attention model).
        save_dir (str): Directory path where samples, models, and plots are saved.

    Returns:
        None. Side effects include saving sample images, model checkpoints, and training plots to `save_dir`.

    Notes:
        - The function automatically detects whether the attention model is in use and handles losses accordingly.
        - Model checkpoints are saved every 10 epochs and at the end of training.
        - Sample output images and loss curves are saved after each epoch.
        - Compatible with both classic Pix2Pix and attention-augmented variants.
    """

    use_attention_model = isinstance(generator, Pix2PixGenerator_wall_attention)
    os.makedirs(save_dir, exist_ok=True)
    generator.to(device)
    discriminator.to(device)

    gen_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    sobel = Sobel()

    history = {
        "loss_D": [], "loss_G": [],
        "loss_real": [], "loss_fake": [],
        "l1_loss": [], "ssim_scores": [],
        "edge_loss": [], "loss_ssim":[], "loss_curv":[]
    }

    for epoch in range(epochs):
        epoch_loss_D = 0
        epoch_loss_G = 0
        epoch_loss_real = 0
        epoch_loss_fake = 0
        epoch_l1 = 0
        epoch_edge = 0
        epoch_loss_ssim = 0
        epoch_loss_curv = 0

        for i, (disp_map, real_img, labels, filenames) in enumerate(dataloader):
            assert labels.max().item() < n_classes, f"Label value {labels.max().item()} exceeds n_classes={n_classes}"
            disp_map = disp_map.to(device)
            real_img = real_img.to(device)
            labels = labels.to(device)

            #Switching between Attention mechanism or no attention
            if use_attention_model:
                z_wall = torch.randn(disp_map.size(0), 8, device=device)
                #create fake_img
                fake_img, attn_mask = generator(disp_map, labels, z=z_wall, return_attn=True)
                wall_mask = estimate_wall_mask(disp_map).to(device)  # [0,1] mask
                loss_attn = F.mse_loss(attn_mask, wall_mask) if attn_mask is not None else 0.0
                loss_div = latent_diversity_loss(generator, disp_map, labels, z_dim=8)
                loss_curv = curvature_loss(fake_img)
            else:
                fake_img = generator(disp_map, labels)
                attn_mask = None
                wall_mask = None
                loss_attn = 0.0
                loss_div = 0.0
                loss_curv = 0.0
            # --- Setup real/fake labels ---
            with torch.no_grad():
                pred_shape = discriminator(disp_map, real_img, labels).shape
            valid = torch.ones(pred_shape, device=device)
            fake = torch.zeros_like(valid)

            # --- Train Generator ---
            generator.zero_grad()

            pred_fake = discriminator(disp_map, fake_img, labels)
            loss_adv = bce_loss(pred_fake, valid)
            loss_l1 = l1_loss(fake_img, real_img)
            loss_edge = F.l1_loss(sobel(fake_img), sobel(real_img))
            loss_ssim = 1.0 - ssim((fake_img + 1) / 2, (real_img + 1) / 2, data_range=1.0)
            
            #Total Loss
            loss_G = (
            loss_adv +
            lambda_l1 * loss_l1 +
            lambda_edge * loss_edge +
            lambda_ssim * loss_ssim +
            (lambda_attn * loss_attn if use_attention_model else 0) +
            (lambda_div * loss_div if use_attention_model else 0) +
            (lambda_curv * loss_curv if use_attention_model else 0)
            )
                    
            loss_G.backward()
            gen_opt.step()

            # --- Train Discriminator ---
            discriminator.zero_grad()
            pred_real = discriminator(disp_map, real_img, labels)
            pred_fake_detach = discriminator(disp_map, fake_img.detach(), labels)
            loss_real = bce_loss(pred_real, valid)
            loss_fake = bce_loss(pred_fake_detach, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            disc_opt.step()

            # --- Accumulate loss ---
            epoch_loss_D      += float(loss_D)
            epoch_loss_G      += float(loss_G)
            epoch_loss_real   += float(loss_real)
            epoch_loss_fake   += float(loss_fake)
            epoch_l1          += float(loss_l1)
            epoch_edge        += float(loss_edge)
            epoch_loss_ssim   += float(loss_ssim)
            epoch_loss_curv   += float(loss_curv)           

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {i+1}/{len(dataloader)} | "
                      f"D Loss: {loss_D:.4f} | G Loss: {loss_G:.4f} | Edge: {loss_edge:.4f}")

        # --- SSIM Evaluation ---
        with torch.no_grad():
            test_disp, test_real, test_labels, test_filenames = next(iter(dataloader))
            test_disp = test_disp.to(device)
            test_real = test_real.to(device)
            test_labels = test_labels.to(device)
            
            if use_attention_model:
                z = torch.randn(test_disp.size(0), 8, device=device)
                test_fake = generator(test_disp, test_labels, z=z)
            else:
                test_fake = generator(test_disp, test_labels)

            ssim_val = ssim(test_fake, test_real, data_range=1.0, size_average=True)
            history["ssim_scores"].append(ssim_val.item())
            print(f"Epoch {epoch+1} SSIM: {ssim_val.item():.4f}")

        # Log epoch
        N = len(dataloader)
        history["loss_D"].append(epoch_loss_D / N)
        history["loss_G"].append(epoch_loss_G / N)
        history["loss_real"].append(epoch_loss_real / N)
        history["loss_fake"].append(epoch_loss_fake / N)
        history["l1_loss"].append(epoch_l1 / N)
        history["edge_loss"].append(epoch_edge / N)
        history["loss_ssim"].append(epoch_loss_ssim / N)
        history["loss_curv"].append(epoch_loss_curv / N)

        # Save sample images
        with torch.no_grad():
            sample = generator(test_disp[:16], test_labels[:16])
            save_image(sample, os.path.join(save_dir, f"epoch_{epoch+1:03d}.png"),normalize=True, nrow=8)

        #track how the attention map evolves over training,
        if use_attention_model and epoch % 1 == 0:
            sample_disp = disp_map[:1]
            visualize_attention_overlay(sample_disp, generator, save_dir=save_dir, epoch=epoch+1, index=0)    
            
        if (epoch + 1) % 10 == 0:
            model_dir = os.path.join(save_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_epoch_{epoch+1:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch+1:03d}.pth"))
        
    # Save final models
    torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))

    plot_loss(history, "loss_G", "Generator Loss", 'tab:blue', save_dir)
    plot_loss(history, "loss_D", "Discriminator Loss", 'tab:orange', save_dir)
    plot_loss(history, "l1_loss", "L1 Loss", 'tab:green', save_dir)
    plot_loss(history, "edge_loss", "Edge Loss", 'tab:purple', save_dir)
    plot_loss(history, "ssim_scores", "SSIM", 'tab:brown', save_dir)
    plot_loss(history, "loss_ssim", "SSIM Loss", 'tab:olive', save_dir)
    
    plot_loss_components(history, save_dir)
    plot_ssim_over_epochs(history, save_dir)