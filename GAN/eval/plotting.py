# Import Libraries
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import seaborn as sns
from pytorch_msssim import ssim
from scipy.stats import ttest_ind
from itertools import combinations
from skimage.metrics import structural_similarity as compare_ssim

# Import Finished

def show_pix2pix_outputs(generator, dataset, num_samples=4, n_classes=4, device="cuda", save_dir="outputs", filename="pix2pix_outputs.png"):
    """
    Visualize Pix2Pix generator output against input and target for a few samples.

    For each sample:
        - Column 1: Displacement map (input)
        - Column 2: Generated STEM image (output)
        - Column 3: Ground truth STEM image

    Args:
        generator (nn.Module): Trained Pix2PixGenerator model.
        dataset (Dataset): Dataset containing (disp_map, real_img, label).
        num_samples (int): Number of samples to display.
        n_classes (int): Total number of classes (to filter invalid labels).
        device (str): Device to run inference on.
        save_dir (str): Directory to save the output figure.
        filename (str): File name for the saved figure.
    """
    generator.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))  # disp → fake → real

    for i in range(num_samples):
        disp_map, real_img, label, _ = dataset[i]
        disp_map = disp_map.unsqueeze(0).to(device)  # add batch dim
        label = torch.tensor([label], device=device)

        if label.item() >= n_classes:
            print(f"Skipping sample {i} due to invalid label: {label.item()}")
            continue
        
        with torch.no_grad():
            fake_img = generator(disp_map, label).cpu()

        axs[i, 0].imshow(disp_map.cpu().squeeze(), cmap='viridis')
        axs[i, 0].set_title(f"Disp Map (Label: {label.item()})")
        axs[i, 1].imshow(fake_img.squeeze(), cmap='gray')
        axs[i, 1].set_title("Generated Image")
        axs[i, 2].imshow(real_img.squeeze(), cmap='gray')
        axs[i, 2].set_title("Real Image")

        for j in range(3):
            axs[i, j].axis("off")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=200)
    print(f"Saved Pix2Pix output visualization to: {save_path}")

    plt.show()


def visualize_per_two_class_samples(generator, dataloader, device, n_classes=4, num_samples=3):
    """
    Display generator outputs per class label (e.g., UP and DN).

    For each class:
        - Select up to `num_samples` examples.
        - Show input displacement map, generated image, and real image.

    This helps inspect how well the generator handles class conditioning.

    Args:
        generator (nn.Module): Trained Pix2PixGenerator.
        dataloader (DataLoader): DataLoader for evaluation set.
        device (str): Device to run inference on.
        num_samples (int): Max number of samples per class to display.
    """
    generator.eval()
    with torch.no_grad():
        for disp, real, labels, _ in dataloader:
            disp = disp.to(device)
            real = real.to(device)
            labels = labels.to(device)
            assert labels.max().item() < n_classes, f"Label {labels.max().item()} exceeds allowed class index {n_classes - 1}"
            fake = generator(disp, labels)

            for cls in [0, 1]:  # UP and DN
                idx = (labels == cls).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    continue

                for i in idx[:num_samples]:
                    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
                    axs[0].imshow(disp[i].cpu().squeeze(), cmap='viridis')
                    axs[0].set_title(f"Disp Map (Label: {cls})")
                    axs[1].imshow(fake[i].cpu().squeeze(), cmap='gray')
                    axs[1].set_title("Generated Image")
                    axs[2].imshow(real[i].cpu().squeeze(), cmap='gray')
                    axs[2].set_title("Real Image")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()
            break  # just 1 batch


def visualize_per_class_samples(generator, dataloader, device, n_classes=4, num_samples=3, save_dir="pix2pix_output"):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    class_counts = {cls: 0 for cls in range(n_classes)}

    with torch.no_grad():
        for disp, real, labels, _ in dataloader:
            disp = disp.to(device)
            real = real.to(device)
            labels = labels.to(device)
            assert labels.max().item() < n_classes, f"Label {labels.max().item()} exceeds allowed class index {n_classes - 1}"
            fake = generator(disp, labels)

            for i in range(labels.size(0)):
                cls = labels[i].item()
                if class_counts[cls] >= num_samples:
                    continue

                fig, axs = plt.subplots(1, 3, figsize=(10, 3))
                axs[0].imshow(disp[i].cpu().squeeze(), cmap='viridis')
                axs[0].set_title(f"Disp Map (Label: {cls})")
                axs[1].imshow(fake[i].cpu().squeeze(), cmap='gray')
                axs[1].set_title("Generated Image")
                axs[2].imshow(real[i].cpu().squeeze(), cmap='gray')
                axs[2].set_title("Real Image")

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"class{cls}_sample{class_counts[cls]}.png"))
                plt.show()
                plt.close()

                class_counts[cls] += 1

            # Stop if all classes are covered
            if all(count >= num_samples for count in class_counts.values()):
                break



def evaluate_pix2pix_quality(generator, dataloader, device, save_dir, n_classes = 4, threshold=0.8, max_samples=None, save_bad=True):
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)
    bad_dir = os.path.join(save_dir, "bad_samples")
    if save_bad:
        os.makedirs(bad_dir, exist_ok=True)

    results = []
    bad_samples = []

    idx = 0
    with torch.no_grad():
        for disp, real, labels, _ in dataloader:
            disp = disp.to(device)
            real = real.to(device)
            labels = labels.to(device)
            assert labels.max().item() < n_classes, f"Label {labels.max().item()} exceeds allowed class index {n_classes - 1}"

            fake = generator(disp, labels)

            # Compute batch SSIM (faster)
            batch_ssim = ssim(fake, real, data_range=1.0, size_average=False)  # (B,)

            for i in range(disp.size(0)):
                ssim_val = batch_ssim[i].item()
                label = labels[i].item()

                results.append({
                    "index": idx,
                    "label": label,
                    "ssim": ssim_val
                })

                # Store info for saving bad sample later
                if ssim_val < threshold and save_bad:
                    bad_samples.append({
                        "index": idx,
                        "label": label,
                        "ssim": ssim_val,
                        "disp": disp[i].cpu(),
                        "fake": fake[i].cpu(),
                        "real": real[i].cpu()
                    })

                idx += 1
                if max_samples and idx >= max_samples:
                    break
            if max_samples and idx >= max_samples:
                break

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "eval_results.csv"), index=False)

    # Save bad samples
    if save_bad:
        for sample in bad_samples:
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(TF.to_pil_image(sample["disp"]), cmap='viridis')
            axs[0].set_title(f"Disp Map (Label: {sample['label']})")
            axs[1].imshow(TF.to_pil_image(sample["fake"]), cmap='gray')
            axs[1].set_title("Generated")
            axs[2].imshow(TF.to_pil_image(sample["real"]), cmap='gray')
            axs[2].set_title("Real")
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            filename = f"sample_{sample['index']:03d}_cls{sample['label']}_ssim{sample['ssim']:.2f}.png"
            plt.savefig(os.path.join(bad_dir, filename))
            plt.close()

    # Print summary
    for cls in sorted(df["label"].unique()):
        subset = df[df["label"] == cls]
        mean_ssim = subset["ssim"].mean()
        pct_good = (subset["ssim"] > threshold).mean() * 100
        print(f"Class {cls}: SSIM mean = {mean_ssim:.4f}, % > {threshold} = {pct_good:.1f}%")



def validate_ssim(generator, dataloader, n_classes=4, device="cuda", save_dir="./outputs"):
    """
    Evaluates SSIM between real and generated images per class label from a dataloader,
    aggregates statistics, performs pairwise t-tests, and saves a boxplot of SSIM scores.

    Args:
        generator (nn.Module): Trained generator model.
        dataloader (DataLoader): PyTorch DataLoader yielding (disp_map, real_img, labels, filenames).
        n_classes (int): Number of domain classes expected in the dataset.
        device (str): Device to run inference on ("cuda" or "cpu").
        save_dir (str): Directory where the SSIM boxplot image will be saved.

    Returns:
        pd.DataFrame: A dataframe with per-sample SSIM scores and corresponding class labels.
    """
    generator.eval()
    all_ssim = []
    all_labels = []

    os.makedirs(save_dir, exist_ok=True)
    save_plot_path = os.path.join(save_dir, "ssim_per_class.png")

    with torch.no_grad():
        for disp_map, real_img, labels, _ in dataloader:
            disp_map = disp_map.to(device)
            real_img = real_img.to(device)
            labels = labels.to(device)
            assert labels.max().item() < n_classes, f"Label {labels.max().item()} exceeds allowed class index {n_classes - 1}"

            fake_img = generator(disp_map, labels)

            for i in range(fake_img.size(0)):
                pred_np = fake_img[i].cpu().numpy().squeeze()
                real_np = real_img[i].cpu().numpy().squeeze()

                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                real_np = (real_np - real_np.min()) / (real_np.max() - real_np.min() + 1e-8)

                ssim_val = compare_ssim(pred_np, real_np, data_range=1.0)
                all_ssim.append(ssim_val)
                all_labels.append(labels[i].item())

    # Create DataFrame
    df = pd.DataFrame({"ssim": all_ssim, "label": all_labels})
    grouped = df.groupby("label")["ssim"]

    print("\n[VALIDATION SSIM STATS]")
    for label, stats in grouped.agg(["count", "mean", "std", "min", "max"]).iterrows():
        print(f"Class {label}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}")

    # Pairwise T-tests
    print("\n[PAIRWISE T-TESTS BETWEEN CLASSES]")
    for a, b in combinations(sorted(df["label"].unique()), 2):
        ssim_a = df[df["label"] == a]["ssim"]
        ssim_b = df[df["label"] == b]["ssim"]
        t_stat, p_val = ttest_ind(ssim_a, ssim_b, equal_var=False)
        print(f"Class {a} vs Class {b}: t = {t_stat:.4f}, p = {p_val:.4g}")

    # Plot and save SSIM boxplot per class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label", y="ssim", data=df)
    plt.title("SSIM Distribution per Class")
    plt.xlabel("Domain Class")
    plt.ylabel("SSIM Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_plot_path)
    plt.show()
    plt.close()

    return df


def plot_ssim_distribution(csv_path, save_dir=None, label_map=None):
    """
    Plot the distribution of SSIM scores per class from GAN evaluation results.

    Parameters
    ----------
    csv_path : str
        Path to the evaluation CSV file produced by `evaluate_pix2pix_quality`.
        The file must contain at least the columns: ["filename", "label", "ssim"].
    
    save_dir : str, optional
        If provided, saves the figure as "ssim_distribution_by_class.png" inside this folder.
        If None, the plot is only displayed.

    label_map : dict, optional
        Optional mapping {class_index: class_name}.
        If provided, replaces numeric labels in the legend with human-readable class names
        (e.g., {0: "UP", 1: "DN", 2: "typeC-DW"}).
    
    Returns
    -------
    None
        Displays (and optionally saves) a histogram of SSIM scores for each class.

    Notes
    -----
    - SSIM (Structural Similarity Index Measure) is a perceptual similarity metric
      that compares two images based on luminance, contrast, and structure.
      It ranges from:
        * -1.0 → completely different
        *  0.0 → no similarity
        *  1.0 → identical images
    - In this context, SSIM measures how close the GAN-generated image is 
      to the ground-truth STEM image for each displacement map.
    - Higher SSIM = better GAN reconstruction quality.
    """

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df, 
        x="ssim", 
        hue="label", 
        bins=25, 
        kde=True, 
        palette="Set2", 
        element="step"
    )
    plt.title("SSIM Distribution per Class")
    plt.xlabel("SSIM")
    plt.ylabel("Count")

    # Build legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if label_map:
        # Replace numeric labels with names
        labels = [label_map.get(int(l), l) for l in labels]
    plt.legend(handles, labels, title="Class")

    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "ssim_distribution_by_class.png"))

    plt.show()



def plot_loss(history, key, label, color, save_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history[key], label=label, color=color, marker='o')
    plt.title(f"{label} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{key}_plot.png"))
    plt.close()

def plot_loss_components(history, save_dir):
    keys = [
        ("loss_real", "Adversarial Real Loss", 'tab:blue'),
        ("loss_fake", "Adversarial Fake Loss", 'tab:orange'),
        ("l1_loss", "L1 Image Loss", 'tab:green'),
    ]
    for key, ylabel, color in keys:
        plt.figure(figsize=(6, 4))
        plt.plot(history[key], label=ylabel, color=color, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{key}_plot.png"))
        plt.close()

def plot_ssim_over_epochs(history, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(history["ssim_scores"], marker='o', color='tab:purple')
    plt.title("SSIM Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ssim_over_epochs.png"))
    plt.close()