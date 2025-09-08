# Import Libraries
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

# Import Finished

def generate_signed_displacement_map(image_np):
    """
    Compute a signed vertical displacement map using a Sobel operator.

    Args:
        image_np (np.ndarray): Grayscale image (2D)

    Returns:
        np.ndarray: Displacement map scaled to [0, 255], where 128 is neutral
    """
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)

    max_val = np.max(np.abs(sobel_y))
    if max_val == 0:
        return np.full_like(image_np, 128, dtype=np.uint8)

    norm = sobel_y / max_val
    norm_shifted = ((norm + 1) / 2) * 255
    return norm_shifted.astype(np.uint8)


def generate_displacement_map(image_np):
    """
    Compute an unsigned vertical displacement map (magnitude only).

    Args:
        image_np (np.ndarray): Grayscale image (2D)

    Returns:
        np.ndarray: Normalized displacement magnitude [0, 255]
    """
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)
    abs_sobel = np.abs(sobel_y)

    if abs_sobel.ptp() == 0:
        return np.zeros_like(image_np, dtype=np.uint8)

    norm = ((abs_sobel - abs_sobel.min()) / abs_sobel.ptp()) * 255
    return norm.astype(np.uint8)


def batch_generate_displacement_maps(
    input_root,
    output_root,
    classes,
    use_signed=True
):
    """
    Batch-process images into displacement maps.

    Parameters:
        input_root (str): Path to the folder containing class folders with .png images
        output_root (str): Output path to save displacement maps
        classes (list of str): List of subfolder names (domain classes)
        use_signed (bool): Whether to use signed or unsigned gradient
    """
    os.makedirs(output_root, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)

    for cls in classes:
        input_dir = os.path.join(input_root, cls)
        output_dir = os.path.join(output_root, cls)
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".png")]

        for fname in tqdm(filenames, desc=f"Processing {cls}"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            img = Image.open(in_path).convert("L")
            img_np = np.array(img)

            if use_signed:
                disp_map = generate_signed_displacement_map(img_np)
            else:
                disp_map = generate_displacement_map(img_np)

            Image.fromarray(disp_map).save(out_path)
