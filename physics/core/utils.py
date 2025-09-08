# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Import Finished

def render_stem_image(
    atoms_projected,
    dim: int = 256,
    base_brightness: dict = None,
    base_sigma: dict = None
):
    """
    Improved STEM image renderer with per-element Gaussiaan blur and brightness,
    and subpixel placement to reduce aliasing artifacts.

    Parameters:
        atoms_projected: list of (element, x_frac, y_frac, z_frac)
        dim: resolution of output image
        base_brightness: optional dict for brightness per element
        base_sigma: optional dict for (sigma_y, sigma_z) per element

    Returns:
        np.ndarray: 2D normalized image
    """
    if base_brightness is None:
        base_brightness = {"Y": 1.5, "Mn": 0.9}
    if base_sigma is None:
        base_sigma = {"Y": (1.2, 2.0), "Mn": (1.0, 1.5)}

    image = np.zeros((dim, dim), dtype=np.float32)

    for atom, _, y_frac, z_frac in atoms_projected:
        y_pix = int(round(y_frac * dim))
        z_pix = int(round(z_frac * dim))

        if 0 <= y_pix < dim and 0 <= z_pix < dim:
            brightness = base_brightness.get(atom, 1.0)
            sigma = base_sigma.get(atom, (1.2, 1.0))

            blob = np.zeros_like(image)
            blob[z_pix, y_pix] = brightness
            image += gaussian_filter(blob, sigma=sigma)

    return np.clip(image / np.max(image), 0, 1)


def apply_image_augmentation(image, noise_type="gaussian", noise_param=0.03, gamma=1.0):
    """
    Apply noise and contrast adjustment to an image.

    Parameters:
        image (np.ndarray): Input image
        noise_type (str): "gaussian" or "poisson"
        noise_param (float): Std for gaussian, scale for poisson
        gamma (float): Gamma correction factor

    Returns:
        np.ndarray: Augmented image
    """
    if noise_type == "gaussian":
        image = np.clip(image + np.random.normal(0, noise_param, image.shape), 0, 1)
    elif noise_type == "poisson":
        image = np.random.poisson(image * noise_param) / noise_param
        image = np.clip(image, 0, 1)
    
    return np.clip(image ** gamma, 0, 1)


def show_stem_image(image, title="STEM Image", figsize=(6, 6)):
    """
    Display a STEM image using matplotlib.

    Parameters:
        image (np.ndarray): STEM image
        title (str): Title for the plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray', origin='lower')
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_yz_projection_scatter(atoms_projected):
    """
    Display a YZ scatter plot of Y and Mn atoms.

    Parameters:
        atoms_projected (list): List of (element, x_frac, y_frac, z_frac)
    """
    y_coords = [(y, z) for elem, _, y, z in atoms_projected if elem == "Y"]
    mn_coords = [(y, z) for elem, _, y, z in atoms_projected if elem == "Mn"]

    plt.figure(figsize=(6, 8))
    if y_coords:
        y_y, y_z = zip(*y_coords)
        plt.scatter(y_y, y_z, label='Y', marker='o', s=20, color='blue')
    if mn_coords:
        mn_y, mn_z = zip(*mn_coords)
        plt.scatter(mn_y, mn_z, label='Mn', marker='s', s=20, color='green')

    plt.xlabel('y (fractional)')
    plt.ylabel('z (fractional)')
    plt.title('Projected Y and Mn Atoms in yz Plane')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()
