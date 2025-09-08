from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Project
from domain_wall_generation.physics.core.domain_walls import generate_type_c_dw
from domain_wall_generation.physics.core.utils import (
    render_stem_image,
    apply_image_augmentation,
)
from domain_wall_generation.physics.core.y_ions import get_y_ions
from domain_wall_generation.gan.ml_utils.augmentation import (
    Apply_Augment,
    SaltPepperNoise,
    AtomicPlaneDistortionFixed,
    AtomicPlaneDistortionRandom,
    ScanDistortion,
    DriftDistortion,
)


def visualize_single_augmented_image(
    atoms,
    render_stem_image,
    apply_image_augmentation,
    augmentation_pipeline=None,
    noise=0.01,
    gamma=1.0,
    label='typeC-DW',
    dz=0.0,
    show=True,
    save_path=None,
    title_suffix=""
):
    """
    Generate and plot a single (augmented) image to visualize
    the effect of parameters and augmentations.

    Parameters
    ----------
    atoms : list
        List of atom tuples (e.g., from your domain wall generator).
    render_stem_image : callable
        Function to render the image from atomic positions.
    apply_image_augmentation : callable
        Function for basic noise/gamma (simulated STEM artifacts).
    augmentation_pipeline : callable or None
        Full augmentation pipeline (or None for no further augmentation).
    noise : float
        Gaussian noise level.
    gamma : float
        Gamma (contrast) parameter.
    label : str
        Domain label (for plot/file naming).
    dz : float
        Displacement parameter (for plot/file naming).
    show : bool
        If True, shows the image using matplotlib.
    save_path : str or None
        File path to save the image.
    title_suffix : str
        Additional text for the plot title.

    Returns
    -------
    image : np.ndarray
        The final augmented image (as uint8 array).
    """
    np.random.seed(42)
    image = render_stem_image(atoms)
    image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
    if augmentation_pipeline is not None:
        image = augmentation_pipeline(image)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    if save_path:
        Image.fromarray(image).save(save_path)
    if show:
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.title(f"{label}: dz={dz}, noise={noise}, gamma={gamma} {title_suffix}")
        plt.axis('off')
        plt.show()
    return image




def make_aug_pipeline():
    return Apply_Augment([
        SaltPepperNoise(prob=0.04),
        AtomicPlaneDistortionFixed(frequency=10, intensity=0.05),
        AtomicPlaneDistortionRandom(probability=0.1, intensity=0.1),
        ScanDistortion(frequency=5, intensity=3),
        DriftDistortion(frequency=6, intensity=2),
    ])

augmentation_pipeline = Apply_Augment([
    SaltPepperNoise(prob=0.01),
    AtomicPlaneDistortionFixed(frequency=10, intensity=0.03),
    AtomicPlaneDistortionRandom(probability=0.05, intensity=0.3),
    ScanDistortion(frequency=10, intensity=2),
    DriftDistortion(frequency=10, intensity=2),
])



supercell_configs_all = [
    [1, 14, 4], [1, 10, 4],
    [1, 9, 4], [1, 8, 3], 
    [1, 7, 4], [1, 6, 4]
]



delta_z_typec = 0.009 # [-0.009, -0.014,  0.009, 0.014] 
noise_levels = 0.050 #[0.01, 0.05, 0.07]   #0.04, 0.07, 0.1]
gamma_values = 0.8 #[0.5, 0.8, 1.0]   
dz = delta_z_typec
single_config = supercell_configs_all[3]
output_dir="outputs/stem_domain/"

# Load your paraelectric CIF file
cif_path = "data/ymno3_unpolar.cif"
original_structure = CifParser(cif_path).parse_structures(primitive=False)[0]
structure = original_structure.copy()
structure.make_supercell(single_config)
structure_flat, y_ions = get_y_ions(structure)
atoms_c_dw = generate_type_c_dw(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=dz)


# Visualize with augmentation:
visualize_single_augmented_image(
    atoms=atoms_c_dw,
    render_stem_image=render_stem_image,
    apply_image_augmentation=apply_image_augmentation,
    augmentation_pipeline=augmentation_pipeline,
    noise=noise_levels,
    gamma=gamma_values,
    label='typeC-DW',
    dz=dz,
    show=True,
    save_path=None,  # Or a path if you want to save
    title_suffix="with full pipeline"
)

