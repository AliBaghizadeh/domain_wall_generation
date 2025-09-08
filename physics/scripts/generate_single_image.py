from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from PIL import Image
import numpy as np

import os
import random

# Project
from domain_wall_generation.physics.core.y_ions import get_y_ions
from domain_wall_generation.physics.core.utils import (
    render_stem_image,
    apply_image_augmentation,
    show_stem_image,
)


def generate_type_B_tdw(
    structure,
    y_b,
    y_a,
    delta_z_base: float = 0.0131,
    y_tol: float = 0.05,
    step_direction: int = -1  # -1 = left, +1 = right
):
    """
    Generate a stepped transverse domain wall (Type B).
    Based on Applied Physics Letters 106, 112903 (2015)

    - [N, ↑, ↑] units on left
    - [N, ↓, ↓] units on right
    - Wall location varies across z-layers using stepped midpoint logic
    """

    # Combine and sort all Y ions by z
    all_y = [fc for _, fc in y_b + y_a]
    all_y_sorted = sorted(all_y, key=lambda fc: fc[2])

    # Group into z-layers
    z_layer_bins = []
    for fc in all_y_sorted:
        z = fc[2]
        placed = False
        for layer in z_layer_bins:
            if abs(np.mean([a[2] for a in layer]) - z) < y_tol:
                layer.append(fc)
                placed = True
                break
        if not placed:
            z_layer_bins.append([fc])

    atoms_projected = []

    # Add Mn atoms unchanged
    for site in structure:
        if site.species_string == "Mn":
            x, y, z = site.frac_coords
            atoms_projected.append(("Mn", 0.0, y, z))

    # Sort one representative row by y to define possible domain wall positions
    row_sorted_example = sorted(z_layer_bins[0], key=lambda fc: fc[1])
    N = len(row_sorted_example)
    valid_range = list(range(3, N - 1, 3))  # valid wall positions
    base_mid_idx = random.choice(valid_range) if valid_range else 3

    # Step wall index across z-layers
    for row_idx, row in enumerate(z_layer_bins):
        row_sorted = sorted(row, key=lambda fc: fc[1])
        N = len(row_sorted)
        stepped_mid_idx = max(0, min(base_mid_idx + 3 * step_direction * row_idx, N - 3))

        for i, fc in enumerate(row_sorted):
            x_frac, y_frac, z_frac = fc
            pattern_index = i % 3

            if i < stepped_mid_idx:
                # Left: [N, ↑, ↑]
                if pattern_index == 0:
                    new_z = z_frac
                else:
                    new_z = (z_frac + delta_z_base) % 1.0
            else:
                # Right: [N, ↓, ↓]
                if pattern_index == 0:
                    new_z = z_frac
                else:
                    new_z = (z_frac - delta_z_base) % 1.0

            atoms_projected.append(("Y", x_frac, y_frac, new_z))
            atoms_projected.append(("Y", 0.0, y_frac, new_z))

    return atoms_projected

# CIF content to create a supercell

original_structure = CifParser("data/ymno3_unpolar.cif").parse_structures(primitive=False)[0]
standardized_structure = SpacegroupAnalyzer(original_structure).get_refined_structure()

# Configuration parameters
supercell_configs = [
    [1, 14, 4], [1, 10, 4],
    [1, 8, 3], [1, 7, 4],
]

supercell = standardized_structure.copy()
supercell.make_supercell(supercell_configs[2])
structure = supercell
structure_flat, y_ions = get_y_ions(structure)


#delta_z_tdw = [-0.014, 0.012, 0.016]      #[-0.016, -0.12, 0.12, 0.16]
delta_z_ldw = [-0.008, -0.011, 0.008, 0.011]

num_aug = 2
noise_levels = [0.04, 0.1]
gamma_values = [0.8, 1.2]
"""
if __name__ == "__main__":
    atoms_tdw = generate_type_B_tdw(structure_flat, y_ions["b"], y_ions["a"], delta_z_base = 0.01)
    image_tdw = render_stem_image(atoms_tdw)
    image_tdw = apply_image_augmentation(image_tdw, noise_type="gaussian", noise_param=0.02, gamma=0.7)
    show_stem_image(image_tdw, title="Type B Domain Wall")


"""
output_dir = "data/generated_images"
os.makedirs(output_dir, exist_ok=True)
if __name__ == "__main__":
    for idx, supercell_config in enumerate(supercell_configs):
        supercell = standardized_structure.copy()
        supercell.make_supercell(supercell_config)
        structure = supercell
        structure_flat, y_ions = get_y_ions(structure)

        for dz in delta_z_ldw:
            atoms = generate_type_B_tdw(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=dz)
            image = render_stem_image(atoms)

            for noise in noise_levels:
                for gamma in gamma_values:
                    aug_image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)

                    sc_str = "x".join(map(str, supercell_config))  # e.g., "1x8x4"
                    title = f"sc{sc_str}_dz{dz}_n{noise}_g{gamma}"
                    file_path = os.path.join(output_dir, f"{title}.png")

                    # Save or display
                    Image.fromarray((aug_image * 255).astype(np.uint8)).save(file_path)
                    print(f"Saved: {file_path}")
