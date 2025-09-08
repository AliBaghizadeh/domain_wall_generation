# Import Libraries
import numpy as np
import os
import random
import time
from PIL import Image
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter, CifParser

from concurrent.futures import ProcessPoolExecutor
import os
from PIL import Image
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

from domain_wall_generation.physics.core.y_ions import get_y_ions
from domain_wall_generation.physics.core.utils import render_stem_image, apply_image_augmentation
from domain_wall_generation.gan.ml_utils.augmentation import (
    Apply_Augment,
    SaltPepperNoise,
    AtomicPlaneDistortionFixed,
    AtomicPlaneDistortionRandom,
    ScanDistortion,
    DriftDistortion,
)

# Import Finished

def generate_tail_to_tail_domain_wall(
    structure,
    b_position,
    a_position,
    delta_z_base: float = 0.0395,
):
    """
    Generate a tail-to-tail domain wall by displacing Y ions from neutral positions.

    Bottom half: switch to downward polarization.
    Top half: slightly enhance upward polarization.

    Parameters:
        structure: pymatgen Structure
        b_position: list of (index, frac_coords) for Wyckoff 4b Y ions
        a_position: list of (index, frac_coords) for Wyckoff 2a Y ions
        delta_z_base: base z displacement in Angstroms

    Returns:
        atoms_projected: list of (element, x_frac, y_frac, new_z_frac)
    """
    lattice = structure.lattice
    all_y = [(fc, 'b') for _, fc in b_position] + [(fc, 'a') for _, fc in a_position]

    # Compute z_cart and determine mid-plane split
    all_y_cart = [(fc, label, lattice.get_cartesian_coords(fc)[2]) for fc, label in all_y]
    z_values = [z_cart for _, _, z_cart in all_y_cart]
    z_split = (min(z_values) + max(z_values)) / 2
    
    atoms_projected = []

    # Add Mn atoms without modification
    for site in structure:
        if site.species_string == "Mn":
            _, y, z = site.frac_coords
            atoms_projected.append(("Mn", 0.0, y, z))

    # Process Y atoms
    for fc, label, z_cart in all_y_cart:
        x_frac, y_frac, _ = fc

        if label == "a":
            new_z_frac = fc[2]
        else:
            # Shift upward or downward from neutral
            sign = 1 if z_cart >= z_split else -1
            cart_coords = lattice.get_cartesian_coords(fc)
            cart_coords[2] += sign * delta_z_base
            new_frac = lattice.get_fractional_coords(cart_coords)
            new_z_frac = new_frac[2] % 1.0

        atoms_projected.append(("Y", x_frac, y_frac, new_z_frac))
        atoms_projected.append(("Y", 0.0, y_frac, new_z_frac))

    return atoms_projected

def generate_type_c_dw(
    structure,
    y_b,
    y_a,
    delta_z_base: float = 0.0131,
    y_tol: float = 0.05,
    step_direction: int = -1  # -1 = left, +1 = right
):
    """
    Definition:
    Scientific Reports volume 3, Article number: 2741 (2013)   https://doi.org/10.1038/srep02741
    Applied Physics Letters 106, 112903 (2015); doi: 10.1063/1.4915259
    type-C DW is a mixed DW composed of both TDWs and LDWs.
    Generate a Type-A transverse domain wall (TDW, Type I) with custom asymmetric pattern:
    [... neutral, ↑, ↑, neutral, ↑, ↑, neutral, ↑, neutral, neutral, ↑, neutral, neutral...]

    Parameters:
        structure: pymatgen Structure
        y_b: list of (index, frac_coords) for Wyckoff 4b Y ions
        y_a: list of (index, frac_coords) for Wyckoff 2a Y ions
        delta_z_base: z displacement in fractional units for polarization
        y_tol: tolerance to group Y atoms into z-layers

    Returns:
        atoms_projected: list of (element, x_frac, y_frac, z_frac)
    """

    import numpy as np

    # Combine and sort Y ions
    all_y = [fc for _, fc in y_b + y_a]
    all_y_sorted = sorted(all_y, key=lambda fc: fc[2])  # by z

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

    # Add Mn atoms
    for site in structure:
        if site.species_string == "Mn":
            x, y, z = site.frac_coords
            atoms_projected.append(("Mn", 0.0, y, z))
    #selects and sorts the first z-layer (z_layer_bins[0]) by fractional y-coordinate -> a list of atoms in left-to-right order in y.
    row_sorted_example = sorted(z_layer_bins[0], key=lambda fc: fc[1])
    #N is the number of Y atoms in that row — used to know how many possible positions are available for the domain wall in index space.
    N = len(row_sorted_example)
    #defines a list of valid starting indices for the domain wall: Starts at 3 (to avoid placing wall too far left). Ends at N - 1.
    #Steps by 3 to align with the [N, ↑, ↑] pattern. if N = 30, this gives [3, 6, 9, 12, 15, 18, 21].
    valid_range = list(range(3, N - 1, 3))  # allow wall closer to edge (possible truncation)

    #valid_range = list(range(3, N - 6, 3))
    #Randomly selects a starting wall index from valid_range. Fallbacks to 3 if the range is empty (e.g., very narrow domain).
    #stepped_mid_idx = min(base_mid_idx + 3 * step_direction * row_idx, N)
    base_mid_idx = random.choice(valid_range) if valid_range else 3
    
    for row_idx, row in enumerate(z_layer_bins):
        row_sorted = sorted(row, key=lambda fc: fc[1])  # sort by y
        N = len(row_sorted)
    
        # Step the wall deterministically from the random base
        stepped_mid_idx = max(0, min(base_mid_idx + 3 * step_direction * row_idx, N - 3))
        right_counter = 0  # initialize before the loop

        for i, fc in enumerate(row_sorted):
            x_frac, y_frac, z_frac = fc
        
            if i <= stepped_mid_idx:
                # LEFT side: [N, ↑, ↑]
                if i % 3 == 0:
                    new_z = z_frac  # neutral
                else:
                    new_z = (z_frac + delta_z_base) % 1.0  # ↑
            else:
                # RIGHT side: fixed pattern [↑, N, N]
                pattern_index = right_counter % 3
                if pattern_index == 0:
                    new_z = (z_frac + delta_z_base) % 1.0  # ↑
                else:
                    new_z = z_frac  # neutral
                right_counter += 1
        
    
            atoms_projected.append(("Y", x_frac, y_frac, new_z))
            atoms_projected.append(("Y", 0.0, y_frac, new_z))
    
    return atoms_projected


def generate_type_A_tdw(
    structure,
    y_b,
    y_a,
    delta_z_base: float = 0.0131,
    y_tol: float = 0.05
):
    """    
    Scientific Reports volume 3, Article number: 2741 (2013)   https://doi.org/10.1038/srep02741
    Applied Physics Letters 106, 112903 (2015); doi: 10.1063/1.4915259
    Generate a Type-A transverse domain wall (TDW, Type I) with custom asymmetric pattern:
    [... neutral, ↑, ↑, neutral, ↑, ↑, neutral, ↑, neutral, neutral, ↑, neutral, neutral...]

    Parameters:
        structure: pymatgen Structure
        y_b: list of (index, frac_coords) for Wyckoff 4b Y ions
        y_a: list of (index, frac_coords) for Wyckoff 2a Y ions
        delta_z_base: z displacement in fractional units for polarization
        y_tol: tolerance to group Y atoms into z-layers

    Returns:
        atoms_projected: list of (element, x_frac, y_frac, z_frac)
    """

    # Merge all Y ions
    all_y = [fc for _, fc in y_b + y_a]
    all_y_sorted = sorted(all_y, key=lambda fc: fc[2])  # sort by z

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

    # Apply the precise TDW pattern per z-layer
    for row in z_layer_bins:
        row_sorted = sorted(row, key=lambda fc: fc[1])  # sort by y
        for i, fc in enumerate(row_sorted):
            x_frac, y_frac, z_frac = fc

            if i <= 10:
                if i % 3 == 0:
                    new_z = z_frac  # neutral
                else:
                    new_z = (z_frac + delta_z_base) % 1.0  # ↑
            else:
                pattern_index = (i - 11) % 3
                if pattern_index in [0, 1]:
                    new_z = z_frac  # neutral
                else:
                    new_z = (z_frac + delta_z_base) % 1.0  # ↑

            atoms_projected.append(("Y", x_frac, y_frac, new_z))
            atoms_projected.append(("Y", 0.0, y_frac, new_z))

    return atoms_projected


def generate_type_d_dw(
    structure,
    y_b,
    y_a,
    delta_z_base: float = 0.0131,
    y_tol: float = 0.05
):
    """
    Definition: Applied Physics Letters 106, 112903 (2015); doi: 10.1063/1.4915259
    Generate a domain wall pattern, tpye II (type D):
    - Start with [neutral, ↑, ↑] units from the left
    - Switch to [neutral, ↓, ↓] units at the midpoint
    - Ensures transition happens cleanly and starts with neutral

    Parameters:
        structure: pymatgen Structure
        y_b: list of (index, frac_coords) for Wyckoff 4b Y ions
        y_a: list of (index, frac_coords) for Wyckoff 2a Y ions
        delta_z_base: z displacement in fractional units
        y_tol: tolerance to group Y atoms into z-layers

    Returns:
        atoms_projected: list of (element, x_frac, y_frac, z_frac)
    """

    # Combine all Y ions
    all_y = [fc for _, fc in y_b + y_a]
    all_y_sorted = sorted(all_y, key=lambda fc: fc[2])  # sort by z

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

    for row in z_layer_bins:
        row_sorted = sorted(row, key=lambda fc: fc[1])  # sort by y
        N = len(row_sorted)

        # Ensure the midpoint aligns with start of unit cell Nearest lower multiple of 3
        mid_idx = (N // 2) - (N // 2) % 3

        for i, fc in enumerate(row_sorted):
            x_frac, y_frac, z_frac = fc
            pattern_index = i % 3

            if i < mid_idx:
                # Left side: [N, ↑, ↑]
                if pattern_index == 0:
                    new_z = z_frac  # neutral
                else:
                    new_z = (z_frac + delta_z_base) % 1.0
            else:
                # Right side: [N, ↓, ↓]
                if pattern_index == 0:
                    new_z = z_frac  # neutral
                else:
                    new_z = (z_frac - delta_z_base) % 1.0

            atoms_projected.append(("Y", x_frac, y_frac, new_z))
            atoms_projected.append(("Y", 0.0, y_frac, new_z))

    return atoms_projected

#This is not mentioned in the literature
def generate_type_B_ldw(
    structure,
    y_b,
    y_a,
    delta_z_base: float = 0.0131,
    y_tol: float = 0.05,
    step_direction: int = -1  # -1 = left, +1 = right
):
    """
    Definition: Applied Physics Letters 106, 112903 (2015); doi: 10.1063/1.4915259
    Scientific Reports volume 3, Article number: 2741 (2013)   https://doi.org/10.1038/srep02741
    Generate a stepped transverse domain wall (Type II).
    - Wall location varies across z-layers using stepped midpoint logic
    - Start with [neutral, ↑, ↑] units from the left
    - Switch to [neutral, ↓, ↓] units at the midpoint
    - Ensures transition happens cleanly and starts with neutral

    Parameters:
        structure: pymatgen Structure
        y_b: list of (index, frac_coords) for Wyckoff 4b Y ions
        y_a: list of (index, frac_coords) for Wyckoff 2a Y ions
        delta_z_base: z displacement in fractional units
        y_tol: tolerance to group Y atoms into z-layers

    Returns:
        atoms_projected: list of (element, x_frac, y_frac, z_frac)
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

def generate_domain_wall_dataset(
    original_structure,
    output_dir,
    supercell_configs,
    delta_z_t2t,
    delta_z_tdw,
    delta_z_typec,
    delta_z_typed,
    delta_z_updn,
    num_aug,
    noise_levels=None,
    gamma_values=None
):
    if noise_levels is None:
        noise_levels = [0.03, 0.05, 0.07]
    if gamma_values is None:
        gamma_values = [0.8, 1.0, 1.2]

    # Prepare standardized structure
    symmetry = SpacegroupAnalyzer(original_structure)
    standardized_structure = symmetry.get_refined_structure()

    # Domain types
    CLASSES = ['TT', 'HH', 'typeA-TDW', 'typeC-DW', 'typeD-DW','UP', 'DN']
    for cls in CLASSES:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    for config_id, supercell_size in enumerate(supercell_configs):
        supercell = standardized_structure.copy()
        supercell.make_supercell(supercell_size)

        suffix = ''.join(str(x) for x in supercell_size)
        cif_path = os.path.join(output_dir, f"YMnO3_supercell_{suffix}.cif")
        CifWriter(supercell).write_file(cif_path)
        structure = CifParser(cif_path).parse_structures(primitive=False)[0]
        structure_flat, y_ions = get_y_ions(structure)

        for noise in noise_levels:
            for gamma in gamma_values:
                for i in range(num_aug):
                    tag = f"n{int(noise*1000)}_g{int(gamma*100)}_aug{i}_sc{suffix}"

                    # Tail-to-tail and Head-to-head
                    for dz in delta_z_t2t:
                        for label, dz_sign in [("TT", +1), ("HH", -1)]:
                            atoms = generate_tail_to_tail_domain_wall(
                                structure_flat,
                                y_ions["b"],
                                y_ions["a"],
                                delta_z_base=dz_sign * dz
                            )
                            _save_augmented_image(atoms, noise, gamma, dz, suffix, label, output_dir, i)


                    # TDW
                    for dz in delta_z_tdw:
                        atoms = generate_type_A_tdw(
                            structure_flat,
                            y_ions["b"],
                            y_ions["a"],
                            delta_z_base=dz
                        )
                        _save_augmented_image(atoms, noise, gamma, dz, suffix, label, output_dir, i)

                    #typeC
                    for dz in delta_z_typec:
                        atoms = generate_type_c_dw(
                            structure_flat,
                            y_ions["b"],
                            y_ions["a"],
                            delta_z_base=dz
                        )
                        _save_augmented_image(atoms, noise, gamma, dz, suffix, label, output_dir, i)

                    #type D
                    for dz in delta_z_typed:
                        atoms = generate_type_d_dw(
                            structure_flat,
                            y_ions["b"],
                            y_ions["a"],
                            delta_z_base=dz
                        )
                        _save_augmented_image(atoms, noise, gamma, dz, suffix, label, output_dir, i)


                    # UP / DN
                    for dz in delta_z_updn:
                        for label, dz_sign in [("UP", -1), ("DN", +1)]:
                            atoms_projected = []

                            for site in structure:
                                if site.species_string == "Mn":
                                    _, y, z = site.frac_coords
                                    atoms_projected.append(("Mn", 0.0, y, z))

                            for _, fc in y_ions["b"]:
                                x_frac, y_frac, z_frac = np.array(fc).flatten()
                                z_new = (z_frac + dz_sign * dz) % 1.0
                                atoms_projected.append(("Y", x_frac, y_frac, z_new))
                                atoms_projected.append(("Y", 0.0, y_frac, z_new))

                            for _, fc in y_ions["a"]:
                                x_frac, y_frac, z_frac = np.array(fc).flatten()
                                atoms_projected.append(("Y", x_frac, y_frac, z_frac))
                                atoms_projected.append(("Y", 0.0, y_frac, z_frac))

                            _save_augmented_image(atoms, noise, gamma, dz, suffix, label, output_dir, i)



def _save_augmented_image(atoms_projected, noise, gamma, dz, suffix, label, output_dir, i):
    tag = f"n{int(noise * 1000)}_g{int(gamma * 100)}_aug{i}"
    fname = f"YMnO3_{label}_dz{dz:+.4f}_sc{suffix}_{tag}.png"
    fpath = os.path.join(output_dir, label, fname)
    image = render_stem_image(atoms_projected)
    image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
    Image.fromarray((image * 255).astype(np.uint8)).save(fpath)


augmentation_pipeline = Apply_Augment([
        SaltPepperNoise(prob=0.04),
        AtomicPlaneDistortionFixed(frequency=10, intensity=0.05),
        AtomicPlaneDistortionRandom(probability=0.1, intensity=0.1),
        ScanDistortion(frequency=5, intensity=3),
        DriftDistortion(frequency=6, intensity=2),
])


def generate_single_config(
    config,
    original_structure,
    output_dir,
    delta_z_t2t,
    delta_z_tdw,
    delta_z_typec,
    delta_z_typed,
    delta_z_updn,
    num_aug,
    noise_levels,
    gamma_values,
    augmentation_pipeline=augmentation_pipeline
):
    # unique RNG per worker by reseeding
    seed = (int(time.time() * 1000) + os.getpid()) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    
    suffix = ''.join(str(x) for x in config)
    structure = original_structure.copy()
    structure.make_supercell(config)

    cif_path = os.path.join(output_dir, f"YMnO3_supercell_{suffix}.cif")
    CifWriter(structure).write_file(cif_path)
    structure = CifParser(cif_path).parse_structures(primitive=False)[0]
    structure_flat, y_ions = get_y_ions(structure)

    for noise in noise_levels:
        for gamma in gamma_values:
            for i in range(num_aug):

                # Tail-to-Tail and Head-to-Head
                if delta_z_t2t:
                    for dz in delta_z_t2t:
                        atoms_tt = generate_tail_to_tail_domain_wall(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=dz)
                        atoms_hh = generate_tail_to_tail_domain_wall(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=-dz)

                        for label, atoms in [("TT", atoms_tt), ("HH", atoms_hh)]:
                            image = render_stem_image(atoms)
                            image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
                            if augmentation_pipeline is not None:
                                image = augmentation_pipeline(image)
                            image = np.clip(image * 255, 0, 255).astype(np.uint8)
                            tag = f"n{int(noise * 1000)}_g{int(gamma * 100)}_aug{i}"
                            fname = f"YMnO3_{label}_dz{dz:+.4f}_sc{suffix}_{tag}.png"
                            path = os.path.join(output_dir, label, fname)
                            Image.fromarray(image).save(path)

                # TDW
                if delta_z_tdw:
                    for dz in delta_z_tdw:
                        atoms_tdw = generate_type_A_tdw(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=dz)
                        image = render_stem_image(atoms_tdw)
                        image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
                        if augmentation_pipeline is not None:
                            image = augmentation_pipeline(image)
                        image = np.clip(image * 255, 0, 255).astype(np.uint8)
                        tag = f"n{int(noise * 1000)}_g{int(gamma * 100)}_aug{i}"
                        fname = f"YMnO3_TDW_dz{dz:+.4f}_sc{suffix}_{tag}.png"
                        path = os.path.join(output_dir, "typeA-TDW", fname)
                        Image.fromarray(image).save(path)

                # type C TDW + LDW
                if delta_z_typec:
                    for dz in delta_z_typec:
                        atoms_tdw = generate_type_c_dw(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=dz)
                        image = render_stem_image(atoms_tdw)
                        image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
                        if augmentation_pipeline is not None:
                            image = augmentation_pipeline(image)
                        image = np.clip(image * 255, 0, 255).astype(np.uint8)
                        tag = f"n{int(noise * 1000)}_g{int(gamma * 100)}_aug{i}"
                        fname = f"YMnO3_C_dz{dz:+.4f}_sc{suffix}_{tag}.png"
                        path = os.path.join(output_dir, "typeC-DW", fname)
                        Image.fromarray(image).save(path)

                
                #type D
                if delta_z_typed:
                    for dz in delta_z_typed:
                        atoms_tdw = generate_type_d_dw(structure_flat, y_ions["b"], y_ions["a"], delta_z_base=dz)
                        image = render_stem_image(atoms_tdw)
                        image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
                        if augmentation_pipeline is not None:
                            image = augmentation_pipeline(image)
                        image = np.clip(image * 255, 0, 255).astype(np.uint8)
                        tag = f"n{int(noise * 1000)}_g{int(gamma * 100)}_aug{i}"
                        fname = f"YMnO3_D_dz{dz:+.4f}_sc{suffix}_{tag}.png"
                        path = os.path.join(output_dir, "typeD-LDW", fname)
                        Image.fromarray(image).save(path)
                
                # UP and DN
                if delta_z_updn:
                    for dz in delta_z_updn:
                        for label, dz_sign in [("UP", -1), ("DN", +1)]:
                            atoms_projected = []

                            for site in structure:
                                if site.species_string == "Mn":
                                    _, y, z = site.frac_coords
                                    atoms_projected.append(("Mn", 0.0, y, z))

                            for _, fc in y_ions["b"]:
                                x_frac, y_frac, z_frac = np.array(fc).flatten()
                                z_new = (z_frac + dz_sign * dz) % 1.0
                                atoms_projected.append(("Y", x_frac, y_frac, z_new))
                                atoms_projected.append(("Y", 0.0, y_frac, z_new))

                            for _, fc in y_ions["a"]:
                                x_frac, y_frac, z_frac = np.array(fc).flatten()
                                atoms_projected.append(("Y", x_frac, y_frac, z_frac))
                                atoms_projected.append(("Y", 0.0, y_frac, z_frac))

                            image = render_stem_image(atoms_projected)
                            image = apply_image_augmentation(image, noise_type="gaussian", noise_param=noise, gamma=gamma)
                            if augmentation_pipeline is not None:
                                image = augmentation_pipeline(image)
                            image = np.clip(image * 255, 0, 255).astype(np.uint8)
                            tag = f"n{int(noise * 1000)}_g{int(gamma * 100)}_aug{i}"
                            fname = f"YMnO3_{label}_dz{dz:+.4f}_sc{suffix}_{tag}.png"
                            path = os.path.join(output_dir, label, fname)
                            Image.fromarray(image).save(path)



def generate_domain_wall_dataset_parallel(
        
    original_structure,
    output_dir,
    supercell_configs,
    delta_z_t2t,
    delta_z_tdw,    #type A TDW
    delta_z_typec,  #type C LDW + TDW
    delta_z_typed,  #type D
    delta_z_updn,
    augmentation_pipeline,
    num_aug=10,
    noise_levels=[0.03, 0.05, 0.07],
    gamma_values=[0.8, 1.0, 1.2],
):
    os.makedirs(output_dir, exist_ok=True)
    for label in ['TT', 'HH', 'typeA-TDW', 'typeC-DW', 'typeD-LDW','UP', 'DN']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    # Use all available cores
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for config in supercell_configs:
            seed = (int(time.time() * 1000) + os.getpid()) % (2**32)   #provoke randdom start for each TDW and LDW walls
            futures.append(executor.submit(
                generate_single_config,
                config,
                original_structure,
                output_dir,
                delta_z_t2t,
                delta_z_tdw,
                delta_z_typec,
                delta_z_typed,
                delta_z_updn,
                num_aug,
                noise_levels,
                gamma_values
            ))
        # Wait for completion
        for f in futures:
            f.result()
