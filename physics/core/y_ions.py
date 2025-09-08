# Import Libraries
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Import Finished

def get_y_ions(supercell):
    """
    function is meant to produce all neutral Y ions, then it should flatten all 4b Y ions to the same z-coordinate as the 2a ions.
    
    Parameters:
        supercell: pymatgen Structure
        delta_z_base: how much to subtract from 4b Y ion z to flatten to 2a

    Returns:
        new_structure: Structure with flattened Y z coords
        y_ions: dict with keys 'b' and 'a' of (index, frac_coords)
    """
    sga = SpacegroupAnalyzer(supercell)
    dataset = sga.get_symmetry_dataset()
    wyckoff_positions = dataset.wyckoffs

    new_structure = supercell.copy()
    y_ions = {"b": [], "a": []}

    for i, site in enumerate(supercell.sites):
        if site.species_string == "Y":
            new_fc = list(site.frac_coords)
            if wyckoff_positions[i] == "b":
                
                y_ions["b"].append((i, np.array(new_fc)))
            elif wyckoff_positions[i] == "a":
                y_ions["a"].append((i, site.frac_coords))

    return new_structure, y_ions