# Launcher Script

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# Project
from domain_wall_generation.physics.core.domain_walls import generate_domain_wall_dataset_parallel
from domain_wall_generation.gan.ml_utils.augmentation import (
    Apply_Augment,
    SaltPepperNoise,
    AtomicPlaneDistortionFixed,
    AtomicPlaneDistortionRandom,
    ScanDistortion,
    DriftDistortion,
)

# End of Imports

def make_aug_pipeline():
    return Apply_Augment([
        SaltPepperNoise(prob=0.02),
        AtomicPlaneDistortionFixed(frequency=10, intensity=0.05),
        AtomicPlaneDistortionRandom(probability=0.05, intensity=0.1),
        ScanDistortion(frequency=10, intensity=3),
        DriftDistortion(frequency=10, intensity=3),
    ])

# Load your paraelectric CIF file
cif_path = "data/ymno3_unpolar.cif"
original_structure = CifParser(cif_path).parse_structures(primitive=False)[0]

# Configuration parameters
supercell_configs_old = [
    [1, 8, 3], [1, 8, 4],
    [1, 7, 3], [1, 7, 4],
    [1, 6, 3], [1, 6, 4]
]

supercell_configs = [
    [1, 14, 4], [1, 10, 4],
    [1, 9, 4], [1, 8, 3], 
    [1, 7, 4], [1, 6, 4]
]

#Strength of polarization for each type of domain
delta_z_t2t =  None # [0.350, 0.50, 0.65]
delta_z_tdw =  None #[ -0.013,  0.008, 0.013]  # [-0.008, -0.011, 0.008, 0.011]     #[-0.016, -0.12, 0.12, 0.16]
#delta_z_ldw = [-0.008, -0.013,  0.008,0.013]       #[-0.008, -0.011, 0.008, 0.011]
delta_z_typec = None # [-0.009, -0.014,  0.014] 
delta_z_typed = None # [-0.008, -0.013,  0.008,0.013] 
delta_z_updn = [0.014, 0.018, 0.022]  #[-0.008, -0.013,  0.008,0.013]

#Parameters for augmentation
num_aug = 3
noise_levels = [0.01, 0.05, 0.07]   #0.04, 0.07, 0.1]
gamma_values = [0.5, 0.8, 1.0]      #[0.8, 1.0, 1.2]
output_dir="outputs/stem_domain/"

if __name__ == "__main__":
    aug_pipeline = make_aug_pipeline()
    generate_domain_wall_dataset_parallel(
        original_structure=original_structure,
        output_dir=output_dir,   
        supercell_configs=supercell_configs,
        delta_z_t2t=delta_z_t2t,
        delta_z_tdw=delta_z_tdw,
        delta_z_typec = delta_z_typec,
        delta_z_typed = delta_z_typed,
        delta_z_updn=delta_z_updn,
        num_aug=num_aug,
        noise_levels=noise_levels,
        gamma_values=gamma_values,
        augmentation_pipeline=aug_pipeline
    )
