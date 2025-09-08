from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# Project
from domain_wall_generation.physics.core.domain_walls import generate_domain_wall_dataset

# Load your paraelectric CIF file
structure = CifParser("data/ymno3_unpolar.cif").parse_structures(primitive=False)[0]
standardized_structure = SpacegroupAnalyzer(structure).get_refined_structure()

# Define parameters
supercell_configs = [[1, 8, 3], [1, 8, 4],
[1, 7, 3], [1, 7, 4],
[1, 6, 3], [1, 6, 4]]
delta_z_t2t = [0.2950, 0.3950, 0.4950]
delta_z_h2h = [-dz for dz in delta_z_t2t]
delta_z_tdw = [-0.2, -0.15, -0.1, 0.1, 0.15, 0.2]
delta_z_updn = [0.01, 0.015, 0.02]

# Run the dataset generator
generate_domain_wall_dataset(
    original_structure=standardized_structure,
    output_dir="/outputs/stem_domains",    #output_dir="/kaggle/working/stem_domains",
    supercell_configs=[[1, 8, 3], [1, 8, 4]],
    delta_z_t2t=[0.4950],   #[0.2950, 0.3950, 0.4950],
    delta_z_tdw=[-0.2, 0.2],  #[-0.2, -0.15, -0.1, 0.1, 0.15, 0.2],
    delta_z_updn=[0.01],    #delta_z_updn=[0.01, 0.015, 0.02],
    num_aug=10,
    noise_levels=[0.03, 0.07],    #[0.03, 0.05, 0.07],
    gamma_values=[0.8, 1.0, 1.2]
)

from displacement import batch_generate_displacement_maps

CLASSES = ["HH", "TT", "TDW", "UP", "DN"]

batch_generate_displacement_maps(
    input_root="outputs/stem_domains",
    output_root="outputs/stem_domains_disp",
    classes=CLASSES,
    use_signed=True
)