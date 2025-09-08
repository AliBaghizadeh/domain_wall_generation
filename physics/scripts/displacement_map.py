import os
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from domain_wall_generation.physics.core.displacement import (
    batch_generate_displacement_maps,
    generate_displacement_map,
)
from domain_wall_generation.config_labels import POLAR_LABELS


#CLASSES = ['TT', 'HH', 'typeA-TDW', 'typeC-DW', 'typeD-LDW','UP', 'DN']
CLASSES = [cls for cls, idx in sorted(POLAR_LABELS.items(), key=lambda x: x[1])]
# Or just:
CLASSES = list(POLAR_LABELS.keys())

#input_root="outputs/stem_domain"
#output_root="outputs/stem_domains_disp"
input_root=r"C:\Ali\microscopy datasets\physics-informed-STEM-image-model\Training and generated images\generated_gan_dataset"
output_root=r"C:\Ali\microscopy datasets\physics-informed-STEM-image-model\Training and generated images\generated_gan_dataset_disp"

print(os.getcwd())

batch_generate_displacement_maps(
    input_root=input_root,
    output_root=output_root,
    classes=CLASSES,
    use_signed=True
)


