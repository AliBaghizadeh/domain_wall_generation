# Import Libraries
import torch
import torch.nn as nn

from domain_wall_generation.gan.models.base_generators.sharp_pix2pix import Pix2PixGenerator
from domain_wall_generation.gan.models.base_generators.wall_attention_generator import Pix2PixGenerator_wall_attention
# Import Finished

class UnifiedWallGenerator(nn.Module):
    def __init__(self, sharp_classes=[0,1,2,3,4,5], complex_classes=[6]):
        super().__init__()
        self.sharp_model = Pix2PixGenerator(n_classes=len(sharp_classes))
        self.complex_model = Pix2PixGenerator_wall_attention(n_classes=len(complex_classes))
        self.sharp_class_ids = set(sharp_classes)
        self.complex_class_ids = set(complex_classes)

    def forward(self, x, labels, z=None, return_attn=False):
        if all(lbl.item() in self.sharp_class_ids for lbl in labels):
            return self.sharp_model(x, labels)
        elif all(lbl.item() in self.complex_class_ids for lbl in labels):
            return self.complex_model(x, labels, z=z, return_attn=return_attn)
        else:
            raise ValueError("Mixed-class batches not supported. Use separate batches.")
