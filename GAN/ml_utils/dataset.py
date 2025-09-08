# Import Libraries
import os
import random
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from PIL import Image
from domain_wall_generation.config_labels import POLAR_LABELS

from collections import defaultdict

# Import Finished

class DisplacementToImageDataset(Dataset):
    """
    PyTorch Dataset for loading paired (displacement map, STEM image, label) data
    used for training conditional GANs (e.g., Pix2Pix) on YMnO3 ferroelectric domain structures.

    Each item in the dataset returns:
        - disp_map: Tensor of shape (1, H, W), the input displacement map.
        - real_img: Tensor of shape (1, H, W), the corresponding real STEM image.
        - label: Integer class label (0 for 'UP', 1 for 'DN').

    The class expects that both `disp_root` and `image_root` have the same folder structure:
        disp_root/
            UP/
                *.png
            DN/
                *.png
        image_root/
            UP/
                *.png
            DN/
                *.png

    Args:
        image_root (str): Path to directory containing real domain images.
        disp_root (str): Path to directory containing corresponding displacement maps.
        transform (callable, optional): Optional joint transform to apply to both
                                        image and displacement map (e.g., resizing, flipping, etc.).
    """
    def __init__(self, image_root, disp_root, transform=None):
        self.pairs = []
        self.transform = transform

        for class_name in os.listdir(disp_root):
            disp_class_path = os.path.join(disp_root, class_name)
            image_class_path = os.path.join(image_root, class_name)

            if not os.path.isdir(disp_class_path):
                continue

            #label = POLAR_LABELS.get(class_name.upper(), -1)
            label = POLAR_LABELS.get(class_name, -1)

            
            if label == -1:
                continue  # Skip unknown classes

            disp_filenames = sorted([
            f for f in os.listdir(disp_class_path) if f.endswith(".png")])
            image_filenames = sorted([f for f in os.listdir(image_class_path) if f.endswith(".png")        ])
            
            # Keep only matching filenames in both folders
            common_filenames = sorted(set(disp_filenames) & set(image_filenames))
            
            for fname in common_filenames:
                disp_path = os.path.join(disp_class_path, fname)
                image_path = os.path.join(image_class_path, fname)
                self.pairs.append((disp_path, image_path, label))
        
        random.shuffle(self.pairs)
        #for disp_path, image_path, label in self.pairs[:10]:
        #    print(f"{label}: {os.path.basename(disp_path)} ↔ {os.path.basename(image_path)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        disp_path, image_path, label = self.pairs[idx]
    
        #print(f"[DEBUG] Index {idx} → DISP: {disp_path} | IMG: {image_path}")
    
        disp = Image.open(disp_path).convert("L")
        img = Image.open(image_path).convert("L")

        if self.transform:
            disp, img = self.transform(disp, img)
        else:
            disp = torch.tensor(np.array(disp), dtype=torch.float32).unsqueeze(0) / 255.0
            img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0

        return disp, img, label
    
    def get_label(self, idx):
        _, _, label = self.pairs[idx]
        return label


def filter_dataset_by_classnames(dataset, classnames, label_indices):
    """
    Args:
        dataset: an instance of DisplacementToImageDataset
        classnames: list of class name strings, e.g. ['typeC-DW']
        label_indices: dict mapping class names to their integer labels
    Returns:
        Subset of dataset filtered by class
        set: selected label indices
    """
    target_label_indices = {label_indices[name] for name in classnames}
    indices = [
        i for i in range(len(dataset))
        if dataset.get_label(i) in target_label_indices
    ]
    return torch.utils.data.Subset(dataset, indices), indices

class DatasetWithFilenames(Dataset):
    """
    Wraps a Subset of DisplacementToImageDataset to include filenames.
    """
    def __init__(self, subset, original_dataset):
        self.subset = subset
        self.original_dataset = original_dataset
        self.indices = subset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        disp, img, label = self.original_dataset[original_idx]
        disp_path, img_path, _ = self.original_dataset.pairs[original_idx]
        filename = os.path.basename(img_path)
        return disp, img, label, filename

class RemappedDatasetWithFilenames(Dataset):
    """
    Wraps a filtered subset of DisplacementToImageDataset and remaps class labels.
    Ensures labels are mapped to [0, N-1] for N selected classes.
    """
    def __init__(self, subset, original_dataset, selected_label_indices):
        self.subset = subset
        self.original_dataset = original_dataset
        self.indices = subset.indices
        self.selected_label_indices = sorted(list(selected_label_indices))
        self.label_map = {old: new for new, old in enumerate(self.selected_label_indices)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        disp, img, label = self.original_dataset[original_idx]
        disp_path, img_path, _ = self.original_dataset.pairs[original_idx]
        filename = os.path.basename(img_path)

        # Map original label to new index
        label = torch.tensor(self.label_map[int(label)], dtype=torch.long)
        return disp, img, label, filename
