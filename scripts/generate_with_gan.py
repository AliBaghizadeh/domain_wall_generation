#!/usr/bin/env python3
"""
Generate images with a trained GAN using your eval.image_generation utilities.

Examples:
  PYTHONPATH=. python scripts/generate_with_gan.py \
    --checkpoint outputs/gan_runs/run1/models/generator.pth \
    --model attention \
    --classes '["typeC-DW"]' \
    --disp-root outputs/stem_domain_disp \
    --image-root outputs/stem_domain \
    --save-dir outputs/gan_infer/typeC \
    --num-variants 5 --z-scale 0.4

Notes:
- Uses the same dataset + label mapping logic as training to build a small loader.
- If you only want to sample from noise (no real images), pass a tiny subset via --limit.
"""
import argparse
import ast
import os
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional

try:
    import yaml  # optional
except Exception:
    yaml = None

# Project
from domain_wall_generation.config_labels import POLAR_LABELS
from domain_wall_generation.gan.ml_utils.dataset import (
    DisplacementToImageDataset,
    RemappedDatasetWithFilenames,
    filter_dataset_by_classnames,
)
from domain_wall_generation.gan.eval.image_generation import generate_gan_images
from domain_wall_generation.gan.models.base_generators.sharp_pix2pix import Pix2PixGenerator
from domain_wall_generation.gan.models.base_generators.wall_attention_generator import (
    Pix2PixGenerator_wall_attention,
)


def _maybe_list(s: Optional[str]) -> Optional[List[Any]]:
    if s is None:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        raise ValueError(f"Expected a Python list literal, got: {s}")

def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed but --config was provided.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _merge(cli: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in cli.items():
        if v is not None:
            out[k] = v
    return out

def _build_loader(disp_root: str, image_root: str, target_classes: List[str], batch_size: int, limit: Optional[int]):
    dataset_full = DisplacementToImageDataset(image_root=image_root, disp_root=disp_root, transform=None)
    label_indices = {k: v for k, v in POLAR_LABELS.items()}
    selected_indices = {label_indices[name] for name in target_classes}
    filtered_subset, _ = filter_dataset_by_classnames(dataset_full, target_classes, label_indices)
    dataset = RemappedDatasetWithFilenames(filtered_subset, dataset_full, selected_indices)

    if limit is not None:
        # take first N samples (simple, reproducible)
        indices = list(range(min(limit, len(dataset))))
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def main():
    p = argparse.ArgumentParser(description="Generate images with trained GAN.")
    p.add_argument("--config", type=str, help="Optional YAML config.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to generator .pth/.pt file.")
    p.add_argument("--model", type=str, choices=["sharp", "attention"], default="attention",
                   help="Generator architecture to instantiate for loading weights.")
    p.add_argument("--classes", type=str, default='["typeC-DW"]', help='List of classes to sample, e.g. \'["typeC-DW"]\'')
    p.add_argument("--disp-root", type=str, required=True, help="Displacement maps root (same as training).")
    p.add_argument("--image-root", type=str, required=True, help="Images root (same as training).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-dir", type=str, default="outputs/gan_infer")
    p.add_argument("--num-variants", type=int, default=5, help="How many z-samples per input.")
    p.add_argument("--z-scale", type=float, default=0.4)
    p.add_argument("--save-z", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on dataset size for quick sampling.")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    cli = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "target_classes": _maybe_list(args.classes),
        "disp_root": args.disp_root,
        "image_root": args.image_root,
        "batch_size": args.batch_size,
        "device": args.device,
        "save_dir": args.save_dir,
        "num_variants": args.num_variants,
        "z_scale": args.z_scale,
        "save_z": args.save_z,
        "limit": args.limit,
    }
    params = _merge(cli, cfg)

    target_classes = params["target_classes"] or ["typeC-DW"]
    n_classes = len(target_classes)

    # --- build generator & load weights ---
    device = torch.device(params["device"] if torch.cuda.is_available() and params["device"].startswith("cuda") else "cpu")
    if params["model"] == "attention":
        generator = Pix2PixGenerator_wall_attention(in_channels=1, out_channels=1, n_classes=n_classes).to(device)
        # match your training-time attributes
        generator.use_learned_attention = True
        generator.noise_scale = params["z_scale"]
    else:
        generator = Pix2PixGenerator(in_channels=1, out_channels=1, n_classes=n_classes).to(device)

    state = torch.load(params["checkpoint"], map_location=device)
    # accommodates state dict or full checkpoint w/ 'state_dict'
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    generator.load_state_dict(state_dict, strict=False)
    generator.eval()

    # --- dataloader to drive generation ---
    loader = _build_loader(
        disp_root=params["disp_root"],
        image_root=params["image_root"],
        target_classes=target_classes,
        batch_size=params["batch_size"],
        limit=params["limit"],
    )

    os.makedirs(params["save_dir"], exist_ok=True)
    # For generate_gan_images we need remapped label names and POLAR_LABELS
    remapped_label_names = target_classes  # one-to-one with the remapped indices
    generate_gan_images(
        generator=generator,
        gan_loader=loader,
        remapped_label_names=remapped_label_names,
        POLAR_LABELS=POLAR_LABELS,
        save_dir=params["save_dir"],
        device=device,
        use_z=True,
        num_variants=int(params["num_variants"]),
        z_scale=float(params["z_scale"]),
        save_z=bool(params["save_z"]),
    )
    print(f"✅ Generated images saved to: {params['save_dir']}")

if __name__ == "__main__":
    main()
