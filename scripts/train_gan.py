#!/usr/bin/env python3
"""
Train your Pix2Pix / Pix2Pix + WallAttention GAN using your existing train_router.main().

Examples:
  PYTHONPATH=. python scripts/train_gan.py \
    --disp-root outputs/stem_domain_disp \
    --image-root outputs/stem_domain \
    --classes '["typeC-DW"]' \
    --save-dir outputs/gan_runs/run1 \
    --epochs 40 --batch-size 16 \
    --lambda-l1 30 --lambda-edge 15 --lambda-ssim 10 --lambda-attn 3 --lambda-div 5 --lambda-curv 10

Or use a YAML file:
  PYTHONPATH=. python scripts/train_gan.py --config gan/configs/train_defaults.yaml
"""
import argparse
import ast
from typing import Any, Dict, List, Optional

try:
    import yaml  # optional
except Exception:
    yaml = None

# Project
from domain_wall_generation.gan.train.train_router import main as train_main


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

def main():
    p = argparse.ArgumentParser(description="Train GAN (Pix2Pix / WallAttention).")
    p.add_argument("--config", type=str, help="YAML config path.")
    p.add_argument("--disp-root", type=str, help="Path to displacement maps root.")
    p.add_argument("--image-root", type=str, help="Path to images root.")
    p.add_argument("--classes", type=str, help='Target classes list, e.g. \'["typeC-DW"]\'')
    p.add_argument("--save-dir", type=str, default="./output")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lambda-l1", type=float, default=30.0)
    p.add_argument("--lambda-edge", type=float, default=15.0)
    p.add_argument("--lambda-ssim", type=float, default=10.0)
    p.add_argument("--lambda-attn", type=float, default=3.0)
    p.add_argument("--lambda-div", type=float, default=5.0)
    p.add_argument("--lambda-curv", type=float, default=10.0)
    p.add_argument("--noise-scale", type=float, default=0.4)
    args = p.parse_args()

    cfg = _load_yaml(args.config)

    cli = {
        "disp_root": args.disp_root,
        "image_root": args.image_root,
        "target_classes": _maybe_list(args.classes),
        "save_dir": args.save_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda_l1": args.lambda_l1,
        "lambda_edge": args.lambda_edge,
        "lambda_ssim": args.lambda_ssim,
        "lambda_attn": args.lambda_attn,
        "lambda_div": args.lambda_div,
        "lambda_curv": args.lambda_curv,
        "noise_scale": args.noise_scale,
    }

    params = _merge(cli, cfg)

    # Reasonable default (matches your train_router default)
    if not params.get("target_classes"):
        params["target_classes"] = ["typeC-DW"]

    # hand off to your existing training entrypoint
    train_main(**params)

if __name__ == "__main__":
    main()
