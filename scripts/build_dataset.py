#!/usr/bin/env python3
"""
Build a physics-based STEM image dataset and (optionally) displacement maps.

Examples:
  PYTHONPATH=. python scripts/build_dataset.py \
    --cif data/ymno3_unpolar.cif \
    --out outputs/stem_domain \
    --supercells "[[1,14,4],[1,10,4],[1,9,4],[1,8,3],[1,7,4],[1,6,4]]" \
    --dz-updn "[0.014,0.018,0.022]" \
    --num-aug 3 --noise-levels "[0.01,0.05,0.07]" --gamma-values "[0.5,0.8,1.0]" \
    --make-disp

You can also pass a YAML with the same keys via --config.
"""
import argparse
import ast
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import yaml  # optional
except Exception:
    yaml = None

from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Your existing modules (keep as-is)
from domain_wall_generation.physics.core.domain_walls import generate_domain_wall_dataset_parallel
from domain_wall_generation.physics.core.displacement import batch_generate_displacement_maps
from domain_wall_generation.config_labels import POLAR_LABELS

# --- helpers ---
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
        raise RuntimeError("PyYAML is not installed but --config was provided.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _merge(cli: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in cli.items():
        if v is not None:
            out[k] = v
    return out

def main():
    p = argparse.ArgumentParser(description="Build physics-based dataset (+displacement maps).")
    p.add_argument("--config", type=str, help="YAML config with the same keys as CLI flags.")
    p.add_argument("--cif", type=str, required=False, help="Path to CIF file.")
    p.add_argument("--out", type=str, default="outputs/stem_domain", help="Output directory for images.")
    p.add_argument("--supercells", type=str, help="e.g. '[[1,14,4],[1,10,4]]'")
    p.add_argument("--dz-t2t", type=str, help="e.g. '[0.35,0.50,0.65]' or 'null'")
    p.add_argument("--dz-tdw", type=str, help="e.g. '[-0.013,0.013]' or 'null'")
    p.add_argument("--dz-typec", type=str, help="e.g. '[-0.009,0.014]' or 'null'")
    p.add_argument("--dz-typed", type=str, help="e.g. '[-0.008,0.013]' or 'null'")
    p.add_argument("--dz-updn", type=str, help="e.g. '[0.014,0.018,0.022]'")
    p.add_argument("--num-aug", type=int, default=3)
    p.add_argument("--noise-levels", type=str, default="[0.01,0.05,0.07]")
    p.add_argument("--gamma-values", type=str, default="[0.5,0.8,1.0]")
    p.add_argument("--make-disp", action="store_true", help="Also compute displacement maps.")
    p.add_argument("--disp-out", type=str, default=None, help="Output dir for displacement maps (default: out+'_disp').")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    cli = {
        "cif": args.cif,
        "out": args.out,
        "supercells": _maybe_list(args.supercells),
        "dz_t2t": _maybe_list(args.dz_t2t),
        "dz_tdw": _maybe_list(args.dz_tdw),
        "dz_typec": _maybe_list(args.dz_typec),
        "dz_typed": _maybe_list(args.dz_typed),
        "dz_updn": _maybe_list(args.dz_updn),
        "num_aug": args.num_aug,
        "noise_levels": _maybe_list(args.noise_levels),
        "gamma_values": _maybe_list(args.gamma_values),
        "make_disp": args.make_disp,
        "disp_out": args.disp_out,
    }
    params = _merge(cli, cfg)

    # --- load structure ---
    if not params.get("cif"):
        p.error("--cif is required (or provide it in --config)")
    structure = CifParser(params["cif"]).parse_structures(primitive=False)[0]
    structure = SpacegroupAnalyzer(structure).get_refined_structure()

    # --- defaults (matching your examples) ---
    supercells = params["supercells"] or [[1,14,4],[1,10,4],[1,9,4],[1,8,3],[1,7,4],[1,6,4]]
    dz_t2t    = params.get("dz_t2t", None)      # e.g., None or [0.35, 0.50, 0.65]
    dz_tdw    = params.get("dz_tdw", None)      # e.g., None or [-0.013, 0.013]
    dz_typec  = params.get("dz_typec", None)    # e.g., None or [-0.009, 0.014]
    dz_typed  = params.get("dz_typed", None)    # e.g., None
    dz_updn   = params.get("dz_updn", [0.014, 0.018, 0.022])

    out_dir = params["out"]
    os.makedirs(out_dir, exist_ok=True)

    print("🧪 Generating physics-based images...")
    generate_domain_wall_dataset_parallel(
        original_structure=structure,
        output_dir=out_dir,
        supercell_configs=supercells,
        delta_z_t2t=dz_t2t,
        delta_z_tdw=dz_tdw,
        delta_z_typec=dz_typec,
        delta_z_typed=dz_typed,
        delta_z_updn=dz_updn,
        num_aug=int(params["num_aug"]),
        noise_levels=params["noise_levels"],
        gamma_values=params["gamma_values"],
        augmentation_pipeline=None,  # keep simple; your file accepts this
    )
    print(f"✅ Images saved to: {out_dir}")

    if params["make_disp"]:
        disp_out = params["disp_out"] or (out_dir.rstrip("/\\") + "_disp")
        classes = list(POLAR_LABELS.keys())
        print("📐 Generating displacement maps...")
        batch_generate_displacement_maps(
            input_root=out_dir,
            output_root=disp_out,
            classes=classes,
            use_signed=True
        )
        print(f"✅ Displacement maps saved to: {disp_out}")

if __name__ == "__main__":
    main()
