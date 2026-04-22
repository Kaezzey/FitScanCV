from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_resnet18" / "best.pt"
DEFAULT_SMOKE_MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_val.csv"
DEFAULT_SMOKE_SPLIT = "val"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.bodym_inference import (
    BodyMInferenceError,
    load_inference_service,
    resolve_repo_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local BodyM measurement prediction from mask image paths."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Path to the trained checkpoint. "
            f"Defaults to {DEFAULT_CHECKPOINT_PATH}."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config override for inference preprocessing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--front-image",
        type=Path,
        default=None,
        help="Path to the front/bodym mask image. If omitted, smoke mode uses the first manifest row.",
    )
    parser.add_argument(
        "--side-image",
        type=Path,
        default=None,
        help="Path to the side/bodym mask_left image. Required for dual-view checkpoints.",
    )
    parser.add_argument(
        "--hwg-gender",
        type=str,
        default=None,
        help="HWG gender value: female or male.",
    )
    parser.add_argument(
        "--hwg-height-cm",
        type=float,
        default=None,
        help="Height in centimeters.",
    )
    parser.add_argument(
        "--hwg-weight-kg",
        type=float,
        default=None,
        help="Weight in kilograms.",
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=DEFAULT_SMOKE_MANIFEST_PATH,
        help=(
            "Manifest used when no explicit image/HWG inputs are provided. "
            f"Defaults to {DEFAULT_SMOKE_MANIFEST_PATH}."
        ),
    )
    parser.add_argument(
        "--smoke-split",
        type=str,
        default=DEFAULT_SMOKE_SPLIT,
        help=f"Split used for smoke mode. Defaults to {DEFAULT_SMOKE_SPLIT}.",
    )
    return parser.parse_args(argv)


def _load_smoke_inputs(manifest_path: str | Path, split: str) -> dict[str, Any]:
    resolved_manifest_path = resolve_repo_path(manifest_path)
    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(f"Smoke manifest does not exist: {resolved_manifest_path}")

    frame = pd.read_csv(
        resolved_manifest_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    split_frame = frame.loc[frame["split"] == split].copy()
    if split_frame.empty:
        raise ValueError(f"No rows found for smoke split: {split!r}")

    row = split_frame.iloc[0]
    return {
        "front_image_path": row["mask_path"],
        "side_image_path": row["mask_left_path"],
        "hwg_gender": row["hwg_gender"],
        "hwg_height_cm": float(row["hwg_height_cm"]),
        "hwg_weight_kg": float(row["hwg_weight_kg"]),
        "smoke_manifest_path": str(resolved_manifest_path),
        "smoke_split": split,
        "photo_id": row.get("photo_id"),
    }


def _resolve_prediction_inputs(args: argparse.Namespace) -> dict[str, Any]:
    explicit_values = [
        args.front_image,
        args.side_image,
        args.hwg_gender,
        args.hwg_height_cm,
        args.hwg_weight_kg,
    ]
    has_any_explicit = any(value is not None for value in explicit_values)
    if not has_any_explicit:
        return _load_smoke_inputs(args.smoke_manifest, args.smoke_split)

    missing = []
    if args.front_image is None:
        missing.append("--front-image")
    if args.hwg_gender is None:
        missing.append("--hwg-gender")
    if args.hwg_height_cm is None:
        missing.append("--hwg-height-cm")
    if args.hwg_weight_kg is None:
        missing.append("--hwg-weight-kg")
    if missing:
        raise ValueError(
            "Explicit prediction mode is missing required arguments: "
            + ", ".join(missing)
        )

    return {
        "front_image_path": args.front_image,
        "side_image_path": args.side_image,
        "hwg_gender": args.hwg_gender,
        "hwg_height_cm": args.hwg_height_cm,
        "hwg_weight_kg": args.hwg_weight_kg,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        service = load_inference_service(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
        )
        inputs = _resolve_prediction_inputs(args)
        result = service.predict_from_paths(
            front_image_path=inputs["front_image_path"],
            side_image_path=inputs.get("side_image_path"),
            hwg_gender=inputs["hwg_gender"],
            hwg_height_cm=inputs["hwg_height_cm"],
            hwg_weight_kg=inputs["hwg_weight_kg"],
        )
        if "smoke_manifest_path" in inputs:
            result["inputs"]["smoke_manifest_path"] = inputs["smoke_manifest_path"]
            result["inputs"]["smoke_split"] = inputs["smoke_split"]
            result["inputs"]["photo_id"] = inputs["photo_id"]
    except (FileNotFoundError, BodyMInferenceError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
