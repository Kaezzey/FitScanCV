from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_resnet18" / "best.pt"
DEFAULT_VAL_MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_val.csv"
DEFAULT_FULL_MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_manifest.csv"
DEFAULT_START_SPLIT = "val"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gui.bodym_gui import (
    BodyMExplorerConfig,
    BodyMExplorerError,
    launch_bodym_accuracy_explorer,
)
from scripts.inference.bodym_inference import BodyMInferenceError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the BodyM Tkinter accuracy explorer."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Path to the tuned BodyM checkpoint. "
            f"Defaults to {DEFAULT_CHECKPOINT_PATH}."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional inference config override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=DEFAULT_VAL_MANIFEST_PATH,
        help=(
            "Validation manifest used for the explorer val rows. "
            f"Defaults to {DEFAULT_VAL_MANIFEST_PATH}."
        ),
    )
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=DEFAULT_FULL_MANIFEST_PATH,
        help=(
            "Canonical manifest used for holdout rows. "
            f"Defaults to {DEFAULT_FULL_MANIFEST_PATH}."
        ),
    )
    parser.add_argument(
        "--start-split",
        type=str,
        choices=("val", "testA", "testB"),
        default=DEFAULT_START_SPLIT,
        help=f"Initial split filter. Defaults to {DEFAULT_START_SPLIT}.",
    )
    parser.add_argument(
        "--start-photo-id",
        type=str,
        default=None,
        help="Optional initial photo_id selection inside the chosen start split.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        launch_bodym_accuracy_explorer(
            BodyMExplorerConfig(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                device=args.device,
                val_manifest_path=args.val_manifest,
                full_manifest_path=args.full_manifest,
                start_split=args.start_split,
                start_photo_id=args.start_photo_id,
            )
        )
    except (FileNotFoundError, BodyMExplorerError, BodyMInferenceError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
