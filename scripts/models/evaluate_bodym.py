from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "bodym_baseline.yaml"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_baseline" / "best.pt"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.models.bodym_training import (
    TrainingPipelineError,
    evaluate_checkpoint,
    load_experiment_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BodyM checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the YAML experiment config. "
            f"Defaults to {DEFAULT_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Path to a saved checkpoint file. "
            f"Defaults to {DEFAULT_CHECKPOINT_PATH}."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional override for the evaluation manifest path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional override for the evaluation split.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional override for the configured output.run_dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_experiment_config(
            config_path=args.config,
            run_dir_override=args.run_dir,
        )
        result = evaluate_checkpoint(
            config=config,
            checkpoint_path=args.checkpoint,
            manifest_path_override=args.manifest_path,
            split_override=args.split,
        )
    except (FileNotFoundError, TrainingPipelineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Evaluation completed")
    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Split: {result['split']}")
    print(f"Loss: {result['loss']:.6f}")
    print(f"Mean MAE: {result['mean_mae']:.6f}")
    print(f"Mean RMSE: {result['mean_rmse']:.6f}")
    if result["prediction_output_path"] is not None:
        print(f"Predictions CSV: {result['prediction_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
