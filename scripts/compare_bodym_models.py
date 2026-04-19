from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "bodym_baseline.yaml"
DEFAULT_BASELINE_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_baseline" / "best.pt"
DEFAULT_CANDIDATE_CONFIG_PATH = REPO_ROOT / "configs" / "bodym_resnet18.yaml"
DEFAULT_CANDIDATE_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_resnet18" / "best.pt"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_val.csv"
DEFAULT_SPLIT = "val"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "model_comparison" / DEFAULT_SPLIT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.models.bodym_benchmarking import benchmark_checkpoints
from scripts.models.bodym_training import TrainingPipelineError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the baseline and stronger BodyM models on one split."
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
        help=(
            "Path to the baseline YAML config. "
            f"Defaults to {DEFAULT_BASELINE_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_CHECKPOINT_PATH,
        help=(
            "Path to the baseline checkpoint. "
            f"Defaults to {DEFAULT_BASELINE_CHECKPOINT_PATH}."
        ),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=DEFAULT_CANDIDATE_CONFIG_PATH,
        help=(
            "Path to the stronger-model YAML config. "
            f"Defaults to {DEFAULT_CANDIDATE_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=DEFAULT_CANDIDATE_CHECKPOINT_PATH,
        help=(
            "Path to the stronger-model checkpoint. "
            f"Defaults to {DEFAULT_CANDIDATE_CHECKPOINT_PATH}."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Manifest to compare on. "
            f"Defaults to {DEFAULT_MANIFEST_PATH}."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help=f"Split to compare on. Defaults to {DEFAULT_SPLIT}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for the comparison report and delta table. "
            f"Defaults to {DEFAULT_OUTPUT_DIR}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = benchmark_checkpoints(
            baseline_config_path=args.baseline_config,
            baseline_checkpoint_path=args.baseline_checkpoint,
            candidate_config_path=args.candidate_config,
            candidate_checkpoint_path=args.candidate_checkpoint,
            manifest_path=args.manifest_path,
            split=args.split,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, TrainingPipelineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Model comparison completed")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Split: {result['split']}")
    print(f"Baseline mean MAE: {result['baseline']['mean_mae']:.6f}")
    print(f"Candidate mean MAE: {result['candidate']['mean_mae']:.6f}")
    print(f"Delta mean MAE (candidate-baseline): {result['deltas']['mean_mae']:.6f}")
    print(f"Winner by mean MAE: {result['winner_by_mean_mae']}")
    print(f"Report JSON: {result['report_output_path']}")
    print(f"Per-target delta CSV: {result['table_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
