from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "bodym_resnet18.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.models.bodym_training import TrainingPipelineError, load_experiment_config, train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the stronger ResNet18 BodyM model with the default stronger config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to the stronger-model YAML config. "
            f"Defaults to {DEFAULT_CONFIG_PATH}."
        ),
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
        result = train_model(config=config, config_path=args.config)
    except (FileNotFoundError, TrainingPipelineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Stronger-model training completed")
    print(f"Run directory: {result['run_dir']}")
    print(f"Best checkpoint: {result['best_checkpoint_path']}")
    print(f"Last checkpoint: {result['last_checkpoint_path']}")
    print(f"Best val mean MAE: {result['best_metrics']['val_mean_mae']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
