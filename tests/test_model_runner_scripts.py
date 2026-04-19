from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_bodym_models import (
    DEFAULT_BASELINE_CHECKPOINT_PATH as COMPARE_BASELINE_CHECKPOINT,
)
from scripts.compare_bodym_models import (
    DEFAULT_CANDIDATE_CHECKPOINT_PATH as COMPARE_CANDIDATE_CHECKPOINT,
)
from scripts.compare_bodym_models import DEFAULT_MANIFEST_PATH as COMPARE_MANIFEST_PATH
from scripts.compare_bodym_models import DEFAULT_OUTPUT_DIR as COMPARE_OUTPUT_DIR
from scripts.compare_bodym_models import DEFAULT_SPLIT as COMPARE_SPLIT
from scripts.compare_bodym_models import parse_args as parse_compare_args
from scripts.train_bodym_baseline import DEFAULT_CONFIG_PATH as BASELINE_CONFIG_PATH
from scripts.train_bodym_baseline import parse_args as parse_baseline_args
from scripts.train_bodym_resnet18 import DEFAULT_CONFIG_PATH as RESNET18_CONFIG_PATH
from scripts.train_bodym_resnet18 import parse_args as parse_resnet18_args


def test_train_bodym_baseline_defaults_to_baseline_config() -> None:
    args = parse_baseline_args([])

    assert args.config == BASELINE_CONFIG_PATH
    assert args.run_dir is None


def test_train_bodym_resnet18_defaults_to_stronger_config() -> None:
    args = parse_resnet18_args([])

    assert args.config == RESNET18_CONFIG_PATH
    assert args.run_dir is None


def test_compare_bodym_models_defaults_to_validation_comparison() -> None:
    args = parse_compare_args([])

    assert args.baseline_checkpoint == COMPARE_BASELINE_CHECKPOINT
    assert args.candidate_checkpoint == COMPARE_CANDIDATE_CHECKPOINT
    assert args.manifest_path == COMPARE_MANIFEST_PATH
    assert args.split == COMPARE_SPLIT
    assert args.output_dir == COMPARE_OUTPUT_DIR
