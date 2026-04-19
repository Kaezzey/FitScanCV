from __future__ import annotations

from contextlib import contextmanager
import csv
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

import scripts.bodym_dataset as bodym_dataset
from scripts.models.benchmark_bodym import (
    DEFAULT_BASELINE_CONFIG_PATH,
    DEFAULT_CANDIDATE_CONFIG_PATH,
    parse_args,
)
from scripts.models.bodym_benchmarking import benchmark_checkpoints
from scripts.models.bodym_training import load_experiment_config, train_model

SCRATCH_ROOT = REPO_ROOT / "data" / "interim" / "test_scratch"


@contextmanager
def scratch_dir() -> Path:
    SCRATCH_ROOT.mkdir(exist_ok=True)
    directory = SCRATCH_ROOT / f"case_{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def fake_image_tensor(_: Path) -> torch.Tensor:
    return torch.ones((1, 960, 720), dtype=torch.float32)


def write_manifest(path: Path, split_name: str, rows: int, start_index: int = 0) -> Path:
    manifest_rows: list[dict[str, str]] = []

    for index in range(start_index, start_index + rows):
        photo_id = f"{split_name}-photo-{index}"
        mask_path = path.parent / f"{photo_id}-mask.png"
        mask_left_path = path.parent / f"{photo_id}-mask-left.png"
        mask_path.write_bytes(b"")
        mask_left_path.write_bytes(b"")

        manifest_rows.append(
            {
                "split": split_name,
                "subject_id": f"subject-{index}",
                "subject_key": f"{split_name}::subject-{index}",
                "photo_id": photo_id,
                "mask_path": str(mask_path),
                "mask_left_path": str(mask_left_path),
                "hwg_gender": "female" if index % 2 == 0 else "male",
                "hwg_height_cm": str(160.0 + index),
                "hwg_weight_kg": str(60.0 + index),
                "measurement_ankle": str(21.0 + (index % 3)),
                "measurement_arm_length": str(45.0 + (index % 3)),
                "measurement_bicep": str(30.0 + (index % 3)),
                "measurement_calf": str(35.0 + (index % 3)),
                "measurement_chest": str(95.0 + (index % 3)),
                "measurement_forearm": str(25.0 + (index % 3)),
                "measurement_height": str(161.0 + (index % 3)),
                "measurement_hip": str(96.0 + (index % 3)),
                "measurement_leg_length": str(72.0 + (index % 3)),
                "measurement_shoulder_breadth": str(36.0 + (index % 3)),
                "measurement_shoulder_to_crotch": str(61.0 + (index % 3)),
                "measurement_thigh": str(52.0 + (index % 3)),
                "measurement_waist": str(83.0 + (index % 3)),
                "measurement_wrist": str(15.0 + (index % 3)),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(path, index=False)
    return path


def write_holdout_manifest(path: Path) -> Path:
    test_a = pd.read_csv(write_manifest(path.parent / "testA_only.csv", "testA", rows=2, start_index=200))
    test_b = pd.read_csv(write_manifest(path.parent / "testB_only.csv", "testB", rows=2, start_index=300))
    pd.concat([test_a, test_b], ignore_index=True).to_csv(path, index=False)
    return path


def write_config(
    path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    run_dir: Path,
    *,
    backbone_name: str,
    pretrained: bool,
    resize: list[int] | None,
) -> Path:
    config = {
        "data": {
            "train_manifest_path": str(train_manifest_path),
            "val_manifest_path": str(val_manifest_path),
            "train_split": "train",
            "val_split": "val",
            "view_mode": "paired",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "resize": resize,
        },
        "model": {
            "variant": "dual_view_late_fusion",
            "backbone_name": backbone_name,
            "pretrained": pretrained,
            "num_targets": 14,
            "single_view_name": "mask",
            "image_embedding_dim": 64,
            "metadata_embedding_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "seed": 42,
            "device": "cpu",
            "optimizer_name": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "loss_name": "smooth_l1",
            "log_every_n_steps": 0,
        },
        "output": {
            "run_dir": str(run_dir),
            "save_predictions_on_eval": False,
        },
    }

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def train_checkpoint(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_manifest: Path,
    val_manifest: Path,
    config_name: str,
    run_name: str,
    backbone_name: str,
    pretrained: bool,
    resize: list[int] | None,
) -> tuple[Path, Path]:
    config_path = write_config(
        tmp_path / config_name,
        train_manifest_path=train_manifest,
        val_manifest_path=val_manifest,
        run_dir=tmp_path / run_name,
        backbone_name=backbone_name,
        pretrained=pretrained,
        resize=resize,
    )
    monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)
    config = load_experiment_config(config_path)
    result = train_model(config, config_path=config_path)
    return config_path, Path(result["best_checkpoint_path"])


def test_benchmark_cli_defaults_are_wired() -> None:
    args = parse_args([])

    assert args.baseline_config == DEFAULT_BASELINE_CONFIG_PATH
    assert args.candidate_config == DEFAULT_CANDIDATE_CONFIG_PATH
    assert args.split is None


def test_benchmark_checkpoints_writes_report_and_delta_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torchvision")

    with scratch_dir() as tmp_path:
        train_manifest = write_manifest(tmp_path / "train.csv", "train", rows=4)
        val_manifest = write_manifest(tmp_path / "val.csv", "val", rows=2, start_index=100)
        baseline_config_path, baseline_checkpoint_path = train_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            config_name="baseline.yaml",
            run_name="baseline_run",
            backbone_name="light_cnn",
            pretrained=False,
            resize=None,
        )
        candidate_config_path, candidate_checkpoint_path = train_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            config_name="candidate.yaml",
            run_name="candidate_run",
            backbone_name="resnet18",
            pretrained=False,
            resize=[64, 64],
        )

        result = benchmark_checkpoints(
            baseline_config_path=baseline_config_path,
            baseline_checkpoint_path=baseline_checkpoint_path,
            candidate_config_path=candidate_config_path,
            candidate_checkpoint_path=candidate_checkpoint_path,
            manifest_path=val_manifest,
            split="val",
            output_dir=tmp_path / "benchmark_val",
        )

        report_path = Path(result["report_output_path"])
        table_path = Path(result["table_output_path"])
        assert report_path.is_file()
        assert table_path.is_file()
        assert "mean_mae" in result["baseline"]
        assert "mean_mae" in result["candidate"]
        assert len(result["deltas"]["per_target_mae"]) == 14
        assert len(result["deltas"]["per_target_rmse"]) == 14

        with report_path.open("r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
        assert report_payload["split"] == "val"
        assert report_payload["comparison_basis"] == "candidate_minus_baseline"

        with table_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 14
        assert "delta_mae_candidate_minus_baseline" in rows[0]


@pytest.mark.parametrize("split_name", ["testA", "testB"])
def test_benchmark_checkpoints_support_holdout_manifest_splits(
    monkeypatch: pytest.MonkeyPatch,
    split_name: str,
) -> None:
    pytest.importorskip("torchvision")

    with scratch_dir() as tmp_path:
        train_manifest = write_manifest(tmp_path / "train.csv", "train", rows=4)
        val_manifest = write_manifest(tmp_path / "val.csv", "val", rows=2, start_index=100)
        holdout_manifest = write_holdout_manifest(tmp_path / "holdout.csv")

        baseline_config_path, baseline_checkpoint_path = train_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            config_name="baseline.yaml",
            run_name="baseline_run",
            backbone_name="light_cnn",
            pretrained=False,
            resize=None,
        )
        candidate_config_path, candidate_checkpoint_path = train_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            config_name="candidate.yaml",
            run_name="candidate_run",
            backbone_name="resnet18",
            pretrained=False,
            resize=[64, 64],
        )

        result = benchmark_checkpoints(
            baseline_config_path=baseline_config_path,
            baseline_checkpoint_path=baseline_checkpoint_path,
            candidate_config_path=candidate_config_path,
            candidate_checkpoint_path=candidate_checkpoint_path,
            manifest_path=holdout_manifest,
            split=split_name,
            output_dir=tmp_path / f"benchmark_{split_name}",
        )

        assert result["split"] == split_name
        assert Path(result["report_output_path"]).is_file()
        assert Path(result["table_output_path"]).is_file()
