from __future__ import annotations

from contextlib import contextmanager
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
from scripts.models.evaluate_bodym import DEFAULT_CONFIG_PATH as EVAL_DEFAULT_CONFIG_PATH
from scripts.models.evaluate_bodym import parse_args as parse_evaluate_args
from scripts.models.train_bodym import DEFAULT_CONFIG_PATH as TRAIN_DEFAULT_CONFIG_PATH
from scripts.models.train_bodym import parse_args as parse_train_args
from scripts.models.bodym_training import evaluate_checkpoint, load_experiment_config, train_model

REAL_TRAIN_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_train.csv"
REAL_VAL_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_val.csv"
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


def write_config(
    path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    run_dir: Path,
    *,
    model_variant: str,
    view_mode: str,
    single_view_name: str = "mask",
    backbone_name: str = "light_cnn",
    pretrained: bool = False,
    epochs: int = 1,
    resize: list[int] | None = None,
    resize_mode: str = "nearest",
    normalize_mean: float | None = None,
    normalize_std: float | None = None,
    save_predictions_on_eval: bool = False,
) -> Path:
    config = {
        "data": {
            "train_manifest_path": str(train_manifest_path),
            "val_manifest_path": str(val_manifest_path),
            "train_split": "train",
            "val_split": "val",
            "view_mode": view_mode,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "resize": resize,
            "resize_mode": resize_mode,
            "normalize_mean": normalize_mean,
            "normalize_std": normalize_std,
        },
        "model": {
            "variant": model_variant,
            "num_targets": 14,
            "single_view_name": single_view_name,
            "backbone_name": backbone_name,
            "pretrained": pretrained,
            "image_embedding_dim": 64,
            "metadata_embedding_dim": 16,
            "hidden_dim": 64,
            "dropout": 0.0,
        },
        "training": {
            "epochs": epochs,
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
            "save_predictions_on_eval": save_predictions_on_eval,
        },
    }

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def test_load_experiment_config_maps_yaml_sections() -> None:
    with scratch_dir() as tmp_path:
        train_manifest = write_manifest(tmp_path / "train.csv", "train", rows=4)
        val_manifest = write_manifest(tmp_path / "val.csv", "val", rows=2, start_index=100)
        config_path = write_config(
            tmp_path / "config.yaml",
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            run_dir=tmp_path / "run",
            model_variant="dual_view_late_fusion",
            view_mode="paired",
        )

        config = load_experiment_config(config_path)

        assert config.data.train_manifest_path == train_manifest.resolve()
        assert config.data.val_manifest_path == val_manifest.resolve()
        assert config.data.view_mode == "paired"
        assert config.data.resize_mode == "nearest"
        assert config.model.variant == "dual_view_late_fusion"
        assert config.model.backbone_name == "light_cnn"
        assert config.training.epochs == 1
        assert config.output.run_dir == (tmp_path / "run").resolve()


def test_train_cli_defaults_to_baseline_config() -> None:
    args = parse_train_args([])

    assert args.config == TRAIN_DEFAULT_CONFIG_PATH
    assert args.run_dir is None


def test_evaluate_cli_defaults_to_baseline_config() -> None:
    args = parse_evaluate_args(["--checkpoint", "dummy.pt"])

    assert args.config == EVAL_DEFAULT_CONFIG_PATH
    assert args.checkpoint == Path("dummy.pt")


@pytest.mark.parametrize(
    ("model_variant", "view_mode", "single_view_name", "backbone_name", "pretrained", "resize"),
    [
        ("single_view", "mask", "mask", "light_cnn", False, None),
        ("dual_view_late_fusion", "paired", "mask", "light_cnn", False, None),
        ("dual_view_late_fusion", "paired", "mask", "resnet18", False, [64, 64]),
    ],
)
def test_one_train_step_runs_for_both_variants(
    monkeypatch: pytest.MonkeyPatch,
    model_variant: str,
    view_mode: str,
    single_view_name: str,
    backbone_name: str,
    pretrained: bool,
    resize: list[int] | None,
) -> None:
    with scratch_dir() as tmp_path:
        if backbone_name == "resnet18":
            pytest.importorskip("torchvision")
        train_manifest = write_manifest(tmp_path / "train.csv", "train", rows=4)
        val_manifest = write_manifest(tmp_path / "val.csv", "val", rows=2, start_index=100)
        config_path = write_config(
            tmp_path / "config.yaml",
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            run_dir=tmp_path / "run",
            model_variant=model_variant,
            view_mode=view_mode,
            single_view_name=single_view_name,
            backbone_name=backbone_name,
            pretrained=pretrained,
            resize=resize,
        )
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        config = load_experiment_config(config_path)
        result = train_model(config, config_path=config_path)

        assert len(result["history"]) == 1
        assert Path(result["best_checkpoint_path"]).is_file()
        assert Path(result["last_checkpoint_path"]).is_file()


def test_training_writes_metrics_and_best_checkpoint_matches_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        train_manifest = write_manifest(tmp_path / "train.csv", "train", rows=6)
        val_manifest = write_manifest(tmp_path / "val.csv", "val", rows=4, start_index=100)
        config_path = write_config(
            tmp_path / "config.yaml",
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            run_dir=tmp_path / "run",
            model_variant="dual_view_late_fusion",
            view_mode="paired",
            epochs=2,
        )
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        config = load_experiment_config(config_path)
        result = train_model(config, config_path=config_path)
        run_dir = Path(result["run_dir"])

        history_path = run_dir / "history.json"
        best_metrics_path = run_dir / "best_metrics.json"
        config_resolved_path = run_dir / "config_resolved.yaml"

        assert history_path.is_file()
        assert best_metrics_path.is_file()
        assert config_resolved_path.is_file()
        assert (run_dir / "best.pt").is_file()
        assert (run_dir / "last.pt").is_file()

        with history_path.open("r", encoding="utf-8") as handle:
            history_payload = json.load(handle)
        with best_metrics_path.open("r", encoding="utf-8") as handle:
            best_metrics_payload = json.load(handle)

        best_history_metric = min(epoch["val_mean_mae"] for epoch in history_payload["history"])
        assert best_metrics_payload["val_mean_mae"] == pytest.approx(best_history_metric)


def test_evaluate_checkpoint_reloads_saved_model_and_reports_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        pytest.importorskip("torchvision")
        train_manifest = write_manifest(tmp_path / "train.csv", "train", rows=4)
        val_manifest = write_manifest(tmp_path / "val.csv", "val", rows=2, start_index=100)
        config_path = write_config(
            tmp_path / "config.yaml",
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            run_dir=tmp_path / "run",
            model_variant="dual_view_late_fusion",
            view_mode="paired",
            backbone_name="resnet18",
            pretrained=False,
            epochs=1,
            resize=[64, 64],
            resize_mode="bilinear",
            normalize_mean=0.449,
            normalize_std=0.226,
            save_predictions_on_eval=True,
        )
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        config = load_experiment_config(config_path)
        train_result = train_model(config, config_path=config_path)
        evaluation_result = evaluate_checkpoint(
            config=config,
            checkpoint_path=train_result["best_checkpoint_path"],
        )

        assert evaluation_result["split"] == "val"
        assert len(evaluation_result["per_target_mae"]) == 14
        assert len(evaluation_result["per_target_rmse"]) == 14
        assert Path(evaluation_result["prediction_output_path"]).is_file()
        assert Path(train_result["best_checkpoint_path"]).parent.joinpath(
            "evaluation_summary_val.json"
        ).is_file()


def test_real_local_small_train_eval_cycle_if_assets_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not REAL_TRAIN_PATH.is_file() or not REAL_VAL_PATH.is_file():
        pytest.skip("Local BodyM train/val split CSVs are not available.")

    with scratch_dir() as tmp_path:
        train_subset = pd.read_csv(REAL_TRAIN_PATH).head(6)
        val_subset = pd.read_csv(REAL_VAL_PATH).head(4)
        train_manifest = tmp_path / "train_subset.csv"
        val_manifest = tmp_path / "val_subset.csv"
        train_subset.to_csv(train_manifest, index=False)
        val_subset.to_csv(val_manifest, index=False)

        config_path = write_config(
            tmp_path / "config.yaml",
            train_manifest_path=train_manifest,
            val_manifest_path=val_manifest,
            run_dir=tmp_path / "run",
            model_variant="dual_view_late_fusion",
            view_mode="paired",
            epochs=1,
            resize=[64, 64],
        )
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        config = load_experiment_config(config_path)
        train_result = train_model(config, config_path=config_path)
        evaluation_result = evaluate_checkpoint(
            config=config,
            checkpoint_path=train_result["best_checkpoint_path"],
        )

        assert Path(train_result["best_checkpoint_path"]).is_file()
        assert Path(train_result["last_checkpoint_path"]).is_file()
        assert evaluation_result["mean_mae"] >= 0.0
        assert evaluation_result["mean_rmse"] >= 0.0
