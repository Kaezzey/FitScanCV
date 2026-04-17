from __future__ import annotations

from contextlib import contextmanager
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

import scripts.bodym_dataset as bodym_dataset
from scripts.bodym_dataset import create_bodym_dataloader
from scripts.models.bodym_models import (
    BodyMModelConfig,
    DualViewLateFusionBodyMRegressor,
    SingleViewBodyMRegressor,
    build_bodym_model,
    build_regression_loss,
)

TRAIN_SPLIT_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_train.csv"
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


def make_minimal_manifest(tmp_path: Path, rows: int = 4) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    manifest_rows: list[dict[str, str]] = []

    for index in range(rows):
        photo_id = f"photo-{index}"
        mask_path = tmp_path / f"{photo_id}-mask.png"
        mask_left_path = tmp_path / f"{photo_id}-mask-left.png"
        mask_path.write_bytes(b"")
        mask_left_path.write_bytes(b"")

        manifest_rows.append(
            {
                "split": "train",
                "subject_id": f"subject-{index}",
                "subject_key": f"train::subject-{index}",
                "photo_id": photo_id,
                "mask_path": str(mask_path),
                "mask_left_path": str(mask_left_path),
                "hwg_gender": "female" if index % 2 == 0 else "male",
                "hwg_height_cm": str(160.0 + index),
                "hwg_weight_kg": str(60.0 + index),
                "measurement_ankle": "21.0",
                "measurement_arm_length": "45.0",
                "measurement_bicep": "30.0",
                "measurement_calf": "35.0",
                "measurement_chest": "95.0",
                "measurement_forearm": "25.0",
                "measurement_height": "161.0",
                "measurement_hip": "96.0",
                "measurement_leg_length": "72.0",
                "measurement_shoulder_breadth": "36.0",
                "measurement_shoulder_to_crotch": "61.0",
                "measurement_thigh": "52.0",
                "measurement_waist": "83.0",
                "measurement_wrist": "15.0",
            }
        )

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def fake_image_tensor(_: Path) -> torch.Tensor:
    return torch.ones((1, 960, 720), dtype=torch.float32)


def make_metadata_batch(batch_size: int) -> dict[str, object]:
    gender_values = ["female" if index % 2 == 0 else "male" for index in range(batch_size)]
    height_values = torch.linspace(160.0, 160.0 + batch_size - 1, steps=batch_size)
    weight_values = torch.linspace(60.0, 60.0 + batch_size - 1, steps=batch_size)
    return {
        "hwg_gender": gender_values,
        "hwg_height_cm": height_values,
        "hwg_weight_kg": weight_values,
    }


@pytest.mark.parametrize("single_view_name", ["mask", "mask_left"])
def test_single_view_model_forward_supports_both_views(single_view_name: str) -> None:
    config = BodyMModelConfig(
        variant="single_view",
        single_view_name=single_view_name,  # type: ignore[arg-type]
    )
    model = build_bodym_model(config)
    view = torch.ones((2, 1, 960, 720), dtype=torch.float32)
    metadata = make_metadata_batch(batch_size=2)

    predictions = model(view, metadata)

    assert isinstance(model, SingleViewBodyMRegressor)
    assert predictions.dtype == torch.float32
    assert tuple(predictions.shape) == (2, 14)


def test_dual_view_late_fusion_model_forward_shape() -> None:
    config = BodyMModelConfig(variant="dual_view_late_fusion")
    model = build_bodym_model(config)
    mask = torch.ones((2, 1, 960, 720), dtype=torch.float32)
    mask_left = torch.ones((2, 1, 960, 720), dtype=torch.float32)
    metadata = make_metadata_batch(batch_size=2)

    predictions = model(mask, mask_left, metadata)

    assert isinstance(model, DualViewLateFusionBodyMRegressor)
    assert predictions.dtype == torch.float32
    assert tuple(predictions.shape) == (2, 14)


def test_invalid_variant_errors_clearly() -> None:
    with pytest.raises(ValueError, match="variant"):
        BodyMModelConfig(variant="ensemble")  # type: ignore[arg-type]


def test_invalid_single_view_name_errors_clearly() -> None:
    with pytest.raises(ValueError, match="single_view_name"):
        BodyMModelConfig(single_view_name="front")  # type: ignore[arg-type]


def test_unknown_hwg_gender_raises_value_error() -> None:
    model = SingleViewBodyMRegressor(BodyMModelConfig(variant="single_view"))
    view = torch.ones((2, 1, 960, 720), dtype=torch.float32)
    metadata = {
        "hwg_gender": ["female", "unknown"],
        "hwg_height_cm": torch.tensor([160.0, 161.0], dtype=torch.float32),
        "hwg_weight_kg": torch.tensor([60.0, 61.0], dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="Unknown hwg_gender value"):
        model(view, metadata)


def test_build_regression_loss_returns_scalar_tensor() -> None:
    predictions = torch.zeros((2, 14), dtype=torch.float32)
    targets = torch.ones((2, 14), dtype=torch.float32)

    smooth_l1_loss = build_regression_loss()
    mse_loss = build_regression_loss("mse")

    smooth_l1_value = smooth_l1_loss(predictions, targets)
    mse_value = mse_loss(predictions, targets)

    assert smooth_l1_value.ndim == 0
    assert mse_value.ndim == 0
    assert smooth_l1_value.dtype == torch.float32
    assert mse_value.dtype == torch.float32


def test_models_accept_loader_batch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path, rows=4)
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        dataloader = create_bodym_dataloader(
            manifest_path=manifest_path,
            split="train",
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(dataloader))

        single_view_model = SingleViewBodyMRegressor(
            BodyMModelConfig(variant="single_view", single_view_name="mask")
        )
        dual_view_model = DualViewLateFusionBodyMRegressor(
            BodyMModelConfig(variant="dual_view_late_fusion")
        )

        single_view_predictions = single_view_model(
            batch["views"]["mask"],
            batch["metadata"],
        )
        dual_view_predictions = dual_view_model(
            batch["views"]["mask"],
            batch["views"]["mask_left"],
            batch["metadata"],
        )

        assert tuple(single_view_predictions.shape) == (2, 14)
        assert tuple(dual_view_predictions.shape) == (2, 14)


def test_real_local_training_split_smoke_if_assets_exist() -> None:
    if not TRAIN_SPLIT_PATH.is_file():
        pytest.skip("Local BodyM training split CSV is not available.")
    pytest.importorskip("torchvision")

    dataloader = create_bodym_dataloader(
        manifest_path=TRAIN_SPLIT_PATH,
        split="train",
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    model = DualViewLateFusionBodyMRegressor(
        BodyMModelConfig(variant="dual_view_late_fusion")
    )
    predictions = model(
        batch["views"]["mask"],
        batch["views"]["mask_left"],
        batch["metadata"],
    )

    assert tuple(predictions.shape) == (2, 14)
    assert predictions.dtype == torch.float32
