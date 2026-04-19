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
from scripts.bodym_dataset import (
    BodyMDataError,
    BodyMManifestDataset,
    BodyMTransformConfig,
    build_bodym_transform,
    create_bodym_dataloader,
)

MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_manifest.csv"
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


def require_local_bodym_assets() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip("Local BodyM manifest is not available.")


def require_torchvision() -> None:
    pytest.importorskip("torchvision")


def make_minimal_manifest(tmp_path: Path, rows: int = 1) -> Path:
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
                "hwg_gender": "female",
                "hwg_height_cm": "160.0",
                "hwg_weight_kg": "60.0",
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


def test_dataset_paired_mode_returns_both_views_and_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        dataset = BodyMManifestDataset(manifest_path, split="train")
        sample = dataset[0]

        assert dataset.view_names == ("mask", "mask_left")
        assert len(dataset.target_columns) == 14
        assert sample["metadata"] == {
            "hwg_gender": "female",
            "hwg_height_cm": 160.0,
            "hwg_weight_kg": 60.0,
        }
        assert set(sample["views"].keys()) == {"mask", "mask_left"}
        assert sample["views"]["mask"].dtype == torch.float32
        assert tuple(sample["views"]["mask"].shape) == (1, 960, 720)
        assert tuple(sample["views"]["mask_left"].shape) == (1, 960, 720)
        assert sample["targets"].dtype == torch.float32
        assert tuple(sample["targets"].shape) == (14,)


@pytest.mark.parametrize(
    ("view_mode", "expected_key"),
    [
        ("mask", "mask"),
        ("mask_left", "mask_left"),
    ],
)
def test_dataset_single_view_modes_return_single_active_view(
    monkeypatch: pytest.MonkeyPatch,
    view_mode: str,
    expected_key: str,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        dataset = BodyMManifestDataset(
            manifest_path,
            split="train",
            view_mode=view_mode,  # type: ignore[arg-type]
        )
        sample = dataset[0]

        assert dataset.view_names == (expected_key,)
        assert tuple(sample["views"].keys()) == (expected_key,)
        assert tuple(sample["views"][expected_key].shape) == (1, 960, 720)


def test_dataset_resize_transform_changes_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        transform = build_bodym_transform(BodyMTransformConfig(resize=(256, 192)))
        dataset = BodyMManifestDataset(
            manifest_path,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        assert tuple(sample["views"]["mask"].shape) == (1, 256, 192)
        assert tuple(sample["views"]["mask_left"].shape) == (1, 256, 192)


def test_dataset_transform_supports_bilinear_resize_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        transform = build_bodym_transform(
            BodyMTransformConfig(
                resize=(128, 128),
                resize_mode="bilinear",
                normalize_mean=0.5,
                normalize_std=0.25,
            )
        )
        dataset = BodyMManifestDataset(
            manifest_path,
            split="train",
            transform=transform,
        )
        sample = dataset[0]

        assert tuple(sample["views"]["mask"].shape) == (1, 128, 128)
        assert float(sample["views"]["mask"].mean().item()) == pytest.approx(2.0)


def test_dataloader_batch_collates_views_and_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path, rows=4)
        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        dataloader = create_bodym_dataloader(
            manifest_path,
            split="train",
            batch_size=4,
            shuffle=False,
        )
        batch = next(iter(dataloader))

        assert set(batch["metadata"].keys()) == {
            "hwg_gender",
            "hwg_height_cm",
            "hwg_weight_kg",
        }
        assert tuple(batch["views"]["mask"].shape) == (4, 1, 960, 720)
        assert tuple(batch["views"]["mask_left"].shape) == (4, 1, 960, 720)
        assert tuple(batch["targets"].shape) == (4, 14)
        assert len(batch["photo_id"]) == 4
        assert len(batch["subject_id"]) == 4


def test_invalid_view_mode_raises_value_error() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)

        with pytest.raises(ValueError, match="view_mode"):
            BodyMManifestDataset(
                manifest_path,
                split="train",
                view_mode="front",  # type: ignore[arg-type]
            )


def test_missing_manifest_columns_raise_error() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)
        frame = pd.read_csv(manifest_path)
        frame = frame.drop(columns=["measurement_waist"])
        frame.to_csv(manifest_path, index=False)

        with pytest.raises(BodyMDataError, match="missing required columns"):
            BodyMManifestDataset(manifest_path, split="train")


def test_missing_image_path_raises_file_not_found() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_minimal_manifest(tmp_path)
        frame = pd.read_csv(manifest_path)
        frame.loc[0, "mask_path"] = str(tmp_path / "does-not-exist.png")
        frame.to_csv(manifest_path, index=False)

        with pytest.raises(FileNotFoundError, match="mask_path"):
            BodyMManifestDataset(manifest_path, split="train")


def test_real_local_assets_load_when_torchvision_is_available() -> None:
    require_local_bodym_assets()
    require_torchvision()

    dataset = BodyMManifestDataset(MANIFEST_PATH, split="train")
    sample = dataset[0]

    assert dataset.view_names == ("mask", "mask_left")
    assert len(dataset.target_columns) == 14
    assert set(sample["metadata"].keys()) == {
        "hwg_gender",
        "hwg_height_cm",
        "hwg_weight_kg",
    }
    assert tuple(sample["views"]["mask"].shape) == (1, 960, 720)
    assert tuple(sample["views"]["mask_left"].shape) == (1, 960, 720)
    assert tuple(sample["targets"].shape) == (14,)
