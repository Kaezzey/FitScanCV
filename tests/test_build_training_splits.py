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
from scripts.bodym_dataset import BodyMManifestDataset
from scripts.build_training_splits import (
    TrainingSplitError,
    build_training_split_artifacts,
    get_trainable_target_columns,
    load_manifest_frame,
    validate_subject_level_consistency,
    validate_target_columns,
    write_split_frame,
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


def make_split_manifest(tmp_path: Path, subjects_per_split: int = 10, photos_per_subject: int = 2) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    rows: list[dict[str, str]] = []
    measurement_values = {
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

    for split_name in ("train", "testA", "testB"):
        for subject_index in range(subjects_per_split):
            gender = "male" if subject_index % 2 == 0 else "female"
            subject_key = f"{split_name}::subject-{subject_index}"
            subject_id = f"subject-{subject_index}"
            for photo_index in range(photos_per_subject):
                photo_id = f"{split_name}-photo-{subject_index}-{photo_index}"
                mask_path = tmp_path / f"{photo_id}-mask.png"
                mask_left_path = tmp_path / f"{photo_id}-mask-left.png"
                mask_path.write_bytes(b"")
                mask_left_path.write_bytes(b"")

                row = {
                    "split": split_name,
                    "subject_id": subject_id,
                    "subject_key": subject_key,
                    "photo_id": photo_id,
                    "mask_path": str(mask_path),
                    "mask_left_path": str(mask_left_path),
                    "hwg_gender": gender,
                    "hwg_height_cm": "170.0",
                    "hwg_weight_kg": "70.0",
                }
                row.update(measurement_values)
                rows.append(row)

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def fake_image_tensor(_: Path) -> torch.Tensor:
    return torch.ones((1, 960, 720), dtype=torch.float32)


def test_get_trainable_target_columns_detects_all_measurements() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        frame = load_manifest_frame(manifest_path)

        target_columns = get_trainable_target_columns(frame)

        assert len(target_columns) == 14
        assert all(column.startswith("measurement_") for column in target_columns)


def test_get_trainable_target_columns_errors_when_none_exist() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        frame = pd.read_csv(manifest_path)
        frame = frame.drop(columns=[column for column in frame.columns if column.startswith("measurement_")])
        frame.to_csv(manifest_path, index=False)

        loaded_frame = load_manifest_frame(manifest_path)

        with pytest.raises(TrainingSplitError, match="does not contain any trainable target columns"):
            get_trainable_target_columns(loaded_frame)


def test_validate_target_columns_errors_on_missing_values() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        frame = load_manifest_frame(manifest_path)
        target_columns = get_trainable_target_columns(frame)
        frame.loc[0, "measurement_waist"] = ""

        with pytest.raises(TrainingSplitError, match="empty values"):
            validate_target_columns(frame, target_columns)


def test_validate_target_columns_errors_on_non_numeric_values() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        frame = load_manifest_frame(manifest_path)
        target_columns = get_trainable_target_columns(frame)
        frame.loc[0, "measurement_waist"] = "not-a-number"

        with pytest.raises(TrainingSplitError, match="non-numeric values"):
            validate_target_columns(frame, target_columns)


def test_validate_subject_level_consistency_errors_on_target_variation() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        frame = load_manifest_frame(manifest_path)
        frame["measurement_waist"] = pd.to_numeric(frame["measurement_waist"])
        frame.loc[1, "measurement_waist"] = 99.0

        with pytest.raises(TrainingSplitError, match="varies within subject_key"):
            validate_subject_level_consistency(frame, ["measurement_waist"])


def test_validate_subject_level_consistency_errors_on_hwg_variation() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        frame = load_manifest_frame(manifest_path)
        frame["hwg_weight_kg"] = pd.to_numeric(frame["hwg_weight_kg"])
        frame.loc[1, "hwg_weight_kg"] = 71.0

        with pytest.raises(TrainingSplitError, match="varies within subject_key"):
            validate_subject_level_consistency(frame, ["hwg_weight_kg"])


def test_build_training_split_artifacts_creates_disjoint_subject_splits() -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)

        train_frame, val_frame, summary = build_training_split_artifacts(
            manifest_path=manifest_path,
            val_size=0.10,
            random_state=42,
        )

        assert set(train_frame["split"].unique()) == {"train"}
        assert set(val_frame["split"].unique()) == {"val"}
        assert set(train_frame["source_split"].unique()) == {"train"}
        assert set(val_frame["source_split"].unique()) == {"train"}
        assert not (
            set(train_frame["subject_key"].unique()) & set(val_frame["subject_key"].unique())
        )
        assert len(train_frame.index) + len(val_frame.index) == 20
        assert summary["source_train"]["row_count"] == 20
        assert summary["outputs"]["train"]["row_count"] + summary["outputs"]["val"]["row_count"] == 20
        assert summary["holdouts"]["testA"]["row_count"] == 20
        assert summary["holdouts"]["testB"]["row_count"] == 20


def test_generated_split_csvs_load_with_bodym_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        manifest_path = make_split_manifest(tmp_path)
        train_frame, val_frame, _ = build_training_split_artifacts(
            manifest_path=manifest_path,
            val_size=0.10,
            random_state=42,
        )
        train_output = tmp_path / "train.csv"
        val_output = tmp_path / "val.csv"
        write_split_frame(train_frame, train_output)
        write_split_frame(val_frame, val_output)

        monkeypatch.setattr(bodym_dataset, "_read_grayscale_image", fake_image_tensor)

        train_dataset = BodyMManifestDataset(train_output, split="train")
        val_dataset = BodyMManifestDataset(val_output, split="val")

        assert len(train_dataset) == len(train_frame.index)
        assert len(val_dataset) == len(val_frame.index)
        assert train_dataset[0]["metadata"]["hwg_gender"] in {"male", "female"}
        assert val_dataset[0]["metadata"]["hwg_gender"] in {"male", "female"}


def test_real_manifest_acceptance_if_local_assets_exist() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip("Local BodyM manifest is not available.")

    train_frame, val_frame, summary = build_training_split_artifacts(
        manifest_path=MANIFEST_PATH,
        val_size=0.10,
        random_state=42,
    )

    assert len(summary["trainable_target_columns"]) == 14
    assert all(
        stats["missing_count"] == 0 for stats in summary["target_summary"].values()
    )
    assert summary["source_train"]["row_count"] == 6134
    assert summary["source_train"]["subject_count"] == 2018
    assert len(train_frame.index) + len(val_frame.index) == 6134
    assert train_frame["subject_key"].nunique() + val_frame["subject_key"].nunique() == 2018
    assert not (
        set(train_frame["subject_key"].unique()) & set(val_frame["subject_key"].unique())
    )
