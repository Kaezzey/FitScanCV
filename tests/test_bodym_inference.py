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

import scripts.inference.bodym_inference as bodym_inference
from scripts.inference.bodym_inference import (
    DEFAULT_CHECKPOINT_PATH,
    BodyMInferenceService,
    load_inference_service,
)
from scripts.models.bodym_models import BodyMModelConfig, build_bodym_model
from scripts.predict_bodym import (
    DEFAULT_SMOKE_MANIFEST_PATH,
    DEFAULT_SMOKE_SPLIT,
    parse_args,
)

SCRATCH_ROOT = REPO_ROOT / "data" / "interim" / "test_scratch"
REAL_VAL_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_val.csv"


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


def write_inference_config(path: Path, resize: list[int] | None = None) -> Path:
    config = {
        "data": {
            "resize": resize,
            "resize_mode": "bilinear" if resize is not None else "nearest",
            "normalize_mean": 0.5,
            "normalize_std": 0.25,
        }
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def write_checkpoint(
    path: Path,
    *,
    variant: str = "dual_view_late_fusion",
    single_view_name: str = "mask",
) -> Path:
    config = BodyMModelConfig(
        variant=variant,  # type: ignore[arg-type]
        backbone_name="light_cnn",
        pretrained=False,
        single_view_name=single_view_name,  # type: ignore[arg-type]
        image_embedding_dim=32,
        metadata_embedding_dim=8,
        hidden_dim=16,
        dropout=0.0,
    )
    model = build_bodym_model(config)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "model_config": {
            "variant": config.variant,
            "num_targets": config.num_targets,
            "single_view_name": config.single_view_name,
            "backbone_name": config.backbone_name,
            "pretrained": config.pretrained,
            "image_embedding_dim": config.image_embedding_dim,
            "metadata_embedding_dim": config.metadata_embedding_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
        },
        "data_view_mode": "paired" if variant == "dual_view_late_fusion" else single_view_name,
        "target_names": [
            "measurement_ankle",
            "measurement_arm_length",
            "measurement_bicep",
            "measurement_calf",
            "measurement_chest",
            "measurement_forearm",
            "measurement_height",
            "measurement_hip",
            "measurement_leg_length",
            "measurement_shoulder_breadth",
            "measurement_shoulder_to_crotch",
            "measurement_thigh",
            "measurement_waist",
            "measurement_wrist",
        ],
        "metrics": {},
    }
    torch.save(checkpoint, path)
    return path


def write_smoke_manifest(path: Path, mask_path: Path, mask_left_path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "split": "val",
                "subject_id": "subject-1",
                "subject_key": "val::subject-1",
                "photo_id": "photo-1",
                "mask_path": str(mask_path),
                "mask_left_path": str(mask_left_path),
                "hwg_gender": "female",
                "hwg_height_cm": "165.0",
                "hwg_weight_kg": "62.0",
            }
        ]
    )
    frame.to_csv(path, index=False)
    return path


def test_cli_defaults_point_to_tuned_checkpoint_and_smoke_manifest() -> None:
    args = parse_args([])

    assert args.checkpoint == DEFAULT_CHECKPOINT_PATH
    assert args.smoke_manifest == DEFAULT_SMOKE_MANIFEST_PATH
    assert args.smoke_split == DEFAULT_SMOKE_SPLIT


def test_service_predicts_from_explicit_dual_view_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        checkpoint_path = write_checkpoint(tmp_path / "best.pt")
        config_path = write_inference_config(tmp_path / "config_resolved.yaml", resize=[64, 64])
        front_path = tmp_path / "front.png"
        side_path = tmp_path / "side.png"
        front_path.write_bytes(b"")
        side_path.write_bytes(b"")
        monkeypatch.setattr(bodym_inference, "_read_grayscale_image", fake_image_tensor)

        service = BodyMInferenceService(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cpu",
        )
        result = service.predict_from_paths(
            front_image_path=front_path,
            side_image_path=side_path,
            hwg_gender="female",
            hwg_height_cm=165.0,
            hwg_weight_kg=62.0,
        )

        assert len(result["predictions"]) == 14
        assert result["model"]["variant"] == "dual_view_late_fusion"
        assert result["model"]["backbone_name"] == "light_cnn"
        assert result["model"]["device"] == "cpu"
        assert result["inputs"]["hwg_gender"] == "female"


def test_load_inference_service_helper_returns_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        checkpoint_path = write_checkpoint(tmp_path / "best.pt")
        config_path = write_inference_config(tmp_path / "config_resolved.yaml")
        monkeypatch.setattr(bodym_inference, "_read_grayscale_image", fake_image_tensor)

        service = load_inference_service(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cpu",
        )

        assert isinstance(service, BodyMInferenceService)
        assert service.target_names[0] == "measurement_ankle"


def test_unknown_gender_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with scratch_dir() as tmp_path:
        checkpoint_path = write_checkpoint(tmp_path / "best.pt")
        config_path = write_inference_config(tmp_path / "config_resolved.yaml")
        front_path = tmp_path / "front.png"
        side_path = tmp_path / "side.png"
        front_path.write_bytes(b"")
        side_path.write_bytes(b"")
        monkeypatch.setattr(bodym_inference, "_read_grayscale_image", fake_image_tensor)
        service = BodyMInferenceService(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cpu",
        )

        with pytest.raises(ValueError, match="Unknown hwg_gender"):
            service.predict_from_paths(
                front_image_path=front_path,
                side_image_path=side_path,
                hwg_gender="unknown",
                hwg_height_cm=165.0,
                hwg_weight_kg=62.0,
            )


def test_missing_image_path_raises_file_not_found() -> None:
    with scratch_dir() as tmp_path:
        checkpoint_path = write_checkpoint(tmp_path / "best.pt")
        config_path = write_inference_config(tmp_path / "config_resolved.yaml")
        side_path = tmp_path / "side.png"
        side_path.write_bytes(b"")
        service = BodyMInferenceService(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cpu",
        )

        with pytest.raises(FileNotFoundError, match="Image file does not exist"):
            service.predict_from_paths(
                front_image_path=tmp_path / "missing.png",
                side_image_path=side_path,
                hwg_gender="female",
                hwg_height_cm=165.0,
                hwg_weight_kg=62.0,
            )


def test_single_view_mask_left_requires_side_image_path() -> None:
    with scratch_dir() as tmp_path:
        checkpoint_path = write_checkpoint(
            tmp_path / "best.pt",
            variant="single_view",
            single_view_name="mask_left",
        )
        config_path = write_inference_config(tmp_path / "config_resolved.yaml")
        front_path = tmp_path / "front.png"
        front_path.write_bytes(b"")
        service = BodyMInferenceService(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cpu",
        )

        with pytest.raises(ValueError, match="mask_left"):
            service.predict_from_paths(
                front_image_path=front_path,
                side_image_path=None,
                hwg_gender="female",
                hwg_height_cm=165.0,
                hwg_weight_kg=62.0,
            )


def test_prediction_payload_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_dir() as tmp_path:
        checkpoint_path = write_checkpoint(tmp_path / "best.pt")
        config_path = write_inference_config(tmp_path / "config_resolved.yaml")
        front_path = tmp_path / "front.png"
        side_path = tmp_path / "side.png"
        front_path.write_bytes(b"")
        side_path.write_bytes(b"")
        monkeypatch.setattr(bodym_inference, "_read_grayscale_image", fake_image_tensor)
        service = BodyMInferenceService(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cpu",
        )

        result = service.predict_from_paths(
            front_image_path=front_path,
            side_image_path=side_path,
            hwg_gender="male",
            hwg_height_cm=180.0,
            hwg_weight_kg=78.0,
        )

        encoded = json.dumps(result)
        assert "measurement_waist" in encoded


def test_smoke_manifest_helper_fixture_shape() -> None:
    with scratch_dir() as tmp_path:
        front_path = tmp_path / "front.png"
        side_path = tmp_path / "side.png"
        front_path.write_bytes(b"")
        side_path.write_bytes(b"")
        manifest_path = write_smoke_manifest(tmp_path / "smoke.csv", front_path, side_path)

        frame = pd.read_csv(manifest_path)

        assert frame.loc[0, "split"] == "val"
        assert frame.loc[0, "mask_path"] == str(front_path)


def test_real_local_tuned_checkpoint_smoke_if_assets_exist() -> None:
    if not DEFAULT_CHECKPOINT_PATH.is_file() or not REAL_VAL_PATH.is_file():
        pytest.skip("Local tuned checkpoint or validation manifest is not available.")
    pytest.importorskip("torchvision")

    frame = pd.read_csv(REAL_VAL_PATH).head(1)
    if frame.empty:
        pytest.skip("Validation manifest has no rows.")
    row = frame.iloc[0]

    service = load_inference_service(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device="cpu",
    )
    result = service.predict_from_paths(
        front_image_path=row["mask_path"],
        side_image_path=row["mask_left_path"],
        hwg_gender=row["hwg_gender"],
        hwg_height_cm=float(row["hwg_height_cm"]),
        hwg_weight_kg=float(row["hwg_weight_kg"]),
    )

    assert result["model"]["backbone_name"] == "resnet18"
    assert len(result["predictions"]) == 14
