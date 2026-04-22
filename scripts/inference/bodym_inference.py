from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("torch is required to use scripts.inference.bodym_inference.") from exc

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml is required to use scripts.inference.bodym_inference.") from exc

from scripts.bodym_dataset import (
    BodyMTransformConfig,
    _read_grayscale_image,
    build_bodym_transform,
)
from scripts.models.bodym_models import BodyMModelConfig, build_bodym_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_resnet18" / "best.pt"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "bodym_resnet18.yaml"


class BodyMInferenceError(RuntimeError):
    """Raised when local BodyM inference setup or inputs are invalid."""


@dataclass(frozen=True)
class BodyMInferenceConfig:
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    config_path: Path | None = None
    device: str = "auto"


@dataclass(frozen=True)
class BodyMPreprocessConfig:
    resize: tuple[int, int] | None = None
    resize_mode: str = "nearest"
    normalize_mean: float | None = None
    normalize_std: float | None = None


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise BodyMInferenceError("CUDA was requested but is not available.")
    if normalized not in ("cpu", "cuda"):
        raise BodyMInferenceError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(normalized)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    resolved_checkpoint_path = resolve_repo_path(checkpoint_path)
    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint_path}")

    try:
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
    except OSError as exc:
        raise BodyMInferenceError(
            f"Failed to load checkpoint: {resolved_checkpoint_path}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise BodyMInferenceError("Checkpoint payload is invalid.")
    return checkpoint


def _read_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise BodyMInferenceError(f"Failed to read config file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise BodyMInferenceError(f"Failed to parse YAML config: {config_path}") from exc
    if not isinstance(config, dict):
        raise BodyMInferenceError("Top-level YAML config must be a mapping.")
    return config


def _resolve_preprocess_config_path(
    checkpoint_path: Path,
    config_path: str | Path | None,
) -> Path:
    if config_path is not None:
        return resolve_repo_path(config_path)

    resolved_checkpoint_path = resolve_repo_path(checkpoint_path)
    checkpoint_config_path = resolved_checkpoint_path.parent / "config_resolved.yaml"
    if checkpoint_config_path.is_file():
        return checkpoint_config_path.resolve()
    return DEFAULT_CONFIG_PATH.resolve()


def _normalize_resize(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise BodyMInferenceError("data.resize must be null or a two-item list.")
    height, width = value
    return int(height), int(width)


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_preprocess_config(config_path: str | Path) -> BodyMPreprocessConfig:
    resolved_config_path = resolve_repo_path(config_path)
    config = _read_yaml_config(resolved_config_path)
    data_section = config.get("data")
    if not isinstance(data_section, dict):
        raise BodyMInferenceError("Config is missing required 'data' section.")

    return BodyMPreprocessConfig(
        resize=_normalize_resize(data_section.get("resize")),
        resize_mode=str(data_section.get("resize_mode", "nearest")),
        normalize_mean=_normalize_optional_float(data_section.get("normalize_mean")),
        normalize_std=_normalize_optional_float(data_section.get("normalize_std")),
    )


def _build_transform(preprocess_config: BodyMPreprocessConfig):
    return build_bodym_transform(
        BodyMTransformConfig(
            resize=preprocess_config.resize,
            resize_mode=preprocess_config.resize_mode,  # type: ignore[arg-type]
            normalize_mean=preprocess_config.normalize_mean,
            normalize_std=preprocess_config.normalize_std,
        )
    )


class BodyMInferenceService:
    """Local filesystem inference service for trained BodyM checkpoints."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        config_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = resolve_repo_path(checkpoint_path)
        self.config_path = _resolve_preprocess_config_path(
            checkpoint_path=self.checkpoint_path,
            config_path=config_path,
        )
        self.device = resolve_device(device)
        self.checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)
        self.model_config = BodyMModelConfig(**self.checkpoint["model_config"])
        self.target_names = tuple(str(name) for name in self.checkpoint.get("target_names", []))
        if not self.target_names:
            raise BodyMInferenceError("Checkpoint is missing target_names.")
        if len(self.target_names) != self.model_config.num_targets:
            raise BodyMInferenceError(
                "Checkpoint target_names length does not match model_config.num_targets."
            )

        self.preprocess_config = load_preprocess_config(self.config_path)
        self.transform = _build_transform(self.preprocess_config)
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = build_bodym_model(
            self.model_config,
            initialize_pretrained=False,
        ).to(self.device)
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.eval()
        return model

    def predict_from_paths(
        self,
        front_image_path: str | Path,
        side_image_path: str | Path | None,
        hwg_gender: str,
        hwg_height_cm: float,
        hwg_weight_kg: float,
    ) -> dict[str, Any]:
        metadata = {
            "hwg_gender": [hwg_gender],
            "hwg_height_cm": torch.tensor([float(hwg_height_cm)], dtype=torch.float32),
            "hwg_weight_kg": torch.tensor([float(hwg_weight_kg)], dtype=torch.float32),
        }

        front_path = resolve_repo_path(front_image_path)
        side_path = resolve_repo_path(side_image_path) if side_image_path is not None else None

        with torch.no_grad():
            if self.model_config.variant == "dual_view_late_fusion":
                if side_path is None:
                    raise ValueError("Dual-view inference requires side_image_path.")
                mask = self._load_view(front_path)
                mask_left = self._load_view(side_path)
                predictions = self.model(mask, mask_left, metadata)  # type: ignore[misc]
            elif self.model_config.variant == "single_view":
                if self.model_config.single_view_name == "mask_left":
                    if side_path is None:
                        raise ValueError(
                            "Single-view mask_left inference requires side_image_path."
                        )
                    required_path = side_path
                else:
                    required_path = front_path
                view = self._load_view(required_path)
                predictions = self.model(view, metadata)  # type: ignore[misc]
            else:
                raise BodyMInferenceError(
                    f"Unsupported model variant: {self.model_config.variant!r}"
                )

        prediction_values = predictions.detach().cpu().reshape(-1).tolist()
        if len(prediction_values) != len(self.target_names):
            raise BodyMInferenceError("Prediction count does not match target_names.")

        return {
            "predictions": {
                name: float(prediction_values[index])
                for index, name in enumerate(self.target_names)
            },
            "model": {
                "checkpoint_path": str(self.checkpoint_path),
                "config_path": str(self.config_path),
                "variant": self.model_config.variant,
                "backbone_name": self.model_config.backbone_name,
                "single_view_name": self.model_config.single_view_name,
                "target_names": list(self.target_names),
                "device": str(self.device),
            },
            "inputs": {
                "front_image_path": str(front_path),
                "side_image_path": str(side_path) if side_path is not None else None,
                "hwg_gender": hwg_gender,
                "hwg_height_cm": float(hwg_height_cm),
                "hwg_weight_kg": float(hwg_weight_kg),
            },
        }

    def _load_view(self, image_path: Path) -> torch.Tensor:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        image = _read_grayscale_image(image_path)
        transformed = self.transform(image)
        return transformed.unsqueeze(0).to(self.device)


def load_inference_service(
    checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str = "auto",
) -> BodyMInferenceService:
    return BodyMInferenceService(
        checkpoint_path=checkpoint_path or DEFAULT_CHECKPOINT_PATH,
        config_path=config_path,
        device=device,
    )


__all__ = [
    "BodyMInferenceConfig",
    "BodyMInferenceError",
    "BodyMInferenceService",
    "BodyMPreprocessConfig",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CONFIG_PATH",
    "load_inference_service",
    "load_preprocess_config",
]
