from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    from torch import Tensor, nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError("torch is required to use scripts.models.bodym_training.") from exc

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml is required to use scripts.models.bodym_training.") from exc

from scripts.bodym_dataset import (
    BodyMTransformConfig,
    build_bodym_transform,
    create_bodym_dataloader,
)
from scripts.models.bodym_models import BodyMModelConfig, build_bodym_model, build_regression_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
VALID_OPTIMIZERS: tuple[str, ...] = ("adam",)
VALID_LOSSES: tuple[str, ...] = ("smooth_l1", "mse")


class TrainingPipelineError(RuntimeError):
    """Raised when the training pipeline configuration or runtime is invalid."""


@dataclass(frozen=True)
class BodyMDataConfig:
    train_manifest_path: Path
    val_manifest_path: Path
    train_split: str = "train"
    val_split: str = "val"
    view_mode: Literal["paired", "mask", "mask_left"] = "paired"
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    resize: tuple[int, int] | None = None
    resize_mode: Literal["nearest", "bilinear"] = "nearest"
    normalize_mean: float | None = None
    normalize_std: float | None = None

    def __post_init__(self) -> None:
        if not self.train_split:
            raise ValueError("train_split must be a non-empty string.")
        if not self.val_split:
            raise ValueError("val_split must be a non-empty string.")
        if self.view_mode not in ("paired", "mask", "mask_left"):
            raise ValueError("view_mode must be 'paired', 'mask', or 'mask_left'.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be zero or greater.")
        if self.resize is not None and len(self.resize) != 2:
            raise ValueError("resize must be null or a [height, width] pair.")
        if self.resize_mode not in ("nearest", "bilinear"):
            raise ValueError("resize_mode must be 'nearest' or 'bilinear'.")
        if (self.normalize_mean is None) != (self.normalize_std is None):
            raise ValueError(
                "normalize_mean and normalize_std must either both be set or both be null."
            )
        if self.normalize_std is not None and self.normalize_std <= 0.0:
            raise ValueError("normalize_std must be greater than zero.")


@dataclass(frozen=True)
class BodyMTrainingConfig:
    epochs: int = 5
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    optimizer_name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_name: str = "smooth_l1"
    log_every_n_steps: int = 50

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}.")
        if self.optimizer_name not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer_name must be one of {VALID_OPTIMIZERS}.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be zero or greater.")
        if self.loss_name not in VALID_LOSSES:
            raise ValueError(f"loss_name must be one of {VALID_LOSSES}.")
        if self.log_every_n_steps < 0:
            raise ValueError("log_every_n_steps must be zero or greater.")


@dataclass(frozen=True)
class BodyMOutputConfig:
    run_dir: Path
    save_predictions_on_eval: bool = False


@dataclass(frozen=True)
class BodyMExperimentConfig:
    data: BodyMDataConfig
    model: BodyMModelConfig
    training: BodyMTrainingConfig
    output: BodyMOutputConfig


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _normalize_resize(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TrainingPipelineError("data.resize must be null or a two-item list.")
    height, width = value
    return int(height), int(width)


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_section(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    section = raw_config.get(key)
    if not isinstance(section, dict):
        raise TrainingPipelineError(f"Config is missing required '{key}' section.")
    return section


def validate_experiment_config(config: BodyMExperimentConfig) -> None:
    if config.model.variant == "single_view":
        if config.data.view_mode != config.model.single_view_name:
            raise TrainingPipelineError(
                "Single-view training requires data.view_mode to match "
                f"model.single_view_name ({config.model.single_view_name!r})."
            )
    elif config.model.variant == "dual_view_late_fusion":
        if config.data.view_mode != "paired":
            raise TrainingPipelineError(
                "Dual-view late-fusion training requires data.view_mode='paired'."
            )


def load_experiment_config(
    config_path: str | Path,
    run_dir_override: str | Path | None = None,
) -> BodyMExperimentConfig:
    resolved_config_path = resolve_repo_path(config_path)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {resolved_config_path}")

    try:
        with resolved_config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise TrainingPipelineError(f"Failed to read config file: {resolved_config_path}") from exc
    except yaml.YAMLError as exc:
        raise TrainingPipelineError(f"Failed to parse YAML config: {resolved_config_path}") from exc

    if not isinstance(raw_config, dict):
        raise TrainingPipelineError("Top-level YAML config must be a mapping.")

    data_section = _require_section(raw_config, "data")
    model_section = _require_section(raw_config, "model")
    training_section = _require_section(raw_config, "training")
    output_section = _require_section(raw_config, "output")

    data_config = BodyMDataConfig(
        train_manifest_path=resolve_repo_path(data_section["train_manifest_path"]),
        val_manifest_path=resolve_repo_path(data_section["val_manifest_path"]),
        train_split=str(data_section.get("train_split", "train")),
        val_split=str(data_section.get("val_split", "val")),
        view_mode=str(data_section.get("view_mode", "paired")),  # type: ignore[arg-type]
        batch_size=int(data_section.get("batch_size", 8)),
        num_workers=int(data_section.get("num_workers", 0)),
        pin_memory=bool(data_section.get("pin_memory", False)),
        resize=_normalize_resize(data_section.get("resize")),
        resize_mode=str(data_section.get("resize_mode", "nearest")),  # type: ignore[arg-type]
        normalize_mean=_normalize_optional_float(data_section.get("normalize_mean")),
        normalize_std=_normalize_optional_float(data_section.get("normalize_std")),
    )
    model_config = BodyMModelConfig(**model_section)
    training_config = BodyMTrainingConfig(
        epochs=int(training_section.get("epochs", 5)),
        seed=int(training_section.get("seed", 42)),
        device=str(training_section.get("device", "auto")),  # type: ignore[arg-type]
        optimizer_name=str(training_section.get("optimizer_name", "adam")),
        learning_rate=float(training_section.get("learning_rate", 1e-3)),
        weight_decay=float(training_section.get("weight_decay", 1e-4)),
        loss_name=str(training_section.get("loss_name", "smooth_l1")),
        log_every_n_steps=int(training_section.get("log_every_n_steps", 50)),
    )
    output_run_dir = (
        resolve_repo_path(run_dir_override)
        if run_dir_override is not None
        else resolve_repo_path(output_section["run_dir"])
    )
    output_config = BodyMOutputConfig(
        run_dir=output_run_dir,
        save_predictions_on_eval=bool(output_section.get("save_predictions_on_eval", False)),
    )

    experiment_config = BodyMExperimentConfig(
        data=data_config,
        model=model_config,
        training=training_config,
        output=output_config,
    )
    validate_experiment_config(experiment_config)
    return experiment_config


def experiment_config_to_dict(config: BodyMExperimentConfig) -> dict[str, Any]:
    return {
        "data": {
            "train_manifest_path": str(config.data.train_manifest_path),
            "val_manifest_path": str(config.data.val_manifest_path),
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
            "view_mode": config.data.view_mode,
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
            "pin_memory": config.data.pin_memory,
            "resize": list(config.data.resize) if config.data.resize is not None else None,
            "resize_mode": config.data.resize_mode,
            "normalize_mean": config.data.normalize_mean,
            "normalize_std": config.data.normalize_std,
        },
        "model": asdict(config.model),
        "training": asdict(config.training),
        "output": {
            "run_dir": str(config.output.run_dir),
            "save_predictions_on_eval": config.output.save_predictions_on_eval,
        },
    }


def save_yaml(data: dict[str, Any], output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    except OSError as exc:
        raise TrainingPipelineError(f"Failed to write YAML file: {output_path}") from exc


def save_json(data: dict[str, Any], output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise TrainingPipelineError(f"Failed to write JSON file: {output_path}") from exc


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise TrainingPipelineError("CUDA was requested but is not available.")
    if normalized not in ("cpu", "cuda"):
        raise TrainingPipelineError(f"Unsupported device name: {device_name!r}")
    return torch.device(normalized)


def build_dataloaders(
    config: BodyMExperimentConfig,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    if (
        config.data.resize is not None
        or config.data.normalize_mean is not None
        or config.data.normalize_std is not None
    ):
        transform = build_bodym_transform(
            BodyMTransformConfig(
                resize=config.data.resize,
                resize_mode=config.data.resize_mode,
                normalize_mean=config.data.normalize_mean,
                normalize_std=config.data.normalize_std,
            )
        )
    else:
        transform = None
    train_loader = create_bodym_dataloader(
        manifest_path=config.data.train_manifest_path,
        split=config.data.train_split,
        view_mode=config.data.view_mode,
        transform=transform,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_bodym_dataloader(
        manifest_path=config.data.val_manifest_path,
        split=config.data.val_split,
        view_mode=config.data.view_mode,
        transform=transform,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    return train_loader, val_loader


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved_views = {
        name: tensor.to(device)
        for name, tensor in batch["views"].items()
    }
    moved_metadata: dict[str, Any] = {}
    for key, value in batch["metadata"].items():
        moved_metadata[key] = value.to(device) if isinstance(value, torch.Tensor) else value

    return {
        **batch,
        "views": moved_views,
        "metadata": moved_metadata,
        "targets": batch["targets"].to(device),
    }


def forward_model_from_batch(
    model: nn.Module,
    batch: dict[str, Any],
    model_config: BodyMModelConfig,
) -> Tensor:
    if model_config.variant == "single_view":
        return model(  # type: ignore[misc]
            batch["views"][model_config.single_view_name],
            batch["metadata"],
        )
    if model_config.variant == "dual_view_late_fusion":
        return model(  # type: ignore[misc]
            batch["views"]["mask"],
            batch["views"]["mask_left"],
            batch["metadata"],
        )
    raise TrainingPipelineError(f"Unsupported model variant: {model_config.variant!r}")


def _build_metric_state(target_names: tuple[str, ...]) -> dict[str, Any]:
    target_count = len(target_names)
    return {
        "target_names": target_names,
        "loss_total": 0.0,
        "sample_count": 0,
        "abs_error_sum": torch.zeros(target_count, dtype=torch.float64),
        "sq_error_sum": torch.zeros(target_count, dtype=torch.float64),
    }


def _update_metric_state(
    state: dict[str, Any],
    loss: Tensor,
    predictions: Tensor,
    targets: Tensor,
) -> None:
    batch_size = int(targets.shape[0])
    state["loss_total"] += float(loss.detach().cpu()) * batch_size
    state["sample_count"] += batch_size

    prediction_cpu = predictions.detach().cpu().to(dtype=torch.float64)
    target_cpu = targets.detach().cpu().to(dtype=torch.float64)
    diff = prediction_cpu - target_cpu
    state["abs_error_sum"] += torch.abs(diff).sum(dim=0)
    state["sq_error_sum"] += diff.pow(2).sum(dim=0)


def _finalize_metric_state(state: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(state["sample_count"])
    if sample_count <= 0:
        raise TrainingPipelineError("Metric finalization requires at least one sample.")

    target_names = state["target_names"]
    per_target_mae_tensor = state["abs_error_sum"] / sample_count
    per_target_rmse_tensor = torch.sqrt(state["sq_error_sum"] / sample_count)

    per_target_mae = {
        name: float(per_target_mae_tensor[index].item())
        for index, name in enumerate(target_names)
    }
    per_target_rmse = {
        name: float(per_target_rmse_tensor[index].item())
        for index, name in enumerate(target_names)
    }
    mean_mae = float(per_target_mae_tensor.mean().item())
    mean_rmse = float(per_target_rmse_tensor.mean().item())
    loss = float(state["loss_total"] / sample_count)

    return {
        "loss": loss,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "per_target_mae": per_target_mae,
        "per_target_rmse": per_target_rmse,
    }


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    model_config: BodyMModelConfig,
    log_every_n_steps: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for step_index, batch in enumerate(dataloader, start=1):
        moved_batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_model_from_batch(model, moved_batch, model_config)
        loss = loss_fn(predictions, moved_batch["targets"])
        loss.backward()
        optimizer.step()

        batch_size = int(moved_batch["targets"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

        if log_every_n_steps > 0 and step_index % log_every_n_steps == 0:
            print(f"  step={step_index} train_loss={float(loss.detach().cpu()):.6f}")

    if total_samples == 0:
        raise TrainingPipelineError("Training dataloader yielded zero samples.")

    return {"loss": float(total_loss / total_samples)}


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[dict[str, Any]],
    loss_fn: nn.Module,
    device: torch.device,
    model_config: BodyMModelConfig,
    target_names: tuple[str, ...],
    collect_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    metric_state = _build_metric_state(target_names)
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            moved_batch = move_batch_to_device(batch, device)
            predictions = forward_model_from_batch(model, moved_batch, model_config)
            loss = loss_fn(predictions, moved_batch["targets"])
            _update_metric_state(metric_state, loss, predictions, moved_batch["targets"])

            if collect_predictions:
                prediction_cpu = predictions.detach().cpu()
                target_cpu = moved_batch["targets"].detach().cpu()
                batch_size = int(target_cpu.shape[0])
                for row_index in range(batch_size):
                    row = {
                        "split": batch["split"][row_index],
                        "subject_id": batch["subject_id"][row_index],
                        "subject_key": batch["subject_key"][row_index],
                        "photo_id": batch["photo_id"][row_index],
                    }
                    for target_index, target_name in enumerate(target_names):
                        row[f"pred_{target_name}"] = float(prediction_cpu[row_index, target_index].item())
                        row[f"target_{target_name}"] = float(target_cpu[row_index, target_index].item())
                    prediction_rows.append(row)

    summary = _finalize_metric_state(metric_state)
    if collect_predictions:
        summary["prediction_rows"] = prediction_rows
    return summary


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: BodyMExperimentConfig,
    target_names: tuple[str, ...],
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": asdict(config.model),
        "data_view_mode": config.data.view_mode,
        "target_names": list(target_names),
        "metrics": metrics,
    }
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
    except OSError as exc:
        raise TrainingPipelineError(f"Failed to save checkpoint: {checkpoint_path}") from exc


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    resolved_path = resolve_repo_path(checkpoint_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")
    try:
        checkpoint = torch.load(resolved_path, map_location=device)
    except OSError as exc:
        raise TrainingPipelineError(f"Failed to load checkpoint: {resolved_path}") from exc
    if not isinstance(checkpoint, dict):
        raise TrainingPipelineError("Checkpoint payload is invalid.")
    return checkpoint


def _ensure_target_alignment(
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    model_config: BodyMModelConfig,
) -> tuple[str, ...]:
    train_targets = train_loader.dataset.target_columns
    val_targets = val_loader.dataset.target_columns
    if train_targets != val_targets:
        raise TrainingPipelineError("Train and validation target columns do not match.")
    if len(train_targets) != model_config.num_targets:
        raise TrainingPipelineError(
            "Configured model.num_targets does not match dataloader target count."
        )
    return train_targets


def format_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_metrics: dict[str, Any],
) -> str:
    return (
        f"Epoch {epoch}/{total_epochs} | "
        f"train_loss={train_loss:.6f} | "
        f"val_loss={val_metrics['loss']:.6f} | "
        f"val_mean_mae={val_metrics['mean_mae']:.6f} | "
        f"val_mean_rmse={val_metrics['mean_rmse']:.6f}"
    )


def train_model(
    config: BodyMExperimentConfig,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    train_loader, val_loader = build_dataloaders(config)
    target_names = _ensure_target_alignment(train_loader, val_loader, config.model)

    model = build_bodym_model(config.model).to(device)
    loss_fn = build_regression_loss(config.training.loss_name)
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    run_dir = config.output.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(experiment_config_to_dict(config), run_dir / "config_resolved.yaml")

    history: list[dict[str, Any]] = []
    best_epoch_record: dict[str, Any] | None = None
    best_mean_mae = float("inf")
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    for epoch in range(1, config.training.epochs + 1):
        train_summary = run_training_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            model_config=config.model,
            log_every_n_steps=config.training.log_every_n_steps,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            model_config=config.model,
            target_names=target_names,
            collect_predictions=False,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_summary["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mean_mae": float(val_metrics["mean_mae"]),
            "val_mean_rmse": float(val_metrics["mean_rmse"]),
            "per_target_mae": val_metrics["per_target_mae"],
            "per_target_rmse": val_metrics["per_target_rmse"],
        }
        history.append(epoch_record)
        print(
            format_epoch_summary(
                epoch=epoch,
                total_epochs=config.training.epochs,
                train_loss=epoch_record["train_loss"],
                val_metrics=val_metrics,
            )
        )

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            target_names=target_names,
            metrics=epoch_record,
        )

        if epoch_record["val_mean_mae"] < best_mean_mae:
            best_mean_mae = epoch_record["val_mean_mae"]
            best_epoch_record = dict(epoch_record)
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                target_names=target_names,
                metrics=epoch_record,
            )

    if best_epoch_record is None:
        raise TrainingPipelineError("Training completed without producing a best checkpoint.")

    history_payload = {
        "history": history,
        "best_epoch": int(best_epoch_record["epoch"]),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "target_names": list(target_names),
    }
    best_metrics_payload = {
        **best_epoch_record,
        "checkpoint_path": str(best_checkpoint_path),
    }
    save_json(history_payload, run_dir / "history.json")
    save_json(best_metrics_payload, run_dir / "best_metrics.json")

    return {
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "history": history,
        "best_metrics": best_metrics_payload,
        "target_names": list(target_names),
        "device": str(device),
        "config_path": str(resolve_repo_path(config_path)) if config_path is not None else None,
    }


def evaluate_checkpoint(
    config: BodyMExperimentConfig,
    checkpoint_path: str | Path,
    manifest_path_override: str | Path | None = None,
    split_override: str | None = None,
) -> dict[str, Any]:
    set_global_seed(config.training.seed)
    device = resolve_device(config.training.device)
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    checkpoint_model_config = BodyMModelConfig(**checkpoint["model_config"])
    target_names = tuple(checkpoint.get("target_names", []))
    if not target_names:
        raise TrainingPipelineError("Checkpoint is missing target_names.")

    resolved_manifest_path = (
        resolve_repo_path(manifest_path_override)
        if manifest_path_override is not None
        else config.data.val_manifest_path
    )
    resolved_split = split_override or config.data.val_split

    if (
        config.data.resize is not None
        or config.data.normalize_mean is not None
        or config.data.normalize_std is not None
    ):
        transform = build_bodym_transform(
            BodyMTransformConfig(
                resize=config.data.resize,
                resize_mode=config.data.resize_mode,
                normalize_mean=config.data.normalize_mean,
                normalize_std=config.data.normalize_std,
            )
        )
    else:
        transform = None
    dataloader = create_bodym_dataloader(
        manifest_path=resolved_manifest_path,
        split=resolved_split,
        view_mode=config.data.view_mode,
        transform=transform,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    if dataloader.dataset.target_columns != target_names:
        raise TrainingPipelineError("Evaluation dataloader target columns do not match the checkpoint.")

    model = build_bodym_model(
        checkpoint_model_config,
        initialize_pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = build_regression_loss(config.training.loss_name)
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        model_config=checkpoint_model_config,
        target_names=target_names,
        collect_predictions=config.output.save_predictions_on_eval,
    )

    output_dir = resolve_repo_path(checkpoint_path).resolve().parent
    summary_path = output_dir / f"evaluation_summary_{resolved_split}.json"
    predictions_path: Path | None = None

    if config.output.save_predictions_on_eval and "prediction_rows" in metrics:
        predictions_path = output_dir / f"evaluation_predictions_{resolved_split}.csv"
        import pandas as pd

        try:
            pd.DataFrame(metrics["prediction_rows"]).to_csv(predictions_path, index=False)
        except OSError as exc:
            raise TrainingPipelineError(
                f"Failed to write evaluation predictions CSV: {predictions_path}"
            ) from exc

    summary_payload = {
        "checkpoint_path": str(resolve_repo_path(checkpoint_path)),
        "manifest_path": str(resolved_manifest_path),
        "split": resolved_split,
        "epoch": int(checkpoint["epoch"]),
        "loss": float(metrics["loss"]),
        "mean_mae": float(metrics["mean_mae"]),
        "mean_rmse": float(metrics["mean_rmse"]),
        "per_target_mae": metrics["per_target_mae"],
        "per_target_rmse": metrics["per_target_rmse"],
        "prediction_output_path": str(predictions_path) if predictions_path is not None else None,
    }
    save_json(summary_payload, summary_path)
    return summary_payload


__all__ = [
    "BodyMDataConfig",
    "BodyMExperimentConfig",
    "BodyMOutputConfig",
    "BodyMTrainingConfig",
    "TrainingPipelineError",
    "evaluate_checkpoint",
    "experiment_config_to_dict",
    "format_epoch_summary",
    "load_experiment_config",
    "resolve_device",
    "save_json",
    "save_yaml",
    "set_global_seed",
    "train_model",
]
