from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("pandas is required to use scripts.bodym_dataset.") from exc

try:
    import torch
    import torch.nn.functional as torch_f
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError("torch is required to use scripts.bodym_dataset.") from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = Path("data/interim/bodym_manifest.csv")
ViewMode = Literal["paired", "mask", "mask_left"]
TensorTransform = Callable[[Tensor], Tensor]

MANIFEST_BASE_COLUMNS: tuple[str, ...] = (
    "split",
    "subject_id",
    "subject_key",
    "photo_id",
    "mask_path",
    "mask_left_path",
)
MANIFEST_HWG_COLUMNS: tuple[str, ...] = (
    "hwg_gender",
    "hwg_height_cm",
    "hwg_weight_kg",
)
MANIFEST_MEASUREMENT_COLUMNS: tuple[str, ...] = (
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
)
MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = (
    MANIFEST_BASE_COLUMNS + MANIFEST_HWG_COLUMNS + MANIFEST_MEASUREMENT_COLUMNS
)
MANIFEST_NUMERIC_COLUMNS: tuple[str, ...] = (
    "hwg_height_cm",
    "hwg_weight_kg",
) + MANIFEST_MEASUREMENT_COLUMNS
VIEW_TO_PATH_COLUMN: dict[str, str] = {
    "mask": "mask_path",
    "mask_left": "mask_left_path",
}
VIEW_MODE_TO_NAMES: dict[str, tuple[str, ...]] = {
    "paired": ("mask", "mask_left"),
    "mask": ("mask",),
    "mask_left": ("mask_left",),
}


class BodyMDataError(RuntimeError):
    """Raised when the BodyM manifest or dataset assets are invalid."""


@dataclass(frozen=True)
class BodyMTransformConfig:
    """Configuration for deterministic mask preprocessing."""

    resize: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.resize is None:
            return
        if len(self.resize) != 2:
            raise ValueError("resize must be a (height, width) tuple.")
        height, width = self.resize
        if height <= 0 or width <= 0:
            raise ValueError("resize values must be positive integers.")


def build_bodym_transform(
    config: BodyMTransformConfig | None = None,
) -> TensorTransform:
    """Build a simple deterministic transform for grayscale BodyM masks."""

    resolved_config = config or BodyMTransformConfig()

    def transform(image: Tensor) -> Tensor:
        if not isinstance(image, torch.Tensor):
            raise BodyMDataError("Transform input must be a torch.Tensor.")

        output = image.to(dtype=torch.float32)
        if output.ndim != 3 or output.shape[0] != 1:
            raise BodyMDataError(
                "BodyM transforms expect a grayscale tensor with shape [1, H, W]."
            )

        if resolved_config.resize is not None:
            output = torch_f.interpolate(
                output.unsqueeze(0),
                size=resolved_config.resize,
                mode="nearest",
            ).squeeze(0)

        return output.contiguous()

    return transform


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _resolve_manifest_path(manifest_path: str | Path) -> Path:
    return _resolve_repo_path(manifest_path)


def _load_manifest_frame(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest path is not a file: {manifest_path}")

    try:
        frame = pd.read_csv(
            manifest_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except OSError as exc:
        raise BodyMDataError(f"Failed to read manifest file: {manifest_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise BodyMDataError(f"Manifest file is empty: {manifest_path}") from exc
    except pd.errors.ParserError as exc:
        raise BodyMDataError(f"Failed to parse manifest file: {manifest_path}") from exc

    frame.columns = [str(column).strip() for column in frame.columns.tolist()]
    missing_columns = [
        column for column in MANIFEST_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise BodyMDataError(
            f"Manifest is missing required columns: {missing_columns}"
        )

    frame = frame.loc[:, list(MANIFEST_REQUIRED_COLUMNS)].copy()
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else ""
        )

    empty_counts: list[str] = []
    for column in MANIFEST_REQUIRED_COLUMNS:
        empty_count = int((frame[column] == "").sum())
        if empty_count > 0:
            empty_counts.append(f"{column}={empty_count}")

    if empty_counts:
        raise BodyMDataError(
            "Manifest contains empty required values: " + ", ".join(empty_counts)
        )

    for column in MANIFEST_NUMERIC_COLUMNS:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        invalid_count = int(numeric_values.isna().sum())
        if invalid_count > 0:
            raise BodyMDataError(
                f"Manifest column '{column}' contains non-numeric values: {invalid_count}"
            )
        frame[column] = numeric_values.astype("float32")

    return frame


def _resolve_view_names(view_mode: ViewMode) -> tuple[str, ...]:
    if view_mode not in VIEW_MODE_TO_NAMES:
        raise ValueError("view_mode must be one of 'paired', 'mask', or 'mask_left'.")
    return VIEW_MODE_TO_NAMES[view_mode]


def _resolve_target_columns(
    frame: pd.DataFrame,
    target_columns: Sequence[str] | None,
) -> tuple[str, ...]:
    if target_columns is None:
        resolved = tuple(
            column for column in frame.columns if column.startswith("measurement_")
        )
    else:
        resolved = tuple(target_columns)

    if not resolved:
        raise ValueError("At least one target column must be selected.")

    missing_columns = [column for column in resolved if column not in frame.columns]
    if missing_columns:
        raise BodyMDataError(
            f"Target columns are missing from the manifest: {missing_columns}"
        )

    non_numeric = [
        column
        for column in resolved
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise BodyMDataError(f"Target columns must be numeric: {non_numeric}")

    return resolved


def _read_grayscale_image(image_path: Path) -> Tensor:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    try:
        from torchvision.io import ImageReadMode, read_image
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to load BodyM image tensors."
        ) from exc

    try:
        image = read_image(str(image_path), mode=ImageReadMode.GRAY)
    except RuntimeError as exc:
        raise BodyMDataError(f"Failed to read BodyM image: {image_path}") from exc

    image = image.to(dtype=torch.float32) / 255.0
    if image.ndim != 3 or image.shape[0] != 1:
        raise BodyMDataError(
            f"Expected a grayscale tensor with shape [1, H, W] for image: {image_path}"
        )
    return image.contiguous()


class BodyMManifestDataset(Dataset[dict[str, Any]]):
    """Manifest-backed dataset for paired or single-view BodyM silhouettes."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        split: str = "train",
        view_mode: ViewMode = "paired",
        target_columns: Sequence[str] | None = None,
        transform: TensorTransform | None = None,
    ) -> None:
        self._view_mode = view_mode
        self._view_names = _resolve_view_names(view_mode)
        self._manifest_path = _resolve_manifest_path(manifest_path)
        self._repo_root = REPO_ROOT

        manifest_frame = _load_manifest_frame(self._manifest_path)
        split_frame = manifest_frame.loc[manifest_frame["split"] == split].copy()
        if split_frame.empty:
            raise ValueError(f"No manifest rows found for split '{split}'.")

        self._split = split
        self._target_columns = _resolve_target_columns(split_frame, target_columns)
        self._transform = transform or build_bodym_transform()
        self._frame = split_frame.reset_index(drop=True)
        self._validate_active_paths()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def split(self) -> str:
        return self._split

    @property
    def view_names(self) -> tuple[str, ...]:
        return self._view_names

    @property
    def target_columns(self) -> tuple[str, ...]:
        return self._target_columns

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._frame.copy()

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._frame.iloc[index]
        views = {
            view_name: self._load_view_tensor(str(row[VIEW_TO_PATH_COLUMN[view_name]]))
            for view_name in self._view_names
        }
        metadata = {
            "hwg_gender": str(row["hwg_gender"]),
            "hwg_height_cm": float(row["hwg_height_cm"]),
            "hwg_weight_kg": float(row["hwg_weight_kg"]),
        }
        targets = torch.tensor(
            [float(row[column]) for column in self._target_columns],
            dtype=torch.float32,
        )
        return {
            "split": str(row["split"]),
            "subject_id": str(row["subject_id"]),
            "subject_key": str(row["subject_key"]),
            "photo_id": str(row["photo_id"]),
            "metadata": metadata,
            "views": views,
            "targets": targets,
        }

    def _validate_active_paths(self) -> None:
        for view_name in self._view_names:
            path_column = VIEW_TO_PATH_COLUMN[view_name]
            missing_examples: list[str] = []
            for relative_path in self._frame[path_column].tolist():
                image_path = _resolve_repo_path(str(relative_path))
                if not image_path.is_file():
                    missing_examples.append(str(relative_path))
                if len(missing_examples) >= 5:
                    break

            if missing_examples:
                raise FileNotFoundError(
                    f"Missing image file(s) for column '{path_column}' in split "
                    f"'{self._split}'. Examples: {missing_examples}"
                )

    def _load_view_tensor(self, relative_path: str) -> Tensor:
        image_path = _resolve_repo_path(relative_path)
        image = _read_grayscale_image(image_path)
        transformed = self._transform(image)
        if not isinstance(transformed, torch.Tensor):
            raise BodyMDataError("Transforms must return a torch.Tensor.")

        transformed = transformed.to(dtype=torch.float32)
        if transformed.ndim != 3 or transformed.shape[0] != 1:
            raise BodyMDataError(
                "Transforms must return grayscale tensors with shape [1, H, W]."
            )
        return transformed.contiguous()


def create_bodym_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    split: str = "train",
    view_mode: ViewMode = "paired",
    target_columns: Sequence[str] | None = None,
    transform: TensorTransform | None = None,
) -> BodyMManifestDataset:
    return BodyMManifestDataset(
        manifest_path=manifest_path,
        split=split,
        view_mode=view_mode,
        target_columns=target_columns,
        transform=transform,
    )


def create_bodym_dataloader(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    split: str = "train",
    view_mode: ViewMode = "paired",
    target_columns: Sequence[str] | None = None,
    transform: TensorTransform | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    dataset = create_bodym_dataset(
        manifest_path=manifest_path,
        split=split,
        view_mode=view_mode,
        target_columns=target_columns,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


__all__ = [
    "BodyMDataError",
    "BodyMManifestDataset",
    "BodyMTransformConfig",
    "ViewMode",
    "build_bodym_transform",
    "create_bodym_dataloader",
    "create_bodym_dataset",
]
