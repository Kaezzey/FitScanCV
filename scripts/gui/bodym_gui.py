from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

# Ensure top-level repo is on sys.path so `import scripts.*` works
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("pandas is required to use scripts.gui.bodym_gui.") from exc

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk
except ImportError as exc:
    raise ImportError("Pillow is required to use scripts.gui.bodym_gui.") from exc

from scripts.bodym_dataset import (
    MANIFEST_BASE_COLUMNS,
    MANIFEST_HWG_COLUMNS,
    MANIFEST_MEASUREMENT_COLUMNS,
)
from scripts.inference.bodym_inference import (
    load_inference_service,
    resolve_repo_path,
)

DEFAULT_CHECKPOINT_PATH = Path("outputs/bodym_resnet18/best.pt")
DEFAULT_VAL_MANIFEST_PATH = Path("data/interim/bodym_training_val.csv")
DEFAULT_FULL_MANIFEST_PATH = Path("data/interim/bodym_manifest.csv")
DEFAULT_START_SPLIT = "val"
VALID_SPLIT_FILTERS: tuple[str, ...] = ("All", "val", "testA", "testB")
DATA_SPLIT_ORDER: tuple[str, ...] = ("val", "testA", "testB")
EXPLORER_REQUIRED_COLUMNS: tuple[str, ...] = (
    MANIFEST_BASE_COLUMNS + MANIFEST_HWG_COLUMNS + MANIFEST_MEASUREMENT_COLUMNS
)
GENX_SOFT_CLUB_THEME: dict[str, Any] = {
    "title": "FitScanCV // BodyM Accuracy Explorer",
    "brand_title": "FITSCAN CV",
    "brand_subtitle": "Body Measurement Vision // Validation + Holdout Workbench",
    "background": "#dff2fb",
    "panel": "#f7fcff",
    "panel_alt": "#edf8ff",
    "panel_deep": "#d7eefb",
    "glass_highlight": "#ffffff",
    "glass_shadow": "#c6e5f2",
    "border": "#a9d5eb",
    "accent": "#7bd7ff",
    "accent_soft": "#ccf4ff",
    "accent_deep": "#2d7fa8",
    "divider": "#90defb",
    "text_primary": "#12334a",
    "text_muted": "#54788f",
    "text_inverse": "#f7fdff",
    "preview_background": "#e7f5fd",
    "preview_placeholder": "#8db6ca",
    "hero_background": "#d8eef8",
    "hero_glow": "#fbffff",
    "hero_band": "#caefff",
    "hero_outline": "#a6dff6",
    "hero_microcopy": "#5f8fa6",
}


class BodyMExplorerError(RuntimeError):
    """Raised when the BodyM desktop explorer inputs or state are invalid."""


@dataclass(frozen=True)
class BodyMExplorerConfig:
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH
    config_path: str | Path | None = None
    device: str = "auto"
    val_manifest_path: str | Path = DEFAULT_VAL_MANIFEST_PATH
    full_manifest_path: str | Path = DEFAULT_FULL_MANIFEST_PATH
    start_split: str = DEFAULT_START_SPLIT
    start_photo_id: str | None = None


@dataclass(frozen=True)
class BodyMExplorerRow:
    split: str
    subject_id: str
    subject_key: str
    photo_id: str
    mask_path: str
    mask_left_path: str
    hwg_gender: str
    hwg_height_cm: float
    hwg_weight_kg: float
    measurements: dict[str, float]
    source_manifest_path: str


@dataclass(frozen=True)
class BodyMMetricComparisonRow:
    target_name: str
    ground_truth: float
    prediction: float
    signed_error: float
    absolute_error: float


@dataclass(frozen=True)
class BodyMSamplePredictionResult:
    row: BodyMExplorerRow
    predictions: dict[str, float]
    metric_rows: tuple[BodyMMetricComparisonRow, ...]
    sample_mean_absolute_error: float
    sample_rmse: float


def _normalize_manifest_frame(
    manifest_path: str | Path,
    allowed_splits: set[str],
) -> pd.DataFrame:
    resolved_manifest_path = resolve_repo_path(manifest_path)
    if not resolved_manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {resolved_manifest_path}")

    try:
        frame = pd.read_csv(
            resolved_manifest_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except OSError as exc:
        raise BodyMExplorerError(
            f"Failed to read explorer manifest: {resolved_manifest_path}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise BodyMExplorerError(
            f"Explorer manifest is empty: {resolved_manifest_path}"
        ) from exc
    except pd.errors.ParserError as exc:
        raise BodyMExplorerError(
            f"Failed to parse explorer manifest: {resolved_manifest_path}"
        ) from exc

    frame.columns = [str(column).strip() for column in frame.columns.tolist()]
    missing_columns = [
        column for column in EXPLORER_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise BodyMExplorerError(
            f"Explorer manifest is missing required columns: {missing_columns}"
        )

    frame = frame.loc[:, list(EXPLORER_REQUIRED_COLUMNS)].copy()
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else ""
        )

    filtered_frame = frame.loc[frame["split"].isin(allowed_splits)].copy()
    if filtered_frame.empty:
        raise BodyMExplorerError(
            f"Explorer manifest {resolved_manifest_path} has no rows for splits "
            f"{sorted(allowed_splits)}."
        )

    empty_columns = [
        column
        for column in EXPLORER_REQUIRED_COLUMNS
        if int((filtered_frame[column] == "").sum()) > 0
    ]
    if empty_columns:
        raise BodyMExplorerError(
            f"Explorer manifest contains empty required values for columns: {empty_columns}"
        )

    numeric_columns = MANIFEST_HWG_COLUMNS[1:] + MANIFEST_MEASUREMENT_COLUMNS
    for column in numeric_columns:
        numeric_values = pd.to_numeric(filtered_frame[column], errors="coerce")
        invalid_count = int(numeric_values.isna().sum())
        if invalid_count > 0:
            raise BodyMExplorerError(
                f"Explorer manifest column '{column}' contains non-numeric values: "
                f"{invalid_count}"
            )
        filtered_frame[column] = numeric_values.astype("float32")

    filtered_frame["source_manifest_path"] = str(resolved_manifest_path)
    return filtered_frame


def _frame_to_rows(frame: pd.DataFrame) -> list[BodyMExplorerRow]:
    rows: list[BodyMExplorerRow] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            BodyMExplorerRow(
                split=str(row["split"]),
                subject_id=str(row["subject_id"]),
                subject_key=str(row["subject_key"]),
                photo_id=str(row["photo_id"]),
                mask_path=str(row["mask_path"]),
                mask_left_path=str(row["mask_left_path"]),
                hwg_gender=str(row["hwg_gender"]),
                hwg_height_cm=float(row["hwg_height_cm"]),
                hwg_weight_kg=float(row["hwg_weight_kg"]),
                measurements={
                    column: float(row[column]) for column in MANIFEST_MEASUREMENT_COLUMNS
                },
                source_manifest_path=str(row["source_manifest_path"]),
            )
        )
    return rows


def load_explorer_rows(
    val_manifest_path: str | Path = DEFAULT_VAL_MANIFEST_PATH,
    full_manifest_path: str | Path = DEFAULT_FULL_MANIFEST_PATH,
) -> list[BodyMExplorerRow]:
    val_rows = _frame_to_rows(
        _normalize_manifest_frame(val_manifest_path, allowed_splits={"val"})
    )
    holdout_rows = _frame_to_rows(
        _normalize_manifest_frame(full_manifest_path, allowed_splits={"testA", "testB"})
    )
    all_rows = val_rows + holdout_rows
    split_order = {split: index for index, split in enumerate(DATA_SPLIT_ORDER)}
    all_rows.sort(
        key=lambda row: (
            split_order.get(row.split, len(split_order)),
            row.subject_key,
            row.photo_id,
        )
    )
    if not all_rows:
        raise BodyMExplorerError("Explorer dataset has no rows after loading manifests.")
    return all_rows


def build_sample_prediction_result(
    row: BodyMExplorerRow,
    predictions: dict[str, Any],
) -> BodyMSamplePredictionResult:
    missing_targets = [
        target_name
        for target_name in MANIFEST_MEASUREMENT_COLUMNS
        if target_name not in predictions
    ]
    if missing_targets:
        raise BodyMExplorerError(
            f"Prediction payload is missing targets: {missing_targets}"
        )

    metric_rows: list[BodyMMetricComparisonRow] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    normalized_predictions: dict[str, float] = {}

    for target_name in MANIFEST_MEASUREMENT_COLUMNS:
        ground_truth = float(row.measurements[target_name])
        prediction = float(predictions[target_name])
        signed_error = prediction - ground_truth
        absolute_error = abs(signed_error)
        normalized_predictions[target_name] = prediction
        absolute_errors.append(absolute_error)
        squared_errors.append(signed_error * signed_error)
        metric_rows.append(
            BodyMMetricComparisonRow(
                target_name=target_name,
                ground_truth=ground_truth,
                prediction=prediction,
                signed_error=signed_error,
                absolute_error=absolute_error,
            )
        )

    sample_mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    sample_rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    return BodyMSamplePredictionResult(
        row=row,
        predictions=normalized_predictions,
        metric_rows=tuple(metric_rows),
        sample_mean_absolute_error=sample_mean_absolute_error,
        sample_rmse=sample_rmse,
    )


class BodyMAccuracyExplorerController:
    """Pure explorer state and prediction orchestration for the desktop app."""

    def __init__(
        self,
        rows: list[BodyMExplorerRow],
        inference_service: Any,
        *,
        start_split: str = DEFAULT_START_SPLIT,
        start_photo_id: str | None = None,
    ) -> None:
        if not rows:
            raise BodyMExplorerError("Explorer controller requires at least one row.")
        if start_split not in VALID_SPLIT_FILTERS[1:]:
            raise ValueError(
                "start_split must be one of 'val', 'testA', or 'testB'."
            )

        target_names = tuple(getattr(inference_service, "target_names", ()))
        if target_names and target_names != MANIFEST_MEASUREMENT_COLUMNS:
            raise BodyMExplorerError(
                "Inference service target_names do not match the explorer manifest schema."
            )

        self.inference_service = inference_service
        self._all_rows = list(rows)
        self._prediction_cache: dict[tuple[str, str], BodyMSamplePredictionResult] = {}
        self._prediction_lock = threading.Lock()
        self._split_filter = start_split
        self._search_query = ""
        self._filtered_rows: list[BodyMExplorerRow] = []
        self._current_index: int | None = None
        self._apply_filters(preferred_photo_id=start_photo_id)

    @property
    def all_rows(self) -> tuple[BodyMExplorerRow, ...]:
        return tuple(self._all_rows)

    @property
    def filtered_rows(self) -> tuple[BodyMExplorerRow, ...]:
        return tuple(self._filtered_rows)

    @property
    def split_filter(self) -> str:
        return self._split_filter

    @property
    def search_query(self) -> str:
        return self._search_query

    @property
    def current_row(self) -> BodyMExplorerRow | None:
        if self._current_index is None:
            return None
        return self._filtered_rows[self._current_index]

    @property
    def current_index(self) -> int | None:
        return self._current_index

    @property
    def prediction_cache_size(self) -> int:
        return len(self._prediction_cache)

    def current_position(self) -> tuple[int, int]:
        if self._current_index is None:
            return 0, len(self._filtered_rows)
        return self._current_index + 1, len(self._filtered_rows)

    def set_split_filter(self, split_filter: str) -> None:
        if split_filter not in VALID_SPLIT_FILTERS:
            raise ValueError(
                "split_filter must be one of 'All', 'val', 'testA', or 'testB'."
            )
        preserve_photo_id = self.current_row.photo_id if self.current_row is not None else None
        self._split_filter = split_filter
        self._apply_filters(preferred_photo_id=preserve_photo_id)

    def set_search_query(self, search_query: str) -> None:
        preserve_photo_id = self.current_row.photo_id if self.current_row is not None else None
        self._search_query = search_query.strip()
        self._apply_filters(preferred_photo_id=preserve_photo_id)

    def select_index(self, index: int) -> BodyMExplorerRow:
        if not self._filtered_rows:
            raise BodyMExplorerError("No explorer rows are available for selection.")
        if index < 0 or index >= len(self._filtered_rows):
            raise IndexError(f"Explorer row index is out of range: {index}")
        self._current_index = index
        return self._filtered_rows[index]

    def select_photo_id(self, photo_id: str) -> bool:
        for index, row in enumerate(self._filtered_rows):
            if row.photo_id == photo_id:
                self._current_index = index
                return True
        return False

    def next_row(self) -> BodyMExplorerRow:
        if not self._filtered_rows:
            raise BodyMExplorerError("No explorer rows are available for navigation.")
        if self._current_index is None:
            self._current_index = 0
        else:
            self._current_index = min(self._current_index + 1, len(self._filtered_rows) - 1)
        return self._filtered_rows[self._current_index]

    def previous_row(self) -> BodyMExplorerRow:
        if not self._filtered_rows:
            raise BodyMExplorerError("No explorer rows are available for navigation.")
        if self._current_index is None:
            self._current_index = 0
        else:
            self._current_index = max(self._current_index - 1, 0)
        return self._filtered_rows[self._current_index]

    def jump_to_position(self, position: int) -> BodyMExplorerRow:
        if position <= 0:
            raise ValueError("Jump position must be a positive 1-based index.")
        return self.select_index(position - 1)

    def predict_current_row(self) -> BodyMSamplePredictionResult:
        row = self.current_row
        if row is None:
            raise BodyMExplorerError("No explorer row is selected for prediction.")
        return self.predict_row(row)

    def predict_row(self, row: BodyMExplorerRow) -> BodyMSamplePredictionResult:
        checkpoint_path = getattr(self.inference_service, "checkpoint_path", "unknown")
        cache_key = (str(checkpoint_path), row.photo_id)

        with self._prediction_lock:
            cached_result = self._prediction_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            output = self.inference_service.predict_from_paths(
                front_image_path=row.mask_path,
                side_image_path=row.mask_left_path,
                hwg_gender=row.hwg_gender,
                hwg_height_cm=row.hwg_height_cm,
                hwg_weight_kg=row.hwg_weight_kg,
            )
            result = build_sample_prediction_result(row, output["predictions"])
            self._prediction_cache[cache_key] = result
            return result

    def _apply_filters(self, preferred_photo_id: str | None = None) -> None:
        search_query = self._search_query.lower()
        filtered_rows = [
            row
            for row in self._all_rows
            if self._matches_filters(row, search_query)
        ]
        self._filtered_rows = filtered_rows

        if not filtered_rows:
            self._current_index = None
            return

        if preferred_photo_id is not None:
            for index, row in enumerate(filtered_rows):
                if row.photo_id == preferred_photo_id:
                    self._current_index = index
                    return

        self._current_index = 0

    def _matches_filters(self, row: BodyMExplorerRow, search_query: str) -> bool:
        if self._split_filter != "All" and row.split != self._split_filter:
            return False
        if not search_query:
            return True
        haystack = " ".join((row.photo_id, row.subject_key, row.subject_id, row.split)).lower()
        return search_query in haystack


class BodyMAccuracyExplorerApp:
    """Tkinter desktop app for browsing BodyM validation and holdout predictions."""

    def __init__(
        self,
        root: tk.Tk,
        controller: BodyMAccuracyExplorerController,
        *,
        preview_size: tuple[int, int] = (320, 430),
        auto_predict: bool = True,
    ) -> None:
        self.root = root
        self.controller = controller
        self.preview_size = preview_size
        self.theme = GENX_SOFT_CLUB_THEME
        self.style = ttk.Style(self.root)
        self._prediction_queue: queue.Queue[tuple[int, str, BodyMSamplePredictionResult | None, Exception | None]] = queue.Queue()
        self._prediction_request_id = 0
        self._mask_photo_image: ImageTk.PhotoImage | None = None
        self._mask_left_photo_image: ImageTk.PhotoImage | None = None
        self._hero_photo_image: ImageTk.PhotoImage | None = None
        self._is_closed = False

        self.split_var = tk.StringVar(value=self.controller.split_filter)
        self.search_var = tk.StringVar(value=self.controller.search_query)
        self.jump_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="Select a sample to inspect predictions.")
        self.status_var = tk.StringVar(value="")
        self.browser_microcopy_var = tk.StringVar(value="")
        self.console_microcopy_var = tk.StringVar(value="")
        self.detail_vars = {
            "split": tk.StringVar(value=""),
            "subject_id": tk.StringVar(value=""),
            "subject_key": tk.StringVar(value=""),
            "photo_id": tk.StringVar(value=""),
            "hwg_gender": tk.StringVar(value=""),
            "hwg_height_cm": tk.StringVar(value=""),
            "hwg_weight_kg": tk.StringVar(value=""),
        }
        self._status_message = "Ready"

        self._configure_theme()
        self._build_ui()
        self._populate_row_tree()
        self._render_current_row()
        self._set_status("Ready")
        self.root.after(75, self._poll_prediction_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if auto_predict and self.controller.current_row is not None:
            self.request_prediction_for_current_row()

    def apply_split_filter(self, split_filter: str) -> None:
        self.controller.set_split_filter(split_filter)
        self.split_var.set(self.controller.split_filter)
        self._populate_row_tree()
        self._render_current_row()
        if self.controller.current_row is not None:
            self.request_prediction_for_current_row()
        else:
            self._set_status("No rows match the current split filter.")

    def apply_search_query(self, search_query: str) -> None:
        self.controller.set_search_query(search_query)
        self.search_var.set(self.controller.search_query)
        self._populate_row_tree()
        self._render_current_row()
        if self.controller.current_row is not None:
            self.request_prediction_for_current_row()
        else:
            self._set_status("No rows match the current search query.")

    def go_to_next_row(self) -> None:
        self.controller.next_row()
        self._sync_tree_selection()
        self._render_current_row()
        self.request_prediction_for_current_row()

    def go_to_previous_row(self) -> None:
        self.controller.previous_row()
        self._sync_tree_selection()
        self._render_current_row()
        self.request_prediction_for_current_row()

    def select_photo_id(self, photo_id: str) -> bool:
        found = self.controller.select_photo_id(photo_id)
        if not found:
            return False
        self._sync_tree_selection()
        self._render_current_row()
        self.request_prediction_for_current_row()
        return True

    def request_prediction_for_current_row(self) -> None:
        row = self.controller.current_row
        if row is None:
            self._clear_prediction_results()
            self._set_status("No explorer row is selected.")
            return

        self._prediction_request_id += 1
        request_id = self._prediction_request_id
        self._clear_prediction_results()
        self._set_status(f"Predicting for photo_id={row.photo_id}...")
        worker = threading.Thread(
            target=self._prediction_worker,
            args=(request_id, row),
            daemon=True,
        )
        worker.start()

    def close(self) -> None:
        self._is_closed = True
        self.root.destroy()

    def _configure_theme(self) -> None:
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.configure(bg=self.theme["background"])
        self.root.option_add("*Font", "{Segoe UI} 10")

        self.style.configure(
            "GenX.Root.TFrame",
            background=self.theme["background"],
        )
        self.style.configure(
            "GenX.Panel.TFrame",
            background=self.theme["panel"],
        )
        self.style.configure(
            "GenX.PanelAlt.TFrame",
            background=self.theme["panel_alt"],
        )
        self.style.configure(
            "GenX.Hero.TFrame",
            background=self.theme["panel"],
        )
        self.style.configure(
            "GenX.TLabel",
            background=self.theme["panel"],
            foreground=self.theme["text_primary"],
        )
        self.style.configure(
            "GenX.Root.TLabel",
            background=self.theme["background"],
            foreground=self.theme["text_primary"],
        )
        self.style.configure(
            "GenX.Brand.TLabel",
            background=self.theme["background"],
            foreground=self.theme["text_primary"],
            font=("Segoe UI Semibold", 24),
        )
        self.style.configure(
            "GenX.Subtitle.TLabel",
            background=self.theme["background"],
            foreground=self.theme["text_muted"],
            font=("Consolas", 10, "italic"),
        )
        self.style.configure(
            "GenX.Meta.TLabel",
            background=self.theme["panel"],
            foreground=self.theme["text_muted"],
            font=("Segoe UI", 9),
        )
        self.style.configure(
            "GenX.Summary.TLabel",
            background=self.theme["panel"],
            foreground=self.theme["accent_deep"],
            font=("Segoe UI Semibold", 10),
        )
        self.style.configure(
            "GenX.Console.TLabel",
            background=self.theme["panel_alt"],
            foreground=self.theme["hero_microcopy"],
            font=("Consolas", 9),
        )
        self.style.configure(
            "GenX.RootConsole.TLabel",
            background=self.theme["background"],
            foreground=self.theme["hero_microcopy"],
            font=("Consolas", 9),
        )
        self.style.configure(
            "GenX.TButton",
            background=self.theme["panel"],
            foreground=self.theme["text_primary"],
            borderwidth=1,
            focusthickness=0,
            focuscolor=self.theme["accent_soft"],
            padding=(10, 6),
        )
        self.style.map(
            "GenX.TButton",
            background=[
                ("pressed", self.theme["accent"]),
                ("active", self.theme["accent_soft"]),
            ],
            foreground=[
                ("pressed", self.theme["text_primary"]),
                ("active", self.theme["text_primary"]),
            ],
        )
        self.style.configure(
            "GenX.TEntry",
            fieldbackground=self.theme["panel"],
            foreground=self.theme["text_primary"],
            bordercolor=self.theme["border"],
            lightcolor=self.theme["accent_soft"],
            darkcolor=self.theme["border"],
            insertcolor=self.theme["accent_deep"],
            padding=(8, 6),
        )
        self.style.configure(
            "GenX.TCombobox",
            fieldbackground=self.theme["panel"],
            foreground=self.theme["text_primary"],
            bordercolor=self.theme["border"],
            arrowcolor=self.theme["accent_deep"],
            selectbackground=self.theme["accent_soft"],
            selectforeground=self.theme["text_primary"],
            padding=(6, 4),
        )
        self.style.map(
            "GenX.TCombobox",
            fieldbackground=[("readonly", self.theme["panel"])],
            selectbackground=[("readonly", self.theme["accent_soft"])],
            selectforeground=[("readonly", self.theme["text_primary"])],
        )
        self.style.configure(
            "GenX.TLabelframe",
            background=self.theme["panel"],
            bordercolor=self.theme["border"],
            relief="solid",
            borderwidth=1,
        )
        self.style.configure(
            "GenX.TLabelframe.Label",
            background=self.theme["panel"],
            foreground=self.theme["accent_deep"],
            font=("Segoe UI Semibold", 10),
        )
        self.style.configure(
            "GenX.Treeview",
            background=self.theme["panel"],
            fieldbackground=self.theme["panel"],
            foreground=self.theme["text_primary"],
            bordercolor=self.theme["border"],
            lightcolor=self.theme["panel"],
            darkcolor=self.theme["panel"],
            rowheight=28,
        )
        self.style.map(
            "GenX.Treeview",
            background=[("selected", self.theme["accent_soft"])],
            foreground=[("selected", self.theme["text_primary"])],
        )
        self.style.configure(
            "GenX.Treeview.Heading",
            background=self.theme["panel_deep"],
            foreground=self.theme["text_primary"],
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        self.style.map(
            "GenX.Treeview.Heading",
            background=[("active", self.theme["accent_soft"])],
        )
        self.style.configure(
            "GenX.Status.TLabel",
            background=self.theme["panel_deep"],
            foreground=self.theme["text_primary"],
            font=("Consolas", 9),
            padding=(10, 6),
        )
        self.style.configure(
            "GenX.TPanedwindow",
            background=self.theme["background"],
        )
        self.style.configure(
            "GenX.Vertical.TScrollbar",
            background=self.theme["panel_deep"],
            troughcolor=self.theme["panel_alt"],
            bordercolor=self.theme["border"],
            arrowcolor=self.theme["accent_deep"],
        )

    def _build_ui(self) -> None:
        self.root.title(self.theme["title"])
        self.root.geometry("1680x960")
        self.root.minsize(1320, 760)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top_controls = ttk.Frame(
            self.root,
            padding=(14, 14, 14, 10),
            style="GenX.Root.TFrame",
        )
        top_controls.grid(row=0, column=0, sticky="ew")
        top_controls.columnconfigure(0, weight=1)
        top_controls.columnconfigure(1, weight=1)

        brand_frame = ttk.Frame(top_controls, style="GenX.Root.TFrame")
        brand_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Label(
            brand_frame,
            text=self.theme["brand_title"],
            style="GenX.Brand.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            brand_frame,
            text=self.theme["brand_subtitle"],
            style="GenX.Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            brand_frame,
            text="console // paired silhouettes // soft metrics // local playback",
            style="GenX.RootConsole.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        controls_frame = ttk.Frame(top_controls, style="GenX.Root.TFrame")
        controls_frame.grid(row=1, column=0, sticky="ew")
        controls_frame.columnconfigure(6, weight=1)

        ttk.Button(
            controls_frame,
            text="Previous",
            command=self.go_to_previous_row,
            style="GenX.TButton",
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Button(
            controls_frame,
            text="Next",
            command=self.go_to_next_row,
            style="GenX.TButton",
        ).grid(row=0, column=1, padx=(0, 16), sticky="w")
        ttk.Label(
            controls_frame,
            text="Jump To Index",
            style="GenX.Root.TLabel",
        ).grid(
            row=0, column=2, padx=(0, 6), sticky="w"
        )
        jump_entry = ttk.Entry(
            controls_frame,
            textvariable=self.jump_var,
            width=10,
            style="GenX.TEntry",
        )
        jump_entry.grid(row=0, column=3, padx=(0, 6), sticky="w")
        jump_entry.bind("<Return>", self._on_jump_requested)
        ttk.Button(
            controls_frame,
            text="Go",
            command=lambda: self._jump_to_index(),
            style="GenX.TButton",
        ).grid(row=0, column=4, padx=(0, 16), sticky="w")
        ttk.Label(
            controls_frame,
            text="Search",
            style="GenX.Root.TLabel",
        ).grid(row=0, column=5, padx=(0, 6), sticky="e")
        search_entry = ttk.Entry(
            controls_frame,
            textvariable=self.search_var,
            width=32,
            style="GenX.TEntry",
        )
        search_entry.grid(row=0, column=6, sticky="ew")
        search_entry.bind("<KeyRelease>", self._on_search_changed)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL, style="GenX.TPanedwindow")
        main_pane.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left_shell = self._create_glass_shell(main_pane, self.theme["panel_alt"])
        left_frame = ttk.Frame(left_shell, padding=10, style="GenX.PanelAlt.TFrame")
        left_frame.pack(fill="both", expand=True)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(3, weight=1)
        main_pane.add(left_shell, weight=1)

        ttk.Label(left_frame, text="Split Filter", style="GenX.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        split_box = ttk.Combobox(
            left_frame,
            textvariable=self.split_var,
            state="readonly",
            values=VALID_SPLIT_FILTERS,
            width=12,
            style="GenX.TCombobox",
        )
        split_box.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        split_box.bind("<<ComboboxSelected>>", self._on_split_filter_changed)
        ttk.Label(
            left_frame,
            textvariable=self.browser_microcopy_var,
            style="GenX.Console.TLabel",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 10))

        self.row_tree = ttk.Treeview(
            left_frame,
            columns=("split", "subject_key", "photo_id"),
            show="headings",
            height=20,
            style="GenX.Treeview",
        )
        self.row_tree.heading("split", text="Split")
        self.row_tree.heading("subject_key", text="Subject Key")
        self.row_tree.heading("photo_id", text="Photo ID")
        self.row_tree.column("split", width=70, stretch=False, anchor="center")
        self.row_tree.column("subject_key", width=220, stretch=True)
        self.row_tree.column("photo_id", width=220, stretch=True)
        self.row_tree.grid(row=3, column=0, sticky="nsew")
        self.row_tree.bind("<<TreeviewSelect>>", self._on_tree_selection_changed)
        row_tree_scrollbar = ttk.Scrollbar(
            left_frame,
            orient="vertical",
            command=self.row_tree.yview,
            style="GenX.Vertical.TScrollbar",
        )
        row_tree_scrollbar.grid(row=3, column=1, sticky="ns")
        self.row_tree.configure(yscrollcommand=row_tree_scrollbar.set)

        right_shell = self._create_glass_shell(main_pane, self.theme["panel"])
        right_frame = ttk.Frame(right_shell, padding=10, style="GenX.Panel.TFrame")
        right_frame.pack(fill="both", expand=True)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(6, weight=1)
        main_pane.add(right_shell, weight=3)

        hero_shell = self._create_glass_shell(right_frame, self.theme["panel"])
        hero_shell.grid(row=0, column=0, sticky="ew")
        hero_frame = ttk.Frame(hero_shell, padding=8, style="GenX.Hero.TFrame")
        hero_frame.pack(fill="both", expand=True)
        hero_frame.columnconfigure(0, weight=1)

        self.hero_canvas = tk.Canvas(
            hero_frame,
            height=176,
            bg=self.theme["hero_background"],
            highlightthickness=0,
            bd=0,
        )
        self.hero_canvas.grid(row=0, column=0, sticky="ew")
        self.hero_canvas.bind("<Configure>", self._on_hero_canvas_configure)

        hero_console_frame = ttk.Frame(hero_frame, style="GenX.Hero.TFrame")
        hero_console_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        hero_console_frame.columnconfigure(0, weight=1)
        ttk.Label(
            hero_console_frame,
            textvariable=self.console_microcopy_var,
            style="GenX.Console.TLabel",
        ).grid(row=0, column=0, sticky="ew")

        self._create_soft_divider(right_frame).grid(row=1, column=0, sticky="ew", pady=(12, 12))

        preview_frame = ttk.Frame(right_frame, style="GenX.Panel.TFrame")
        preview_frame.grid(row=2, column=0, sticky="ew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)

        front_shell = self._create_glass_shell(preview_frame, self.theme["panel"])
        front_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        front_frame = ttk.LabelFrame(
            front_shell,
            text="Front Mask",
            padding=8,
            style="GenX.TLabelframe",
        )
        front_frame.pack(fill="both", expand=True)
        self.front_canvas = tk.Canvas(
            front_frame,
            width=self.preview_size[0],
            height=self.preview_size[1],
            bg=self.theme["preview_background"],
            highlightthickness=0,
        )
        self.front_canvas.pack(fill="both", expand=True)

        side_shell = self._create_glass_shell(preview_frame, self.theme["panel"])
        side_shell.grid(row=0, column=1, sticky="nsew")
        side_frame = ttk.LabelFrame(
            side_shell,
            text="Side Mask",
            padding=8,
            style="GenX.TLabelframe",
        )
        side_frame.pack(fill="both", expand=True)
        self.side_canvas = tk.Canvas(
            side_frame,
            width=self.preview_size[0],
            height=self.preview_size[1],
            bg=self.theme["preview_background"],
            highlightthickness=0,
        )
        self.side_canvas.pack(fill="both", expand=True)

        self._create_soft_divider(right_frame).grid(row=3, column=0, sticky="ew", pady=(12, 12))

        details_shell = self._create_glass_shell(right_frame, self.theme["panel"])
        details_shell.grid(row=4, column=0, sticky="ew")
        details_frame = ttk.LabelFrame(
            details_shell,
            text="Sample Details",
            padding=8,
            style="GenX.TLabelframe",
        )
        details_frame.pack(fill="both", expand=True)
        for column_index in range(4):
            details_frame.columnconfigure(column_index, weight=1)

        detail_pairs = (
            ("Split", "split"),
            ("Subject ID", "subject_id"),
            ("Subject Key", "subject_key"),
            ("Photo ID", "photo_id"),
            ("Gender", "hwg_gender"),
            ("Height (cm)", "hwg_height_cm"),
            ("Weight (kg)", "hwg_weight_kg"),
        )
        for index, (label_text, key) in enumerate(detail_pairs):
            row_index = index // 4
            column_index = (index % 4) * 2
            ttk.Label(details_frame, text=label_text, style="GenX.Meta.TLabel").grid(
                row=row_index,
                column=column_index,
                sticky="w",
                padx=(0, 6),
                pady=2,
            )
            ttk.Label(
                details_frame,
                textvariable=self.detail_vars[key],
                style="GenX.TLabel",
            ).grid(row=row_index, column=column_index + 1, sticky="w", pady=2)

        self._create_soft_divider(right_frame).grid(row=5, column=0, sticky="ew", pady=(12, 12))

        results_shell = self._create_glass_shell(right_frame, self.theme["panel"])
        results_shell.grid(row=6, column=0, sticky="nsew")
        results_frame = ttk.LabelFrame(
            results_shell,
            text="Prediction vs Ground Truth",
            padding=8,
            style="GenX.TLabelframe",
        )
        results_frame.pack(fill="both", expand=True)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("target", "ground_truth", "prediction", "signed_error", "absolute_error"),
            show="headings",
            height=14,
            style="GenX.Treeview",
        )
        self.results_tree.heading("target", text="Target")
        self.results_tree.heading("ground_truth", text="Ground Truth")
        self.results_tree.heading("prediction", text="Prediction")
        self.results_tree.heading("signed_error", text="Signed Error")
        self.results_tree.heading("absolute_error", text="Absolute Error")
        self.results_tree.column("target", width=240, stretch=True)
        self.results_tree.column("ground_truth", width=120, stretch=False, anchor="e")
        self.results_tree.column("prediction", width=120, stretch=False, anchor="e")
        self.results_tree.column("signed_error", width=120, stretch=False, anchor="e")
        self.results_tree.column("absolute_error", width=120, stretch=False, anchor="e")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        results_scrollbar = ttk.Scrollbar(
            results_frame,
            orient="vertical",
            command=self.results_tree.yview,
            style="GenX.Vertical.TScrollbar",
        )
        results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        ttk.Label(
            right_frame,
            textvariable=self.summary_var,
            anchor="w",
            style="GenX.Summary.TLabel",
        ).grid(row=7, column=0, sticky="ew", pady=(10, 0))

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            style="GenX.Status.TLabel",
        ).grid(row=2, column=0, sticky="ew")

    def _create_glass_shell(self, parent: tk.Misc, fill_color: str) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=fill_color,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.theme["glass_highlight"],
            highlightcolor=self.theme["glass_highlight"],
        )

    def _create_soft_divider(self, parent: tk.Misc) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=self.theme["divider"],
            height=2,
            bd=0,
            highlightthickness=0,
        )

    def _on_hero_canvas_configure(self, _: tk.Event[Any]) -> None:
        self._draw_hero_banner()

    def _populate_row_tree(self) -> None:
        current_selection = self.controller.current_row.photo_id if self.controller.current_row else None
        self.row_tree.delete(*self.row_tree.get_children())
        for row in self.controller.filtered_rows:
            self.row_tree.insert(
                "",
                "end",
                iid=row.photo_id,
                values=(row.split, row.subject_key, row.photo_id),
            )
        if current_selection is not None and self.row_tree.exists(current_selection):
            self.row_tree.selection_set(current_selection)
            self.row_tree.focus(current_selection)
            self.row_tree.see(current_selection)
        self._update_browser_microcopy()

    def _render_current_row(self) -> None:
        row = self.controller.current_row
        if row is None:
            for variable in self.detail_vars.values():
                variable.set("")
            self._render_preview_placeholder(self.front_canvas, "No row selected")
            self._render_preview_placeholder(self.side_canvas, "No row selected")
            self._clear_prediction_results()
            self._draw_hero_banner()
            self._update_console_microcopy()
            self._set_status("No explorer rows are available.")
            return

        self.detail_vars["split"].set(row.split)
        self.detail_vars["subject_id"].set(row.subject_id)
        self.detail_vars["subject_key"].set(row.subject_key)
        self.detail_vars["photo_id"].set(row.photo_id)
        self.detail_vars["hwg_gender"].set(row.hwg_gender)
        self.detail_vars["hwg_height_cm"].set(f"{row.hwg_height_cm:.2f}")
        self.detail_vars["hwg_weight_kg"].set(f"{row.hwg_weight_kg:.2f}")
        self._render_preview(self.front_canvas, row.mask_path, is_front=True)
        self._render_preview(self.side_canvas, row.mask_left_path, is_front=False)
        self._draw_hero_banner()
        self._update_console_microcopy()
        self._set_status("Row selected.")

    def _render_preview(self, canvas: tk.Canvas, image_path: str, *, is_front: bool) -> None:
        resolved_path = resolve_repo_path(image_path)
        if not resolved_path.is_file():
            self._render_preview_placeholder(
                canvas,
                f"Missing image\n{resolved_path.name}",
            )
            if is_front:
                self._mask_photo_image = None
            else:
                self._mask_left_photo_image = None
            return

        try:
            image = Image.open(resolved_path).convert("L")
        except OSError:
            self._render_preview_placeholder(canvas, f"Unreadable image\n{resolved_path.name}")
            if is_front:
                self._mask_photo_image = None
            else:
                self._mask_left_photo_image = None
            return

        resized_image = ImageOps.contain(
            image,
            self.preview_size,
            method=_pil_resample_nearest(),
        )
        background = Image.new("RGB", self.preview_size, color=self.theme["preview_background"])
        resized_image_rgb = resized_image.convert("RGB")
        offset_x = (self.preview_size[0] - resized_image_rgb.width) // 2
        offset_y = (self.preview_size[1] - resized_image_rgb.height) // 2
        background.paste(resized_image_rgb, (offset_x, offset_y))
        photo_image = ImageTk.PhotoImage(background)

        canvas.delete("all")
        canvas.create_image(
            self.preview_size[0] // 2,
            self.preview_size[1] // 2,
            image=photo_image,
            anchor="center",
        )
        if is_front:
            self._mask_photo_image = photo_image
        else:
            self._mask_left_photo_image = photo_image

    def _render_preview_placeholder(self, canvas: tk.Canvas, message: str) -> None:
        canvas.delete("all")
        canvas.create_rectangle(
            0,
            0,
            self.preview_size[0],
            self.preview_size[1],
            fill=self.theme["preview_background"],
            outline="",
        )
        canvas.create_text(
            self.preview_size[0] // 2,
            self.preview_size[1] // 2,
            text=message,
            fill=self.theme["preview_placeholder"],
            justify="center",
            font=("Segoe UI", 11, "italic"),
        )

    def _clear_prediction_results(self) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        self.summary_var.set("Waiting for prediction...")

    def _populate_prediction_results(self, result: BodyMSamplePredictionResult) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        for metric_row in result.metric_rows:
            self.results_tree.insert(
                "",
                "end",
                values=(
                    metric_row.target_name,
                    f"{metric_row.ground_truth:.4f}",
                    f"{metric_row.prediction:.4f}",
                    f"{metric_row.signed_error:.4f}",
                    f"{metric_row.absolute_error:.4f}",
                ),
            )
        self.summary_var.set(
            "Sample Mean AE: "
            f"{result.sample_mean_absolute_error:.4f} | "
            f"Sample RMSE: {result.sample_rmse:.4f}"
        )

    def _prediction_worker(self, request_id: int, row: BodyMExplorerRow) -> None:
        try:
            result = self.controller.predict_row(row)
            error: Exception | None = None
        except Exception as exc:  # pragma: no cover - exercised via queue handling
            result = None
            error = exc
        self._prediction_queue.put((request_id, row.photo_id, result, error))

    def _poll_prediction_queue(self) -> None:
        if self._is_closed:
            return

        while True:
            try:
                request_id, photo_id, result, error = self._prediction_queue.get_nowait()
            except queue.Empty:
                break

            current_row = self.controller.current_row
            if request_id != self._prediction_request_id:
                continue
            if current_row is None or current_row.photo_id != photo_id:
                continue

            if error is not None:
                self._clear_prediction_results()
                self.summary_var.set("Prediction failed.")
                self._set_status(f"Error: {error}")
                continue

            assert result is not None
            self._populate_prediction_results(result)
            self._set_status(f"Prediction ready for photo_id={photo_id}.")

        self.root.after(75, self._poll_prediction_queue)

    def _set_status(self, message: str) -> None:
        self._status_message = message
        checkpoint_path = getattr(self.controller.inference_service, "checkpoint_path", "unknown")
        device = getattr(self.controller.inference_service, "device", "unknown")
        current_position, row_count = self.controller.current_position()
        self.status_var.set(
            f"Checkpoint: {checkpoint_path} | "
            f"Device: {device} | "
            f"Row: {current_position}/{row_count} | "
            f"Status: {self._status_message}"
        )
        self._update_browser_microcopy()
        self._update_console_microcopy()

    def _update_browser_microcopy(self) -> None:
        visible = len(self.controller.filtered_rows)
        total = len(self.controller.all_rows)
        split_filter = self.controller.split_filter.lower()
        self.browser_microcopy_var.set(
            f"browser // {visible:04d} visible // pool {total:04d} // "
            f"filter {split_filter} // cache {self.controller.prediction_cache_size:03d}"
        )

    def _update_console_microcopy(self) -> None:
        row = self.controller.current_row
        if row is None:
            self.console_microcopy_var.set(
                "console // standby // select a row to enter playback mode"
            )
            return

        position, total = self.controller.current_position()
        self.console_microcopy_var.set(
            f"console // split {row.split} // subject {row.subject_key} // "
            f"photo {row.photo_id} // lane {position:04d}/{total:04d}"
        )

    def _draw_hero_banner(self) -> None:
        width = max(self.hero_canvas.winfo_width(), 960)
        height = max(self.hero_canvas.winfo_height(), 176)
        base = Image.new(
            "RGBA",
            (width, height),
            _hex_to_rgba(self.theme["hero_background"], 255),
        )
        blur_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        blur_draw = ImageDraw.Draw(blur_layer)
        blur_draw.rectangle(
            (24, 20, width - 220, 88),
            fill=_hex_to_rgba(self.theme["hero_glow"], 86),
        )
        blur_draw.rectangle(
            (width - 320, 28, width - 42, height - 28),
            fill=_hex_to_rgba(self.theme["accent_soft"], 96),
        )
        blur_draw.rectangle(
            (40, height - 68, width // 2, height - 28),
            fill=_hex_to_rgba(self.theme["hero_band"], 112),
        )
        blur_layer = blur_layer.filter(ImageFilter.GaussianBlur(radius=14))
        base = Image.alpha_composite(base, blur_layer)

        line_draw = ImageDraw.Draw(base)
        line_draw.line(
            (0, 22, width, 22),
            fill=_hex_to_rgba(self.theme["hero_outline"], 150),
            width=2,
        )
        line_draw.line(
            (0, height - 24, width, height - 24),
            fill=_hex_to_rgba(self.theme["divider"], 190),
            width=2,
        )
        line_draw.rectangle(
            (width - 290, 30, width - 48, height - 30),
            outline=_hex_to_rgba(self.theme["hero_outline"], 230),
            width=2,
        )
        line_draw.rectangle(
            (width - 258, 46, width - 82, height - 46),
            outline=_hex_to_rgba(self.theme["accent"], 180),
            width=1,
        )
        line_draw.rectangle(
            (54, 38, 178, height - 42),
            outline=_hex_to_rgba(self.theme["hero_outline"], 170),
            width=1,
        )

        hero_image = ImageTk.PhotoImage(base.convert("RGB"))
        self._hero_photo_image = hero_image
        self.hero_canvas.delete("all")
        self.hero_canvas.create_image(0, 0, anchor="nw", image=hero_image)

        row = self.controller.current_row
        if row is None:
            context_title = "Soft Metrics Lounge"
            context_line = "evaluation-first // no sample selected"
            split_copy = "split // standby"
        else:
            context_title = row.photo_id[:24]
            context_line = (
                f"{row.subject_key} // {row.hwg_gender} // "
                f"{row.hwg_height_cm:.1f} cm // {row.hwg_weight_kg:.1f} kg"
            )
            split_copy = f"split // {row.split}"

        self.hero_canvas.create_text(
            42,
            42,
            anchor="w",
            text=self.theme["brand_title"],
            fill=self.theme["text_primary"],
            font=("Segoe UI Semibold", 30),
        )
        self.hero_canvas.create_text(
            44,
            74,
            anchor="w",
            text="BODY MEASUREMENT VISION // SILHOUETTE ACCURACY LAB",
            fill=self.theme["hero_microcopy"],
            font=("Consolas", 11, "italic"),
        )
        self.hero_canvas.create_text(
            44,
            120,
            anchor="w",
            text=context_title,
            fill=self.theme["accent_deep"],
            font=("Segoe UI Semibold", 18),
        )
        self.hero_canvas.create_text(
            44,
            146,
            anchor="w",
            text=context_line,
            fill=self.theme["text_muted"],
            font=("Consolas", 10),
        )
        self.hero_canvas.create_text(
            width - 268,
            56,
            anchor="w",
            text="PLAYBACK NODE",
            fill=self.theme["accent_deep"],
            font=("Consolas", 11, "bold"),
        )
        self.hero_canvas.create_text(
            width - 268,
            86,
            anchor="w",
            text=split_copy,
            fill=self.theme["text_primary"],
            font=("Consolas", 10),
        )
        self.hero_canvas.create_text(
            width - 268,
            112,
            anchor="w",
            text="paired silhouettes // cached replay",
            fill=self.theme["hero_microcopy"],
            font=("Consolas", 10),
        )
        self.hero_canvas.create_text(
            width - 268,
            138,
            anchor="w",
            text="...to work at your own pace",
            fill=self.theme["text_muted"],
            font=("Consolas", 9, "italic"),
        )

    def _sync_tree_selection(self) -> None:
        current_row = self.controller.current_row
        if current_row is None:
            return
        if self.row_tree.exists(current_row.photo_id):
            self.row_tree.selection_set(current_row.photo_id)
            self.row_tree.focus(current_row.photo_id)
            self.row_tree.see(current_row.photo_id)

    def _on_split_filter_changed(self, _: tk.Event[Any]) -> None:
        self.apply_split_filter(self.split_var.get())

    def _on_search_changed(self, _: tk.Event[Any]) -> None:
        self.apply_search_query(self.search_var.get())

    def _on_jump_requested(self, _: tk.Event[Any]) -> None:
        self._jump_to_index()

    def _jump_to_index(self) -> None:
        try:
            position = int(self.jump_var.get().strip())
            self.controller.jump_to_position(position)
        except (TypeError, ValueError, IndexError) as exc:
            self._set_status(f"Invalid jump position: {exc}")
            return

        self._sync_tree_selection()
        self._render_current_row()
        self.request_prediction_for_current_row()

    def _on_tree_selection_changed(self, _: tk.Event[Any]) -> None:
        selection = self.row_tree.selection()
        if not selection:
            return
        photo_id = selection[0]
        found = self.controller.select_photo_id(photo_id)
        if not found:
            self._set_status(f"Unable to find selected photo_id: {photo_id}")
            return
        self._render_current_row()
        self.request_prediction_for_current_row()


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    stripped = hex_color.lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got: {hex_color!r}")
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return red, green, blue, alpha


def _pil_resample_nearest() -> Any:
    resampling = getattr(Image, "Resampling", Image)
    return resampling.NEAREST


def launch_bodym_accuracy_explorer(
    config: BodyMExplorerConfig | None = None,
) -> None:
    resolved_config = config or BodyMExplorerConfig()
    rows = load_explorer_rows(
        val_manifest_path=resolved_config.val_manifest_path,
        full_manifest_path=resolved_config.full_manifest_path,
    )
    inference_service = load_inference_service(
        checkpoint_path=resolved_config.checkpoint_path,
        config_path=resolved_config.config_path,
        device=resolved_config.device,
    )
    controller = BodyMAccuracyExplorerController(
        rows=rows,
        inference_service=inference_service,
        start_split=resolved_config.start_split,
        start_photo_id=resolved_config.start_photo_id,
    )
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise BodyMExplorerError(
            "Tkinter could not start because Tcl/Tk is unavailable in this Python "
            "installation."
        ) from exc
    BodyMAccuracyExplorerApp(root, controller)
    root.mainloop()


__all__ = [
    "BodyMAccuracyExplorerApp",
    "BodyMAccuracyExplorerController",
    "BodyMExplorerConfig",
    "BodyMExplorerError",
    "BodyMExplorerRow",
    "BodyMMetricComparisonRow",
    "BodyMSamplePredictionResult",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_FULL_MANIFEST_PATH",
    "DEFAULT_START_SPLIT",
    "DEFAULT_VAL_MANIFEST_PATH",
    "build_sample_prediction_result",
    "launch_bodym_accuracy_explorer",
    "load_explorer_rows",
]
