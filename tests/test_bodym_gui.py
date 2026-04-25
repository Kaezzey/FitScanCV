from __future__ import annotations

from contextlib import contextmanager
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("PIL")

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bodym_dataset import MANIFEST_MEASUREMENT_COLUMNS
from scripts.gui.bodym_gui import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FULL_MANIFEST_PATH,
    DEFAULT_START_SPLIT,
    DEFAULT_VAL_MANIFEST_PATH,
    BodyMAccuracyExplorerApp,
    BodyMAccuracyExplorerController,
    BodyMExplorerError,
    BodyMExplorerRow,
    build_sample_prediction_result,
    load_explorer_rows,
)
from scripts.inference.bodym_inference import load_inference_service
from scripts.run_bodym_gui import parse_args
from scripts.run_bodym_gui import (
    DEFAULT_CHECKPOINT_PATH as RUNNER_DEFAULT_CHECKPOINT_PATH,
)
from scripts.run_bodym_gui import (
    DEFAULT_FULL_MANIFEST_PATH as RUNNER_DEFAULT_FULL_MANIFEST_PATH,
)
from scripts.run_bodym_gui import DEFAULT_START_SPLIT as RUNNER_DEFAULT_START_SPLIT
from scripts.run_bodym_gui import (
    DEFAULT_VAL_MANIFEST_PATH as RUNNER_DEFAULT_VAL_MANIFEST_PATH,
)

SCRATCH_ROOT = REPO_ROOT / "data" / "interim" / "test_scratch"
REAL_CHECKPOINT_PATH = REPO_ROOT / "outputs" / "bodym_resnet18" / "best.pt"
REAL_VAL_MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_training_val.csv"
REAL_FULL_MANIFEST_PATH = REPO_ROOT / "data" / "interim" / "bodym_manifest.csv"


@contextmanager
def scratch_dir() -> Path:
    SCRATCH_ROOT.mkdir(exist_ok=True)
    directory = SCRATCH_ROOT / f"gui_case_{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def create_hidden_root() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tkinter root is unavailable: {exc}")
    root.withdraw()
    return root


def pump_tk(
    root: tk.Tk,
    condition,
    *,
    timeout: float = 3.0,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update_idletasks()
        root.update()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def close_root(root: tk.Tk, app: BodyMAccuracyExplorerApp | None = None) -> None:
    try:
        if app is not None:
            app.close()
        elif root.winfo_exists():
            root.destroy()
    except tk.TclError:
        pass


def write_mask_image(path: Path, *, fill: int) -> Path:
    image = Image.new("L", (80, 120), color=fill)
    image.save(path)
    return path


def make_measurements(offset: float = 0.0) -> dict[str, str]:
    return {
        target_name: str(float(index + 1) + offset)
        for index, target_name in enumerate(MANIFEST_MEASUREMENT_COLUMNS)
    }


def make_manifest_row(
    *,
    split: str,
    subject_id: str,
    subject_key: str,
    photo_id: str,
    mask_path: str,
    mask_left_path: str,
    measurement_offset: float = 0.0,
) -> dict[str, str]:
    return {
        "split": split,
        "subject_id": subject_id,
        "subject_key": subject_key,
        "photo_id": photo_id,
        "mask_path": mask_path,
        "mask_left_path": mask_left_path,
        "hwg_gender": "female",
        "hwg_height_cm": "165.0",
        "hwg_weight_kg": "62.0",
        **make_measurements(offset=measurement_offset),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class FakeInferenceService:
    def __init__(self, *, prediction_delta: float = 0.5, error: Exception | None = None) -> None:
        self.checkpoint_path = Path("fake_checkpoint.pt")
        self.device = "cpu"
        self.target_names = MANIFEST_MEASUREMENT_COLUMNS
        self.prediction_delta = prediction_delta
        self.error = error
        self.calls: list[dict[str, object]] = []

    def predict_from_paths(
        self,
        *,
        front_image_path: str | Path,
        side_image_path: str | Path | None,
        hwg_gender: str,
        hwg_height_cm: float,
        hwg_weight_kg: float,
    ) -> dict[str, object]:
        if self.error is not None:
            raise self.error

        self.calls.append(
            {
                "front_image_path": str(front_image_path),
                "side_image_path": None if side_image_path is None else str(side_image_path),
                "hwg_gender": hwg_gender,
                "hwg_height_cm": hwg_height_cm,
                "hwg_weight_kg": hwg_weight_kg,
            }
        )
        predictions = {
            target_name: float(index + 1) + self.prediction_delta
            for index, target_name in enumerate(MANIFEST_MEASUREMENT_COLUMNS)
        }
        return {
            "predictions": predictions,
            "model": {
                "checkpoint_path": str(self.checkpoint_path),
                "device": self.device,
            },
        }


def build_synthetic_manifests(tmp_path: Path) -> tuple[Path, Path]:
    val_front_1 = write_mask_image(tmp_path / "val_front_1.png", fill=120)
    val_side_1 = write_mask_image(tmp_path / "val_side_1.png", fill=140)
    val_front_2 = write_mask_image(tmp_path / "val_front_2.png", fill=160)
    val_side_2 = write_mask_image(tmp_path / "val_side_2.png", fill=180)
    testa_front = write_mask_image(tmp_path / "testa_front.png", fill=90)
    testa_side = write_mask_image(tmp_path / "testa_side.png", fill=110)
    testb_front = write_mask_image(tmp_path / "testb_front.png", fill=70)
    testb_side = write_mask_image(tmp_path / "testb_side.png", fill=100)
    train_front = write_mask_image(tmp_path / "train_front.png", fill=200)
    train_side = write_mask_image(tmp_path / "train_side.png", fill=220)

    val_rows = [
        make_manifest_row(
            split="val",
            subject_id="subject-val-1",
            subject_key="val::subject-val-1",
            photo_id="val-photo-1",
            mask_path=str(val_front_1),
            mask_left_path=str(val_side_1),
        ),
        make_manifest_row(
            split="val",
            subject_id="subject-val-2",
            subject_key="val::subject-val-2",
            photo_id="val-photo-2",
            mask_path=str(val_front_2),
            mask_left_path=str(val_side_2),
            measurement_offset=1.0,
        ),
    ]
    full_rows = [
        make_manifest_row(
            split="testA",
            subject_id="subject-holdout-a",
            subject_key="testA::subject-holdout-a",
            photo_id="testA-photo-1",
            mask_path=str(testa_front),
            mask_left_path=str(testa_side),
        ),
        make_manifest_row(
            split="testB",
            subject_id="subject-holdout-b",
            subject_key="testB::subject-holdout-b",
            photo_id="testB-photo-1",
            mask_path=str(testb_front),
            mask_left_path=str(testb_side),
        ),
        make_manifest_row(
            split="train",
            subject_id="subject-train-ignored",
            subject_key="train::subject-train-ignored",
            photo_id="train-photo-ignored",
            mask_path=str(train_front),
            mask_left_path=str(train_side),
        ),
    ]

    val_manifest_path = write_manifest(tmp_path / "val_manifest.csv", val_rows)
    full_manifest_path = write_manifest(tmp_path / "full_manifest.csv", full_rows)
    return val_manifest_path, full_manifest_path


def test_load_explorer_rows_uses_val_and_holdouts_only() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)

        rows = load_explorer_rows(
            val_manifest_path=val_manifest_path,
            full_manifest_path=full_manifest_path,
        )

        assert len(rows) == 4
        assert [row.split for row in rows] == ["val", "val", "testA", "testB"]
        assert all(row.split != "train" for row in rows)


def test_load_explorer_rows_enforces_required_measurement_columns() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)
        frame = pd.read_csv(val_manifest_path)
        frame = frame.drop(columns=["measurement_wrist"])
        frame.to_csv(val_manifest_path, index=False)

        with pytest.raises(BodyMExplorerError, match="missing required columns"):
            load_explorer_rows(
                val_manifest_path=val_manifest_path,
                full_manifest_path=full_manifest_path,
            )


def test_build_sample_prediction_result_computes_error_rows() -> None:
    row = make_manifest_row(
        split="val",
        subject_id="subject-1",
        subject_key="val::subject-1",
        photo_id="photo-1",
        mask_path="front.png",
        mask_left_path="side.png",
    )
    explorer_row = BodyMExplorerRow(
        split=row["split"],
        subject_id=row["subject_id"],
        subject_key=row["subject_key"],
        photo_id=row["photo_id"],
        mask_path=row["mask_path"],
        mask_left_path=row["mask_left_path"],
        hwg_gender=row["hwg_gender"],
        hwg_height_cm=float(row["hwg_height_cm"]),
        hwg_weight_kg=float(row["hwg_weight_kg"]),
        measurements={
            target_name: float(row[target_name]) for target_name in MANIFEST_MEASUREMENT_COLUMNS
        },
        source_manifest_path="manifest.csv",
    )
    predictions = {
        target_name: float(index + 1) + 0.5
        for index, target_name in enumerate(MANIFEST_MEASUREMENT_COLUMNS)
    }

    result = build_sample_prediction_result(explorer_row, predictions)

    assert len(result.metric_rows) == 14
    assert result.metric_rows[0].ground_truth == 1.0
    assert result.metric_rows[0].prediction == 1.5
    assert result.metric_rows[0].signed_error == 0.5
    assert result.sample_mean_absolute_error == pytest.approx(0.5)
    assert result.sample_rmse == pytest.approx(0.5)


def test_controller_supports_split_filter_search_and_photo_lookup() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)
        rows = load_explorer_rows(val_manifest_path, full_manifest_path)
        controller = BodyMAccuracyExplorerController(
            rows=rows,
            inference_service=FakeInferenceService(),
            start_split="val",
        )

        assert controller.current_row is not None
        assert controller.current_row.split == "val"
        assert controller.select_photo_id("val-photo-2") is True
        assert controller.current_row is not None
        assert controller.current_row.photo_id == "val-photo-2"

        controller.set_split_filter("All")
        controller.set_search_query("testA::subject-holdout-a")

        assert len(controller.filtered_rows) == 1
        assert controller.current_row is not None
        assert controller.current_row.subject_key == "testA::subject-holdout-a"


def test_controller_prediction_cache_avoids_duplicate_inference_calls() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)
        rows = load_explorer_rows(val_manifest_path, full_manifest_path)
        service = FakeInferenceService()
        controller = BodyMAccuracyExplorerController(
            rows=rows,
            inference_service=service,
            start_split="val",
        )
        row = controller.current_row
        assert row is not None

        first_result = controller.predict_row(row)
        second_result = controller.predict_row(row)

        assert service.calls
        assert len(service.calls) == 1
        assert first_result == second_result


def test_gui_app_initializes_hidden_and_updates_prediction_state() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)
        rows = load_explorer_rows(val_manifest_path, full_manifest_path)
        controller = BodyMAccuracyExplorerController(
            rows=rows,
            inference_service=FakeInferenceService(),
            start_split="val",
        )
        root = create_hidden_root()
        app: BodyMAccuracyExplorerApp | None = None
        try:
            app = BodyMAccuracyExplorerApp(root, controller)

            ready = pump_tk(
                root,
                lambda: len(app.results_tree.get_children()) == 14,
            )

            assert ready is True
            assert root.state() == "withdrawn"
            assert "Sample Mean AE" in app.summary_var.get()
        finally:
            close_root(root, app)


def test_gui_app_handles_prediction_error_without_crashing() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)
        rows = load_explorer_rows(val_manifest_path, full_manifest_path)
        controller = BodyMAccuracyExplorerController(
            rows=rows,
            inference_service=FakeInferenceService(
                error=FileNotFoundError("Image file does not exist: missing.png")
            ),
            start_split="val",
        )
        root = create_hidden_root()
        app: BodyMAccuracyExplorerApp | None = None
        try:
            app = BodyMAccuracyExplorerApp(root, controller)

            ready = pump_tk(
                root,
                lambda: "Error:" in app.status_var.get(),
            )

            assert ready is True
            assert "Image file does not exist" in app.status_var.get()
        finally:
            close_root(root, app)


def test_gui_navigation_and_split_filter_update_active_row() -> None:
    with scratch_dir() as tmp_path:
        val_manifest_path, full_manifest_path = build_synthetic_manifests(tmp_path)
        rows = load_explorer_rows(val_manifest_path, full_manifest_path)
        controller = BodyMAccuracyExplorerController(
            rows=rows,
            inference_service=FakeInferenceService(),
            start_split="val",
        )
        root = create_hidden_root()
        app: BodyMAccuracyExplorerApp | None = None
        try:
            app = BodyMAccuracyExplorerApp(root, controller, auto_predict=False)

            first_photo_id = controller.current_row.photo_id if controller.current_row else None
            app.apply_split_filter("All")
            app.go_to_next_row()

            assert controller.current_row is not None
            assert controller.current_row.photo_id != first_photo_id

            app.apply_split_filter("testA")

            assert controller.current_row is not None
            assert controller.current_row.split == "testA"
        finally:
            close_root(root, app)


def test_run_bodym_gui_defaults_to_tuned_checkpoint_and_val_start() -> None:
    args = parse_args([])

    assert args.checkpoint == RUNNER_DEFAULT_CHECKPOINT_PATH
    assert args.val_manifest == RUNNER_DEFAULT_VAL_MANIFEST_PATH
    assert args.full_manifest == RUNNER_DEFAULT_FULL_MANIFEST_PATH
    assert args.start_split == RUNNER_DEFAULT_START_SPLIT


def test_real_local_gui_smoke_if_assets_exist() -> None:
    if not REAL_CHECKPOINT_PATH.is_file():
        pytest.skip("Local tuned checkpoint is not available.")
    if not REAL_VAL_MANIFEST_PATH.is_file() or not REAL_FULL_MANIFEST_PATH.is_file():
        pytest.skip("Local validation/full manifests are not available.")

    rows = load_explorer_rows(
        val_manifest_path=REAL_VAL_MANIFEST_PATH,
        full_manifest_path=REAL_FULL_MANIFEST_PATH,
    )
    assert len(rows) == 3402

    service = load_inference_service(
        checkpoint_path=REAL_CHECKPOINT_PATH,
        device="cpu",
    )
    controller = BodyMAccuracyExplorerController(
        rows=rows,
        inference_service=service,
        start_split="val",
    )
    root = create_hidden_root()
    app: BodyMAccuracyExplorerApp | None = None
    try:
        app = BodyMAccuracyExplorerApp(root, controller, auto_predict=False)
        app.request_prediction_for_current_row()

        ready = pump_tk(
            root,
            lambda: (
                len(app.results_tree.get_children()) == 14
                and app._mask_photo_image is not None
                and app._mask_left_photo_image is not None
            ),
            timeout=8.0,
        )

        assert ready is True
        assert controller.current_row is not None
        assert controller.current_row.split == "val"
    finally:
        close_root(root, app)
