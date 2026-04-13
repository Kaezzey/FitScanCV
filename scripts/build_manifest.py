from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Error: pandas is required to run scripts/build_manifest.py. Install pandas and retry."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_CSV_FILES: tuple[str, ...] = (
    "hwg_metadata.csv",
    "measurements.csv",
    "subject_to_photo_map.csv",
)
EXPECTED_IMAGE_DIRS: tuple[str, ...] = ("mask", "mask_left")
PHOTO_MAP_COLUMNS: tuple[str, ...] = ("subject_id", "photo_id")
HWG_COLUMNS: tuple[str, ...] = ("subject_id", "gender", "height_cm", "weight_kg")
MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "ankle",
    "arm-length",
    "bicep",
    "calf",
    "chest",
    "forearm",
    "height",
    "hip",
    "leg-length",
    "shoulder-breadth",
    "shoulder-to-crotch",
    "thigh",
    "waist",
    "wrist",
)
HWG_RENAME_MAP: dict[str, str] = {
    "gender": "hwg_gender",
    "height_cm": "hwg_height_cm",
    "weight_kg": "hwg_weight_kg",
}
MEASUREMENT_RENAME_MAP: dict[str, str] = {
    "ankle": "measurement_ankle",
    "arm-length": "measurement_arm_length",
    "bicep": "measurement_bicep",
    "calf": "measurement_calf",
    "chest": "measurement_chest",
    "forearm": "measurement_forearm",
    "height": "measurement_height",
    "hip": "measurement_hip",
    "leg-length": "measurement_leg_length",
    "shoulder-breadth": "measurement_shoulder_breadth",
    "shoulder-to-crotch": "measurement_shoulder_to_crotch",
    "thigh": "measurement_thigh",
    "waist": "measurement_waist",
    "wrist": "measurement_wrist",
}
OUTPUT_COLUMNS: list[str] = [
    "split",
    "subject_id",
    "subject_key",
    "photo_id",
    "mask_path",
    "mask_left_path",
    "hwg_gender",
    "hwg_height_cm",
    "hwg_weight_kg",
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
]
HWG_NUMERIC_COLUMNS: tuple[str, ...] = ("height_cm", "weight_kg")
MEASUREMENT_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column for column in MEASUREMENT_COLUMNS if column != "subject_id"
)


class ManifestError(RuntimeError):
    """Raised when the BodyM manifest cannot be built safely."""


@dataclass(frozen=True)
class SplitArtifacts:
    name: str
    split_path: Path
    photo_map: pd.DataFrame
    hwg: pd.DataFrame
    measurements: pd.DataFrame
    mask_ids: pd.Index
    mask_left_ids: pd.Index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a verified CSV manifest for the raw BodyM dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/bodym"),
        help="Path to the raw BodyM dataset root.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/interim/bodym_manifest.csv"),
        help="Path to the output manifest CSV.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def ensure_inside_repo(path: Path, label: str) -> None:
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ManifestError(
            f"{label} must be inside the repository root to emit repo-relative paths: {path}"
        ) from exc


def relative_repo_path(path: Path) -> str:
    ensure_inside_repo(path, "Dataset asset")
    return path.relative_to(REPO_ROOT).as_posix()


def split_sort_key(path: Path) -> tuple[int, str]:
    return (0 if path.name == "train" else 1, path.name.lower())


def validate_dataset_root(dataset_root: Path) -> Path:
    if not dataset_root.exists():
        raise ManifestError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise ManifestError(f"Dataset root is not a directory: {dataset_root}")
    ensure_inside_repo(dataset_root, "Dataset root")
    return dataset_root


def discover_splits(dataset_root: Path) -> list[Path]:
    split_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir()],
        key=split_sort_key,
    )
    if not split_dirs:
        raise ManifestError(
            f"No split directories were found under dataset root: {dataset_root}"
        )
    return split_dirs


def load_csv_frame(csv_path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except OSError as exc:
        raise ManifestError(f"Failed to read CSV file: {csv_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ManifestError(f"CSV file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ManifestError(f"Failed to parse CSV file: {csv_path}") from exc

    dataframe.columns = [str(column).strip() for column in dataframe.columns.tolist()]

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ManifestError(
            f"CSV file is missing required columns {missing_columns}: {csv_path}"
        )

    dataframe = dataframe.loc[:, list(required_columns)].copy()
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else ""
        )

    return dataframe


def validate_non_empty_columns(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> None:
    empty_counts: list[str] = []
    for column in columns:
        empty_count = int((dataframe[column] == "").sum())
        if empty_count > 0:
            empty_counts.append(f"{column}={empty_count}")

    if empty_counts:
        raise ManifestError(f"{label} has empty required values: {', '.join(empty_counts)}")


def convert_numeric_columns(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    converted = dataframe.copy()
    for column in columns:
        numeric_values = pd.to_numeric(converted[column], errors="coerce")
        invalid_count = int(numeric_values.isna().sum())
        if invalid_count > 0:
            raise ManifestError(
                f"{label} contains non-numeric values in column '{column}': {invalid_count}"
            )
        converted[column] = numeric_values
    return converted


def validate_unique_column(
    dataframe: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    duplicate_series = dataframe.loc[
        dataframe[column].duplicated(keep=False),
        column,
    ].drop_duplicates().sort_values()

    if not duplicate_series.empty:
        examples = ", ".join(duplicate_series.head(5).astype(str).tolist())
        raise ManifestError(
            f"{label} has duplicate '{column}' values. Examples: {examples}"
        )


def list_png_ids(image_dir: Path) -> pd.Index:
    if not image_dir.exists():
        raise ManifestError(f"Missing expected image directory: {image_dir}")
    if not image_dir.is_dir():
        raise ManifestError(f"Expected a directory for image artifacts: {image_dir}")

    png_ids = pd.Index(
        sorted(
            path.stem
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ),
        dtype="string",
    )

    duplicate_ids = png_ids[png_ids.duplicated()].drop_duplicates().tolist()
    if duplicate_ids:
        examples = ", ".join(duplicate_ids[:5])
        raise ManifestError(
            f"Image directory has duplicate PNG basenames: {image_dir}. Examples: {examples}"
        )

    return png_ids


def validate_image_alignment(
    split_name: str,
    photo_ids: pd.Index,
    image_ids: pd.Index,
    image_dir_name: str,
) -> None:
    photo_id_set = set(photo_ids.tolist())
    image_id_set = set(image_ids.tolist())

    missing_images = sorted(photo_id_set - image_id_set)
    if missing_images:
        examples = ", ".join(missing_images[:5])
        raise ManifestError(
            f"{split_name}: mapped photo_id values are missing from '{image_dir_name}'. Examples: {examples}"
        )

    orphan_images = sorted(image_id_set - photo_id_set)
    if orphan_images:
        examples = ", ".join(orphan_images[:5])
        raise ManifestError(
            f"{split_name}: '{image_dir_name}' contains orphan PNG basenames not present in subject_to_photo_map.csv. Examples: {examples}"
        )


def validate_subject_joins(
    split_name: str,
    photo_map: pd.DataFrame,
    subject_frame: pd.DataFrame,
    subject_frame_name: str,
) -> None:
    missing_subject_ids = sorted(
        set(photo_map["subject_id"].tolist()) - set(subject_frame["subject_id"].tolist())
    )
    if missing_subject_ids:
        examples = ", ".join(missing_subject_ids[:5])
        raise ManifestError(
            f"{split_name}: subject_to_photo_map.csv has subject_id values missing from {subject_frame_name}. Examples: {examples}"
        )


def load_split_artifacts(split_path: Path) -> SplitArtifacts:
    split_name = split_path.name

    csv_paths = {name: split_path / name for name in EXPECTED_CSV_FILES}
    for csv_name, csv_path in csv_paths.items():
        if not csv_path.is_file():
            raise ManifestError(
                f"{split_name}: missing expected CSV file '{csv_name}' at {csv_path}"
            )

    for image_dir_name in EXPECTED_IMAGE_DIRS:
        image_dir = split_path / image_dir_name
        if not image_dir.is_dir():
            raise ManifestError(
                f"{split_name}: missing expected image directory '{image_dir_name}' at {image_dir}"
            )

    photo_map = load_csv_frame(csv_paths["subject_to_photo_map.csv"], PHOTO_MAP_COLUMNS)
    hwg = load_csv_frame(csv_paths["hwg_metadata.csv"], HWG_COLUMNS)
    measurements = load_csv_frame(csv_paths["measurements.csv"], MEASUREMENT_COLUMNS)

    validate_non_empty_columns(
        photo_map,
        PHOTO_MAP_COLUMNS,
        f"{split_name}/subject_to_photo_map.csv",
    )
    validate_non_empty_columns(
        hwg,
        HWG_COLUMNS,
        f"{split_name}/hwg_metadata.csv",
    )
    validate_non_empty_columns(
        measurements,
        MEASUREMENT_COLUMNS,
        f"{split_name}/measurements.csv",
    )

    validate_unique_column(hwg, "subject_id", f"{split_name}/hwg_metadata.csv")
    validate_unique_column(measurements, "subject_id", f"{split_name}/measurements.csv")

    hwg = convert_numeric_columns(hwg, HWG_NUMERIC_COLUMNS, f"{split_name}/hwg_metadata.csv")
    measurements = convert_numeric_columns(
        measurements,
        MEASUREMENT_NUMERIC_COLUMNS,
        f"{split_name}/measurements.csv",
    )

    validate_subject_joins(split_name, photo_map, hwg, "hwg_metadata.csv")
    validate_subject_joins(split_name, photo_map, measurements, "measurements.csv")

    mask_ids = list_png_ids(split_path / "mask")
    mask_left_ids = list_png_ids(split_path / "mask_left")
    photo_ids = pd.Index(photo_map["photo_id"].tolist(), dtype="string")

    validate_image_alignment(split_name, photo_ids, mask_ids, "mask")
    validate_image_alignment(split_name, photo_ids, mask_left_ids, "mask_left")

    return SplitArtifacts(
        name=split_name,
        split_path=split_path,
        photo_map=photo_map,
        hwg=hwg,
        measurements=measurements,
        mask_ids=mask_ids,
        mask_left_ids=mask_left_ids,
    )


def build_split_manifest(split_artifacts: SplitArtifacts) -> pd.DataFrame:
    split_base_path = relative_repo_path(split_artifacts.split_path)

    manifest = split_artifacts.photo_map.copy()
    manifest.insert(0, "split", split_artifacts.name)
    manifest.insert(2, "subject_key", manifest["split"] + "::" + manifest["subject_id"])
    manifest["mask_path"] = manifest["photo_id"].map(
        lambda photo_id: f"{split_base_path}/mask/{photo_id}.png"
    )
    manifest["mask_left_path"] = manifest["photo_id"].map(
        lambda photo_id: f"{split_base_path}/mask_left/{photo_id}.png"
    )

    hwg = split_artifacts.hwg.rename(columns=HWG_RENAME_MAP)
    measurements = split_artifacts.measurements.rename(columns=MEASUREMENT_RENAME_MAP)

    manifest = manifest.merge(
        hwg,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    manifest = manifest.merge(
        measurements,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    missing_output_values: list[str] = []
    for column in OUTPUT_COLUMNS:
        if column in {"split", "subject_id", "subject_key", "photo_id"}:
            empty_count = int((manifest[column] == "").sum())
        elif manifest[column].dtype == object:
            empty_count = int((manifest[column] == "").sum())
        else:
            empty_count = int(manifest[column].isna().sum())

        if empty_count > 0:
            missing_output_values.append(f"{column}={empty_count}")

    if missing_output_values:
        raise ManifestError(
            f"{split_artifacts.name}: merged manifest contains empty required values: "
            + ", ".join(missing_output_values)
        )

    manifest = manifest.loc[:, OUTPUT_COLUMNS].copy()
    return manifest.sort_values(["split", "photo_id"], kind="stable").reset_index(drop=True)


def build_manifest(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    split_paths = discover_splits(dataset_root)
    split_artifacts = [load_split_artifacts(split_path) for split_path in split_paths]

    combined_photo_map = pd.concat(
        [artifacts.photo_map.assign(split=artifacts.name) for artifacts in split_artifacts],
        ignore_index=True,
    )
    validate_unique_column(
        combined_photo_map,
        "photo_id",
        "Combined subject_to_photo_map.csv records",
    )

    split_manifests = [build_split_manifest(artifacts) for artifacts in split_artifacts]
    manifest = pd.concat(split_manifests, ignore_index=True)
    validate_unique_column(manifest, "photo_id", "Final manifest")

    split_row_counts = {
        split_name: int(count)
        for split_name, count in manifest["split"].value_counts(sort=False).items()
    }
    return manifest, split_row_counts


def write_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_path, index=False)
    except OSError as exc:
        raise ManifestError(f"Failed to write manifest CSV: {output_path}") from exc


def format_summary(
    dataset_root: Path,
    output_path: Path,
    manifest: pd.DataFrame,
    split_row_counts: dict[str, int],
) -> str:
    lines = [
        "BodyM Manifest Build",
        "====================",
        f"Dataset root: {dataset_root}",
        f"Output path: {output_path}",
        "",
        "Per-Split Rows",
        "--------------",
    ]

    for split_path in discover_splits(dataset_root):
        lines.append(f"  {split_path.name}: {split_row_counts.get(split_path.name, 0)}")

    lines.extend(
        [
            "",
            f"Total rows: {len(manifest)}",
            f"Unique photo_id values: {manifest['photo_id'].nunique()}",
            f"Unique subject_key values: {manifest['subject_key'].nunique()}",
            "Validation status: PASSED",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        dataset_root = validate_dataset_root(resolve_repo_path(args.dataset_root))
        output_path = resolve_repo_path(args.output_path)
        manifest, split_row_counts = build_manifest(dataset_root)
        write_manifest(manifest, output_path)
        print(format_summary(dataset_root, output_path, manifest, split_row_counts))
    except ManifestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
