from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Error: pandas is required to run scripts/inspect_bodym.py. Install pandas and retry."
    ) from exc

EXPECTED_CSV_FILES: tuple[str, ...] = (
    "hwg_metadata.csv",
    "measurements.csv",
    "subject_to_photo_map.csv",
)
EXPECTED_IMAGE_DIRS: tuple[str, ...] = ("mask", "mask_left")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class InspectionError(RuntimeError):
    """Raised when the dataset cannot be inspected safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the raw BodyM dataset structure and report observed facts."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/bodym"),
        help="Path to the raw BodyM dataset root.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of sample file paths and image dimensions to report per artifact.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the inspection report as JSON.",
    )
    args = parser.parse_args(argv)

    if args.sample_count < 1:
        parser.error("--sample-count must be greater than or equal to 1.")

    return args


def validate_dataset_root(dataset_root: Path) -> Path:
    if not dataset_root.exists():
        raise InspectionError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise InspectionError(f"Dataset root is not a directory: {dataset_root}")
    return dataset_root.resolve()


def split_sort_key(path: Path) -> tuple[int, str]:
    return (0 if path.name == "train" else 1, path.name.lower())


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_splits(dataset_root: Path) -> list[Path]:
    split_dirs = sorted(
        (path for path in dataset_root.iterdir() if path.is_dir()),
        key=split_sort_key,
    )
    if not split_dirs:
        raise InspectionError(
            f"No split directories were found under dataset root: {dataset_root}"
        )
    return split_dirs


def load_csv_frame(csv_path: Path) -> tuple[list[str], pd.DataFrame]:
    try:
        dataframe = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except OSError as exc:
        raise InspectionError(f"Failed to read CSV file: {csv_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise InspectionError(f"CSV file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise InspectionError(f"Failed to parse CSV file: {csv_path}") from exc

    columns = [str(column).strip() for column in dataframe.columns.tolist()]
    if not columns:
        raise InspectionError(f"CSV file is missing a header row: {csv_path}")

    dataframe.columns = columns
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else ""
        )

    return columns, dataframe


def summarize_file_types(dataset_root: Path) -> dict[str, int]:
    suffixes = [
        file_path.suffix.lower() or "<no_extension>"
        for file_path in dataset_root.rglob("*")
        if file_path.is_file()
    ]
    if not suffixes:
        return {}

    counts = pd.Series(suffixes, dtype="string").value_counts().sort_index()
    return {str(extension): int(count) for extension, count in counts.items()}


def sample_paths(paths: list[Path], dataset_root: Path, sample_count: int) -> list[str]:
    return [relative_path(path, dataset_root) for path in paths[:sample_count]]


def count_duplicates(values: list[str], sample_count: int) -> tuple[int, list[str]]:
    if not values:
        return 0, []

    series = pd.Series(values, dtype="string")
    series = series[series != ""]
    if series.empty:
        return 0, []

    value_counts = series.value_counts()
    duplicate_rows = int((value_counts[value_counts > 1] - 1).sum())
    duplicate_examples = sorted(value_counts[value_counts > 1].index.tolist())
    return duplicate_rows, duplicate_examples[:sample_count]


def non_empty_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(dtype="string")
    series = dataframe[column]
    return series[series != ""]


def count_series_duplicates(
    series: pd.Series,
    sample_count: int,
) -> tuple[int, list[str]]:
    if series.empty:
        return 0, []

    duplicate_rows = int(series.duplicated(keep="first").sum())
    duplicate_examples = sorted(series[series.duplicated(keep=False)].unique().tolist())
    return duplicate_rows, duplicate_examples[:sample_count]


def count_pair_duplicates(
    dataframe: pd.DataFrame,
    left_column: str,
    right_column: str,
    sample_count: int,
) -> tuple[int, list[str]]:
    if left_column not in dataframe.columns or right_column not in dataframe.columns:
        return 0, []

    valid_rows = dataframe.loc[
        (dataframe[left_column] != "") & (dataframe[right_column] != ""),
        [left_column, right_column],
    ]
    if valid_rows.empty:
        return 0, []

    duplicate_rows = int(valid_rows.duplicated(keep="first").sum())
    duplicate_example_rows = valid_rows.loc[valid_rows.duplicated(keep=False)]
    duplicate_examples = sorted(
        duplicate_example_rows.apply(
            lambda row: f"{row[left_column]}|{row[right_column]}",
            axis=1,
        ).unique().tolist()
    )
    return duplicate_rows, duplicate_examples[:sample_count]


def column_value_set(dataframe: pd.DataFrame, column: str) -> set[str]:
    if column not in dataframe.columns:
        return set()
    series = non_empty_series(dataframe, column)
    return set(series.tolist())


def read_png_dimensions(image_path: Path) -> tuple[int, int]:
    # PNG width and height are stored in the IHDR chunk immediately after the signature.
    try:
        with image_path.open("rb") as handle:
            header = handle.read(24)
    except OSError as exc:
        raise InspectionError(f"Failed to read image file: {image_path}") from exc

    if len(header) < 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise InspectionError(f"Unsupported or invalid PNG file: {image_path}")

    width = int.from_bytes(header[16:20], byteorder="big")
    height = int.from_bytes(header[20:24], byteorder="big")
    return width, height


def summarize_csv_file(
    csv_path: Path,
    dataset_root: Path,
    sample_count: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns, dataframe = load_csv_frame(csv_path)
    summary: dict[str, Any] = {
        "exists": True,
        "relative_path": relative_path(csv_path, dataset_root),
        "row_count": int(len(dataframe.index)),
        "columns": columns,
        "sample_paths": [relative_path(csv_path, dataset_root)],
    }

    if "subject_id" in columns:
        subject_ids = non_empty_series(dataframe, "subject_id")
        summary["unique_subject_ids"] = int(subject_ids.nunique())
        if "photo_id" not in columns:
            duplicate_rows, duplicate_examples = count_series_duplicates(
                subject_ids,
                sample_count,
            )
            summary["duplicate_subject_id_rows"] = duplicate_rows
            summary["duplicate_subject_id_examples"] = duplicate_examples

    if "photo_id" in columns:
        photo_ids = non_empty_series(dataframe, "photo_id")
        duplicate_rows, duplicate_examples = count_series_duplicates(
            photo_ids,
            sample_count,
        )
        summary["unique_photo_ids"] = int(photo_ids.nunique())
        summary["duplicate_photo_id_rows"] = duplicate_rows
        summary["duplicate_photo_id_examples"] = duplicate_examples

        pair_duplicate_rows, pair_duplicate_examples = count_pair_duplicates(
            dataframe,
            "subject_id",
            "photo_id",
            sample_count,
        )
        summary["duplicate_subject_photo_rows"] = pair_duplicate_rows
        summary["duplicate_subject_photo_examples"] = pair_duplicate_examples

    return summary, dataframe


def summarize_image_dir(
    image_dir: Path,
    dataset_root: Path,
    sample_count: int,
) -> tuple[dict[str, Any], list[str]]:
    png_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    basenames = [path.stem for path in png_paths]
    duplicate_files, duplicate_examples = count_duplicates(basenames, sample_count)

    sample_dimensions: list[dict[str, Any]] = []
    for image_path in png_paths[:sample_count]:
        width, height = read_png_dimensions(image_path)
        sample_dimensions.append(
            {
                "file": relative_path(image_path, dataset_root),
                "width": width,
                "height": height,
            }
        )

    summary = {
        "exists": True,
        "relative_path": relative_path(image_dir, dataset_root),
        "png_count": len(png_paths),
        "unique_basenames": len(set(basenames)),
        "duplicate_basename_files": duplicate_files,
        "duplicate_basename_examples": duplicate_examples,
        "sample_paths": sample_paths(png_paths, dataset_root, sample_count),
        "sample_dimensions": sample_dimensions,
    }
    return summary, basenames


def set_difference_summary(
    left_label: str,
    left_values: set[str],
    right_label: str,
    right_values: set[str],
    sample_count: int,
) -> dict[str, Any]:
    left_only = sorted(left_values - right_values)
    right_only = sorted(right_values - left_values)
    return {
        f"{left_label}_minus_{right_label}_count": len(left_only),
        f"{left_label}_minus_{right_label}_examples": left_only[:sample_count],
        f"{right_label}_minus_{left_label}_count": len(right_only),
        f"{right_label}_minus_{left_label}_examples": right_only[:sample_count],
    }


def photos_per_subject(dataframe: pd.DataFrame) -> dict[str, Any]:
    if "subject_id" not in dataframe.columns or "photo_id" not in dataframe.columns:
        return {"min": 0, "max": 0, "avg": 0.0}

    valid_rows = dataframe.loc[
        (dataframe["subject_id"] != "") & (dataframe["photo_id"] != ""),
        ["subject_id", "photo_id"],
    ]
    if valid_rows.empty:
        return {"min": 0, "max": 0, "avg": 0.0}

    subject_counts = valid_rows.groupby("subject_id").size()
    average = round(float(subject_counts.mean()), 2)
    return {
        "min": int(subject_counts.min()),
        "max": int(subject_counts.max()),
        "avg": average,
    }


def inspect_split(
    split_path: Path,
    dataset_root: Path,
    sample_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    csv_summaries: dict[str, Any] = {}
    csv_frames_by_name: dict[str, pd.DataFrame] = {}
    image_summaries: dict[str, Any] = {}
    image_basenames: dict[str, set[str]] = {}

    for csv_name in EXPECTED_CSV_FILES:
        csv_path = split_path / csv_name
        if not csv_path.is_file():
            warnings.append(f"{split_path.name}: missing expected CSV file '{csv_name}'.")
            csv_summaries[csv_name] = {"exists": False}
            continue

        csv_summary, dataframe = summarize_csv_file(csv_path, dataset_root, sample_count)
        csv_summaries[csv_name] = csv_summary
        csv_frames_by_name[csv_name] = dataframe

        if csv_summary.get("duplicate_subject_id_rows", 0) > 0:
            warnings.append(
                f"{split_path.name}: '{csv_name}' has duplicate subject_id rows."
            )
        if csv_summary.get("duplicate_photo_id_rows", 0) > 0:
            warnings.append(
                f"{split_path.name}: '{csv_name}' has duplicate photo_id rows."
            )
        if csv_summary.get("duplicate_subject_photo_rows", 0) > 0:
            warnings.append(
                f"{split_path.name}: '{csv_name}' has duplicate subject/photo pairs."
            )

    for image_dir_name in EXPECTED_IMAGE_DIRS:
        image_dir = split_path / image_dir_name
        if not image_dir.is_dir():
            warnings.append(
                f"{split_path.name}: missing expected image directory '{image_dir_name}'."
            )
            image_summaries[image_dir_name] = {"exists": False}
            continue

        image_summary, basenames = summarize_image_dir(
            image_dir=image_dir,
            dataset_root=dataset_root,
            sample_count=sample_count,
        )
        image_summaries[image_dir_name] = image_summary
        image_basenames[image_dir_name] = set(basenames)

        if image_summary["duplicate_basename_files"] > 0:
            warnings.append(
                f"{split_path.name}: '{image_dir_name}' has duplicate image basenames."
            )

    empty_frame = pd.DataFrame()
    hwg_frame = csv_frames_by_name.get("hwg_metadata.csv", empty_frame)
    measurement_frame = csv_frames_by_name.get("measurements.csv", empty_frame)
    photo_map_frame = csv_frames_by_name.get("subject_to_photo_map.csv", empty_frame)

    hwg_subject_ids = column_value_set(hwg_frame, "subject_id")
    measurement_subject_ids = column_value_set(measurement_frame, "subject_id")
    photo_map_subject_ids = column_value_set(photo_map_frame, "subject_id")
    photo_map_photo_ids = column_value_set(photo_map_frame, "photo_id")

    mapping_summary = {
        "unique_subject_ids": {
            "hwg_metadata.csv": len(hwg_subject_ids),
            "measurements.csv": len(measurement_subject_ids),
            "subject_to_photo_map.csv": len(photo_map_subject_ids),
        },
        "unique_photo_ids": len(photo_map_photo_ids),
        "photos_per_subject": photos_per_subject(photo_map_frame),
    }

    subject_alignment = {
        **set_difference_summary(
            "hwg",
            hwg_subject_ids,
            "measurements",
            measurement_subject_ids,
            sample_count,
        ),
        **set_difference_summary(
            "hwg",
            hwg_subject_ids,
            "photo_map",
            photo_map_subject_ids,
            sample_count,
        ),
    }

    if any(value > 0 for key, value in subject_alignment.items() if key.endswith("_count")):
        warnings.append(f"{split_path.name}: subject_id coverage differs across CSV files.")

    photo_alignment: dict[str, Any] = {}
    for image_dir_name in EXPECTED_IMAGE_DIRS:
        image_id_set = image_basenames.get(image_dir_name, set())
        alignment = set_difference_summary(
            "photo_map",
            photo_map_photo_ids,
            image_dir_name,
            image_id_set,
            sample_count,
        )
        photo_alignment[image_dir_name] = alignment
        if any(value > 0 for key, value in alignment.items() if key.endswith("_count")):
            warnings.append(
                f"{split_path.name}: photo_id coverage differs between subject_to_photo_map.csv and '{image_dir_name}'."
            )

    mask_alignment = set_difference_summary(
        "mask",
        image_basenames.get("mask", set()),
        "mask_left",
        image_basenames.get("mask_left", set()),
        sample_count,
    )
    photo_alignment["mask_vs_mask_left"] = mask_alignment
    if any(value > 0 for key, value in mask_alignment.items() if key.endswith("_count")):
        warnings.append(f"{split_path.name}: 'mask' and 'mask_left' do not contain the same PNG basenames.")

    integrity_checks = {
        "expected_artifacts_present": {
            "csv_files": {
                csv_name: bool(csv_summaries[csv_name].get("exists", False))
                for csv_name in EXPECTED_CSV_FILES
            },
            "image_directories": {
                image_dir_name: bool(image_summaries[image_dir_name].get("exists", False))
                for image_dir_name in EXPECTED_IMAGE_DIRS
            },
        },
        "subject_id_alignment": subject_alignment,
        "photo_id_alignment": photo_alignment,
        "duplicates": {
            csv_name: {
                "duplicate_subject_id_rows": csv_summaries[csv_name].get(
                    "duplicate_subject_id_rows",
                    0,
                ),
                "duplicate_photo_id_rows": csv_summaries[csv_name].get(
                    "duplicate_photo_id_rows",
                    0,
                ),
                "duplicate_subject_photo_rows": csv_summaries[csv_name].get(
                    "duplicate_subject_photo_rows",
                    0,
                ),
            }
            for csv_name in EXPECTED_CSV_FILES
            if csv_summaries[csv_name].get("exists", False)
        }
        | {
            image_dir_name: {
                "duplicate_basename_files": image_summaries[image_dir_name].get(
                    "duplicate_basename_files",
                    0,
                )
            }
            for image_dir_name in EXPECTED_IMAGE_DIRS
            if image_summaries[image_dir_name].get("exists", False)
        },
    }

    split_summary = {
        "name": split_path.name,
        "relative_path": relative_path(split_path, dataset_root),
        "artifacts": {
            "csv_files": csv_summaries,
            "image_directories": image_summaries,
        },
        "mapping_summary": mapping_summary,
    }

    split_samples = {
        "csv_files": {
            csv_name: csv_summaries[csv_name].get("sample_paths", [])
            for csv_name in EXPECTED_CSV_FILES
        },
        "image_directories": {
            image_dir_name: image_summaries[image_dir_name].get("sample_paths", [])
            for image_dir_name in EXPECTED_IMAGE_DIRS
        },
        "image_dimensions": {
            image_dir_name: image_summaries[image_dir_name].get("sample_dimensions", [])
            for image_dir_name in EXPECTED_IMAGE_DIRS
        },
    }

    csv_schemas = {
        csv_name: csv_summaries[csv_name].get("columns", [])
        for csv_name in EXPECTED_CSV_FILES
    }

    return split_summary, csv_schemas, integrity_checks, split_samples, warnings


def inspect_dataset(dataset_root: Path, sample_count: int) -> dict[str, Any]:
    resolved_root = validate_dataset_root(dataset_root)
    split_paths = discover_splits(resolved_root)

    report: dict[str, Any] = {
        "dataset_root": str(resolved_root),
        "discovered_splits": [path.name for path in split_paths],
        "file_type_summary": summarize_file_types(resolved_root),
        "splits": [],
        "csv_schemas": {},
        "integrity_checks": {},
        "samples": {},
        "warnings": [],
    }

    for split_path in split_paths:
        (
            split_summary,
            csv_schemas,
            integrity_checks,
            split_samples,
            split_warnings,
        ) = inspect_split(
            split_path=split_path,
            dataset_root=resolved_root,
            sample_count=sample_count,
        )

        report["splits"].append(split_summary)
        report["csv_schemas"][split_path.name] = csv_schemas
        report["integrity_checks"][split_path.name] = integrity_checks
        report["samples"][split_path.name] = split_samples
        report["warnings"].extend(split_warnings)

    return report


def format_columns(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "<missing>"


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("BodyM Raw Dataset Inspection")
    lines.append("=" * 28)
    lines.append(f"Dataset root: {report['dataset_root']}")
    lines.append(
        "Discovered splits: "
        + ", ".join(report["discovered_splits"])
    )
    lines.append("")

    lines.append("Per-Split Artifact Summary")
    lines.append("-" * 26)
    for split_summary in report["splits"]:
        lines.append(f"[{split_summary['name']}] {split_summary['relative_path']}")

        csv_files = split_summary["artifacts"]["csv_files"]
        image_directories = split_summary["artifacts"]["image_directories"]

        for csv_name in EXPECTED_CSV_FILES:
            csv_summary = csv_files[csv_name]
            if not csv_summary.get("exists", False):
                lines.append(f"  CSV  {csv_name}: missing")
                continue
            lines.append(
                f"  CSV  {csv_name}: rows={csv_summary['row_count']}, "
                f"columns={len(csv_summary['columns'])}"
            )

        for image_dir_name in EXPECTED_IMAGE_DIRS:
            image_summary = image_directories[image_dir_name]
            if not image_summary.get("exists", False):
                lines.append(f"  DIR  {image_dir_name}: missing")
                continue
            lines.append(
                f"  DIR  {image_dir_name}: png_count={image_summary['png_count']}, "
                f"unique_basenames={image_summary['unique_basenames']}"
            )
        lines.append("")

    lines.append("File Type Summary")
    lines.append("-" * 17)
    for extension, count in report["file_type_summary"].items():
        lines.append(f"  {extension}: {count}")
    lines.append("")

    lines.append("CSV Schema Summary")
    lines.append("-" * 18)
    for split_name in report["discovered_splits"]:
        lines.append(f"[{split_name}]")
        csv_schemas = report["csv_schemas"][split_name]
        for csv_name in EXPECTED_CSV_FILES:
            lines.append(f"  {csv_name}: {format_columns(csv_schemas.get(csv_name, []))}")
        lines.append("")

    lines.append("Mapping Summary")
    lines.append("-" * 15)
    for split_summary in report["splits"]:
        mapping = split_summary["mapping_summary"]
        subject_counts = mapping["unique_subject_ids"]
        photos_per_subject_summary = mapping["photos_per_subject"]
        lines.append(
            f"[{split_summary['name']}] "
            f"subjects(hwg={subject_counts['hwg_metadata.csv']}, "
            f"measurements={subject_counts['measurements.csv']}, "
            f"photo_map={subject_counts['subject_to_photo_map.csv']}), "
            f"photos={mapping['unique_photo_ids']}, "
            f"photos_per_subject(min={photos_per_subject_summary['min']}, "
            f"max={photos_per_subject_summary['max']}, "
            f"avg={photos_per_subject_summary['avg']})"
        )
    lines.append("")

    lines.append("Integrity Checks")
    lines.append("-" * 16)
    for split_name in report["discovered_splits"]:
        checks = report["integrity_checks"][split_name]
        subject_alignment = checks["subject_id_alignment"]
        mask_alignment = checks["photo_id_alignment"]["mask"]
        mask_left_alignment = checks["photo_id_alignment"]["mask_left"]
        mask_vs_left = checks["photo_id_alignment"]["mask_vs_mask_left"]
        lines.append(
            f"[{split_name}] "
            f"subject_mismatches="
            f"{subject_alignment['hwg_minus_measurements_count'] + subject_alignment['measurements_minus_hwg_count'] + subject_alignment['hwg_minus_photo_map_count'] + subject_alignment['photo_map_minus_hwg_count']}, "
            f"photo_map_vs_mask_mismatches="
            f"{mask_alignment['photo_map_minus_mask_count'] + mask_alignment['mask_minus_photo_map_count']}, "
            f"photo_map_vs_mask_left_mismatches="
            f"{mask_left_alignment['photo_map_minus_mask_left_count'] + mask_left_alignment['mask_left_minus_photo_map_count']}, "
            f"mask_vs_mask_left_mismatches="
            f"{mask_vs_left['mask_minus_mask_left_count'] + mask_vs_left['mask_left_minus_mask_count']}"
        )
    lines.append("")

    lines.append("Sample Relative File Paths")
    lines.append("-" * 26)
    for split_name in report["discovered_splits"]:
        lines.append(f"[{split_name}]")
        samples = report["samples"][split_name]
        for csv_name in EXPECTED_CSV_FILES:
            sample_list = samples["csv_files"].get(csv_name, [])
            lines.append(f"  {csv_name}: {sample_list or ['<missing>']}")
        for image_dir_name in EXPECTED_IMAGE_DIRS:
            sample_list = samples["image_directories"].get(image_dir_name, [])
            lines.append(f"  {image_dir_name}: {sample_list or ['<missing>']}")
        lines.append("")

    lines.append("Sampled Image Dimensions")
    lines.append("-" * 24)
    for split_name in report["discovered_splits"]:
        lines.append(f"[{split_name}]")
        dimensions = report["samples"][split_name]["image_dimensions"]
        for image_dir_name in EXPECTED_IMAGE_DIRS:
            sample_dimensions = dimensions.get(image_dir_name, [])
            if not sample_dimensions:
                lines.append(f"  {image_dir_name}: <missing>")
                continue
            formatted_dimensions = ", ".join(
                f"{entry['width']}x{entry['height']} ({Path(entry['file']).name})"
                for entry in sample_dimensions
            )
            lines.append(f"  {image_dir_name}: {formatted_dimensions}")
        lines.append("")

    lines.append("Warnings And Anomalies")
    lines.append("-" * 22)
    warnings = report["warnings"]
    if warnings:
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  None")

    return "\n".join(lines)


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise InspectionError(f"Failed to write JSON report to: {output_path}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        report = inspect_dataset(
            dataset_root=args.dataset_root,
            sample_count=args.sample_count,
        )
        print(format_report(report))

        if args.output_json is not None:
            write_json_report(report, args.output_json)
            print(f"\nSaved JSON report to: {args.output_json.resolve()}")
    except InspectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
