from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Error: pandas is required to run scripts/build_training_splits.py. Install pandas and retry."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = (
    "split",
    "subject_id",
    "subject_key",
    "photo_id",
    "mask_path",
    "mask_left_path",
    "hwg_gender",
    "hwg_height_cm",
    "hwg_weight_kg",
)
HWG_NUMERIC_COLUMNS: tuple[str, ...] = ("hwg_height_cm", "hwg_weight_kg")
DEFAULT_MANIFEST_PATH = Path("data/interim/bodym_manifest.csv")
DEFAULT_TRAIN_OUTPUT = Path("data/interim/bodym_training_train.csv")
DEFAULT_VAL_OUTPUT = Path("data/interim/bodym_training_val.csv")
DEFAULT_SUMMARY_OUTPUT = Path("data/interim/bodym_training_split_summary.json")


class TrainingSplitError(RuntimeError):
    """Raised when training split artifacts cannot be built safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build subject-safe train/val split artifacts from the BodyM manifest."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the canonical BodyM manifest CSV.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help="Path to the output train split CSV.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=DEFAULT_VAL_OUTPUT,
        help="Path to the output validation split CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path to the output summary JSON.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Validation holdout ratio from source train subjects.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subject split assignment.",
    )
    args = parser.parse_args(argv)

    if not 0 < args.val_size < 1:
        parser.error("--val-size must be greater than 0 and less than 1.")

    return args


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def load_manifest_frame(manifest_path: Path) -> pd.DataFrame:
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
        raise TrainingSplitError(f"Failed to read manifest file: {manifest_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise TrainingSplitError(f"Manifest file is empty: {manifest_path}") from exc
    except pd.errors.ParserError as exc:
        raise TrainingSplitError(f"Failed to parse manifest file: {manifest_path}") from exc

    frame.columns = [str(column).strip() for column in frame.columns.tolist()]
    missing_columns = [
        column for column in MANIFEST_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise TrainingSplitError(
            f"Manifest is missing required columns: {missing_columns}"
        )

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
        raise TrainingSplitError(
            "Manifest contains empty required values: " + ", ".join(empty_counts)
        )

    return frame


def get_trainable_target_columns(frame: pd.DataFrame) -> list[str]:
    target_columns = [
        column for column in frame.columns if str(column).startswith("measurement_")
    ]
    if not target_columns:
        raise TrainingSplitError(
            "Manifest does not contain any trainable target columns with the 'measurement_' prefix."
        )
    return target_columns


def validate_numeric_columns(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    label: str,
) -> None:
    for column in columns:
        string_values = frame[column].astype("string")
        empty_mask = string_values.isna() | (string_values.str.strip() == "")
        empty_count = int(empty_mask.sum())
        if empty_count > 0:
            raise TrainingSplitError(f"{label} has empty values in column '{column}': {empty_count}")

        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        invalid_mask = numeric_values.isna() & ~empty_mask
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            raise TrainingSplitError(
                f"{label} contains non-numeric values in column '{column}': {invalid_count}"
            )
        frame[column] = numeric_values.astype("float32")


def validate_target_columns(frame: pd.DataFrame, target_columns: list[str]) -> None:
    validate_numeric_columns(frame, target_columns, "Manifest target columns")


def validate_subject_level_consistency(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    subject_column: str = "subject_key",
) -> None:
    for column in columns:
        inconsistent_subjects = (
            frame.groupby(subject_column)[column].nunique(dropna=False)
        )
        inconsistent_subjects = inconsistent_subjects[inconsistent_subjects > 1]
        if not inconsistent_subjects.empty:
            examples = inconsistent_subjects.index.tolist()[:5]
            raise TrainingSplitError(
                f"Column '{column}' varies within {subject_column}. Example subject keys: {examples}"
            )


def build_subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    subject_columns = ["subject_key", "hwg_gender", "hwg_height_cm", "hwg_weight_kg"]
    subject_frame = frame.loc[:, subject_columns].drop_duplicates().copy()
    if subject_frame["subject_key"].duplicated().any():
        raise TrainingSplitError("Subject table contains duplicate subject_key rows.")
    if len(subject_frame.index) < 2:
        raise TrainingSplitError("At least two unique train subjects are required to build train/val splits.")
    return subject_frame.reset_index(drop=True)


def resolve_validation_subject_count(subject_count: int, val_size: float) -> int:
    requested = int(round(subject_count * val_size))
    requested = max(1, requested)
    requested = min(subject_count - 1, requested)
    if requested <= 0:
        raise TrainingSplitError("Validation split configuration produced zero validation subjects.")
    return requested


def can_stratify(group_counts: dict[str, int], validation_count: int) -> bool:
    if len(group_counts) < 2:
        return False
    if min(group_counts.values()) < 2:
        return False
    if validation_count < len(group_counts):
        return False
    train_count = sum(group_counts.values()) - validation_count
    return train_count >= len(group_counts)


def allocate_group_validation_counts(
    group_counts: dict[str, int],
    validation_count: int,
) -> dict[str, int]:
    total_subjects = sum(group_counts.values())
    base_counts = {
        label: int(math.floor(count * validation_count / total_subjects))
        for label, count in group_counts.items()
    }

    for label, count in group_counts.items():
        if base_counts[label] > count - 1:
            base_counts[label] = count - 1

    if validation_count >= len(group_counts):
        for label, count in group_counts.items():
            if count > 1 and base_counts[label] == 0:
                base_counts[label] = 1

    current_total = sum(base_counts.values())
    if current_total > validation_count:
        ordered_labels = sorted(
            group_counts,
            key=lambda label: (
                group_counts[label] * validation_count / total_subjects - base_counts[label],
                group_counts[label],
                label,
            ),
        )
        for label in ordered_labels:
            while current_total > validation_count and base_counts[label] > 0:
                base_counts[label] -= 1
                current_total -= 1

    if current_total < validation_count:
        ordered_labels = sorted(
            group_counts,
            key=lambda label: (
                group_counts[label] * validation_count / total_subjects - base_counts[label],
                group_counts[label],
                label,
            ),
            reverse=True,
        )
        for label in ordered_labels:
            while current_total < validation_count and base_counts[label] < group_counts[label] - 1:
                base_counts[label] += 1
                current_total += 1

    if sum(base_counts.values()) != validation_count:
        raise TrainingSplitError("Failed to allocate stratified validation subject counts.")

    return base_counts


def sample_subject_keys(
    subject_frame: pd.DataFrame,
    validation_count: int,
    random_state: int,
) -> tuple[set[str], bool]:
    group_counts = subject_frame["hwg_gender"].value_counts().to_dict()
    rng = random.Random(random_state)

    if can_stratify(group_counts, validation_count):
        group_validation_counts = allocate_group_validation_counts(
            group_counts,
            validation_count,
        )
        validation_subjects: list[str] = []
        for gender, group_frame in subject_frame.groupby("hwg_gender", sort=True):
            subject_keys = sorted(group_frame["subject_key"].tolist())
            rng.shuffle(subject_keys)
            validation_subjects.extend(subject_keys[:group_validation_counts[str(gender)]])
        return set(validation_subjects), True

    subject_keys = sorted(subject_frame["subject_key"].tolist())
    rng.shuffle(subject_keys)
    return set(subject_keys[:validation_count]), False


def build_subject_split_assignments(
    train_frame: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subject_frame = build_subject_table(train_frame)
    validation_count = resolve_validation_subject_count(len(subject_frame.index), val_size)
    validation_subjects, stratified = sample_subject_keys(
        subject_frame,
        validation_count,
        random_state,
    )

    assignments = subject_frame.loc[:, ["subject_key"]].copy()
    assignments["split"] = assignments["subject_key"].map(
        lambda subject_key: "val" if subject_key in validation_subjects else "train"
    )

    if assignments["split"].eq("val").sum() == 0:
        raise TrainingSplitError("Validation split assignment produced zero validation subjects.")
    if assignments["split"].eq("train").sum() == 0:
        raise TrainingSplitError("Validation split assignment produced zero training subjects.")

    summary = {
        "stratified": stratified,
        "stratify_column": "hwg_gender",
        "validation_subject_count": int(assignments["split"].eq("val").sum()),
        "training_subject_count": int(assignments["split"].eq("train").sum()),
        "source_subject_count": int(len(assignments.index)),
    }
    return assignments, summary


def build_output_frame(
    source_train_frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    split_map = assignments.set_index("subject_key")["split"]
    output_frame = source_train_frame.copy()
    output_frame.insert(1, "source_split", output_frame["split"])
    output_frame["split"] = output_frame["subject_key"].map(split_map)

    if output_frame["split"].isna().any():
        raise TrainingSplitError("Some source train rows were not assigned to a workflow split.")

    output_columns = ["split", "source_split"] + [
        column for column in source_train_frame.columns if column != "split"
    ]
    return output_frame.loc[:, output_columns].copy()


def target_summary(frame: pd.DataFrame, target_columns: list[str]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for column in target_columns:
        summary[column] = {
            "missing_count": int(frame[column].isna().sum()),
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
    return summary


def subset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    subject_count = int(frame["subject_key"].nunique())
    row_count = int(len(frame.index))
    gender_counts = (
        frame.loc[:, ["subject_key", "hwg_gender"]]
        .drop_duplicates()["hwg_gender"]
        .value_counts(sort=False)
        .to_dict()
    )
    return {
        "row_count": row_count,
        "subject_count": subject_count,
        "gender_counts": {str(key): int(value) for key, value in gender_counts.items()},
    }


def build_training_split_artifacts(
    manifest_path: Path,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_frame = load_manifest_frame(manifest_path)
    target_columns = get_trainable_target_columns(manifest_frame)
    validate_numeric_columns(manifest_frame, list(HWG_NUMERIC_COLUMNS), "Manifest HWG columns")
    validate_target_columns(manifest_frame, target_columns)
    validate_subject_level_consistency(
        manifest_frame,
        list(HWG_NUMERIC_COLUMNS) + ["hwg_gender"] + target_columns,
    )

    source_split_row_counts = {
        str(split): int(count)
        for split, count in manifest_frame["split"].value_counts(sort=False).items()
    }
    source_split_subject_counts = {
        str(split): int(count)
        for split, count in manifest_frame.groupby("split")["subject_key"].nunique().items()
    }

    source_train_frame = manifest_frame.loc[manifest_frame["split"] == "train"].copy()
    if source_train_frame.empty:
        raise TrainingSplitError("Manifest does not contain any source train rows.")

    assignments, assignment_summary = build_subject_split_assignments(
        source_train_frame,
        val_size=val_size,
        random_state=random_state,
    )
    output_frame = build_output_frame(source_train_frame, assignments)

    train_frame = output_frame.loc[output_frame["split"] == "train"].copy()
    val_frame = output_frame.loc[output_frame["split"] == "val"].copy()

    train_subjects = set(train_frame["subject_key"].unique().tolist())
    val_subjects = set(val_frame["subject_key"].unique().tolist())
    if train_subjects & val_subjects:
        raise TrainingSplitError("Train and validation subject sets overlap.")

    if len(train_frame.index) + len(val_frame.index) != len(source_train_frame.index):
        raise TrainingSplitError("Train/val outputs do not cover all source train rows exactly once.")

    holdout_summary = {
        split_name: {
            "row_count": source_split_row_counts[split_name],
            "subject_count": source_split_subject_counts[split_name],
        }
        for split_name in source_split_row_counts
        if split_name != "train"
    }

    summary = {
        "manifest_path": str(manifest_path),
        "trainable_target_columns": target_columns,
        "target_summary": target_summary(manifest_frame, target_columns),
        "source_manifest": {
            "row_count": int(len(manifest_frame.index)),
            "split_row_counts": source_split_row_counts,
            "split_subject_counts": source_split_subject_counts,
        },
        "source_train": subset_summary(source_train_frame),
        "split_config": {
            "subject_column": "subject_key",
            "source_split_filter": "train",
            "val_size": float(val_size),
            "random_state": int(random_state),
            **assignment_summary,
        },
        "outputs": {
            "train": subset_summary(train_frame),
            "val": subset_summary(val_frame),
        },
        "holdouts": holdout_summary,
    }

    return train_frame, val_frame, summary


def write_split_frame(frame: pd.DataFrame, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    except OSError as exc:
        raise TrainingSplitError(f"Failed to write split CSV: {output_path}") from exc


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise TrainingSplitError(f"Failed to write split summary JSON: {output_path}") from exc


def format_summary(
    manifest_path: Path,
    train_output: Path,
    val_output: Path,
    summary_output: Path,
    summary: dict[str, Any],
) -> str:
    lines = [
        "BodyM Training Split Build",
        "==========================",
        f"Manifest path: {manifest_path}",
        f"Train output: {train_output}",
        f"Val output: {val_output}",
        f"Summary output: {summary_output}",
        "",
        f"Trainable targets: {len(summary['trainable_target_columns'])}",
        f"Source train rows: {summary['source_train']['row_count']}",
        f"Source train subjects: {summary['source_train']['subject_count']}",
        f"Stratified by hwg_gender: {summary['split_config']['stratified']}",
        "",
        "Generated Splits",
        "----------------",
        f"train: rows={summary['outputs']['train']['row_count']}, subjects={summary['outputs']['train']['subject_count']}",
        f"val: rows={summary['outputs']['val']['row_count']}, subjects={summary['outputs']['val']['subject_count']}",
        "",
        "Holdouts",
        "--------",
    ]

    if summary["holdouts"]:
        for split_name, split_summary in summary["holdouts"].items():
            lines.append(
                f"{split_name}: rows={split_summary['row_count']}, subjects={split_summary['subject_count']}"
            )
    else:
        lines.append("None")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        manifest_path = resolve_repo_path(args.manifest_path)
        train_output = resolve_repo_path(args.train_output)
        val_output = resolve_repo_path(args.val_output)
        summary_output = resolve_repo_path(args.summary_output)

        train_frame, val_frame, summary = build_training_split_artifacts(
            manifest_path=manifest_path,
            val_size=args.val_size,
            random_state=args.random_seed,
        )
        write_split_frame(train_frame, train_output)
        write_split_frame(val_frame, val_output)
        write_summary(summary, summary_output)
        print(
            format_summary(
                manifest_path=manifest_path,
                train_output=train_output,
                val_output=val_output,
                summary_output=summary_output,
                summary=summary,
            )
        )
    except (FileNotFoundError, TrainingSplitError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
