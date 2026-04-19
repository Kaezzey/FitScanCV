from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.models.bodym_training import (
    TrainingPipelineError,
    evaluate_checkpoint,
    load_experiment_config,
    resolve_repo_path,
    save_json,
)


def _ordered_target_names(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    metric_name: str,
) -> tuple[str, ...]:
    baseline_names = tuple(baseline_metrics.keys())
    candidate_names = tuple(candidate_metrics.keys())
    if baseline_names != candidate_names:
        raise TrainingPipelineError(
            f"Baseline and candidate {metric_name} targets do not match."
        )
    return baseline_names


def _compute_metric_deltas(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    metric_name: str,
) -> dict[str, float]:
    target_names = _ordered_target_names(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        metric_name=metric_name,
    )
    return {
        name: float(candidate_metrics[name] - baseline_metrics[name])
        for name in target_names
    }


def _default_output_dir(candidate_checkpoint_path: str | Path, split: str) -> Path:
    return resolve_repo_path(candidate_checkpoint_path).resolve().parent / "benchmarks" / split


def write_per_target_delta_table(report: dict[str, Any], output_path: str | Path) -> Path:
    resolved_output_path = resolve_repo_path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    target_names = tuple(report["deltas"]["per_target_mae"].keys())
    fieldnames = [
        "target_name",
        "baseline_mae",
        "candidate_mae",
        "delta_mae_candidate_minus_baseline",
        "baseline_rmse",
        "candidate_rmse",
        "delta_rmse_candidate_minus_baseline",
    ]

    try:
        with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for target_name in target_names:
                writer.writerow(
                    {
                        "target_name": target_name,
                        "baseline_mae": report["baseline"]["per_target_mae"][target_name],
                        "candidate_mae": report["candidate"]["per_target_mae"][target_name],
                        "delta_mae_candidate_minus_baseline": report["deltas"][
                            "per_target_mae"
                        ][target_name],
                        "baseline_rmse": report["baseline"]["per_target_rmse"][target_name],
                        "candidate_rmse": report["candidate"]["per_target_rmse"][target_name],
                        "delta_rmse_candidate_minus_baseline": report["deltas"][
                            "per_target_rmse"
                        ][target_name],
                    }
                )
    except OSError as exc:
        raise TrainingPipelineError(
            f"Failed to write per-target delta table: {resolved_output_path}"
        ) from exc

    return resolved_output_path


def benchmark_checkpoints(
    *,
    baseline_config_path: str | Path,
    baseline_checkpoint_path: str | Path,
    candidate_config_path: str | Path,
    candidate_checkpoint_path: str | Path,
    manifest_path: str | Path | None = None,
    split: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    baseline_config = load_experiment_config(baseline_config_path)
    candidate_config = load_experiment_config(candidate_config_path)

    resolved_manifest_path = (
        resolve_repo_path(manifest_path)
        if manifest_path is not None
        else baseline_config.data.val_manifest_path
    )
    resolved_split = split or baseline_config.data.val_split
    resolved_output_dir = (
        resolve_repo_path(output_dir)
        if output_dir is not None
        else _default_output_dir(candidate_checkpoint_path, resolved_split)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = evaluate_checkpoint(
        config=baseline_config,
        checkpoint_path=baseline_checkpoint_path,
        manifest_path_override=resolved_manifest_path,
        split_override=resolved_split,
    )
    candidate_summary = evaluate_checkpoint(
        config=candidate_config,
        checkpoint_path=candidate_checkpoint_path,
        manifest_path_override=resolved_manifest_path,
        split_override=resolved_split,
    )

    per_target_mae_deltas = _compute_metric_deltas(
        baseline_metrics=baseline_summary["per_target_mae"],
        candidate_metrics=candidate_summary["per_target_mae"],
        metric_name="per_target_mae",
    )
    per_target_rmse_deltas = _compute_metric_deltas(
        baseline_metrics=baseline_summary["per_target_rmse"],
        candidate_metrics=candidate_summary["per_target_rmse"],
        metric_name="per_target_rmse",
    )

    report = {
        "comparison_basis": "candidate_minus_baseline",
        "manifest_path": str(resolved_manifest_path),
        "split": resolved_split,
        "baseline": {
            "config_path": str(resolve_repo_path(baseline_config_path)),
            "checkpoint_path": baseline_summary["checkpoint_path"],
            "loss": float(baseline_summary["loss"]),
            "mean_mae": float(baseline_summary["mean_mae"]),
            "mean_rmse": float(baseline_summary["mean_rmse"]),
            "per_target_mae": baseline_summary["per_target_mae"],
            "per_target_rmse": baseline_summary["per_target_rmse"],
        },
        "candidate": {
            "config_path": str(resolve_repo_path(candidate_config_path)),
            "checkpoint_path": candidate_summary["checkpoint_path"],
            "loss": float(candidate_summary["loss"]),
            "mean_mae": float(candidate_summary["mean_mae"]),
            "mean_rmse": float(candidate_summary["mean_rmse"]),
            "per_target_mae": candidate_summary["per_target_mae"],
            "per_target_rmse": candidate_summary["per_target_rmse"],
        },
        "deltas": {
            "loss": float(candidate_summary["loss"] - baseline_summary["loss"]),
            "mean_mae": float(candidate_summary["mean_mae"] - baseline_summary["mean_mae"]),
            "mean_rmse": float(
                candidate_summary["mean_rmse"] - baseline_summary["mean_rmse"]
            ),
            "per_target_mae": per_target_mae_deltas,
            "per_target_rmse": per_target_rmse_deltas,
        },
        "winner_by_mean_mae": (
            "candidate"
            if candidate_summary["mean_mae"] < baseline_summary["mean_mae"]
            else "baseline"
        ),
    }

    report_output_path = resolved_output_dir / f"comparison_report_{resolved_split}.json"
    table_output_path = resolved_output_dir / f"per_target_deltas_{resolved_split}.csv"
    save_json(report, report_output_path)
    write_per_target_delta_table(report, table_output_path)

    return {
        **report,
        "report_output_path": str(report_output_path),
        "table_output_path": str(table_output_path),
    }


__all__ = ["benchmark_checkpoints", "write_per_target_delta_table"]
