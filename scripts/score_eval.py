#!/usr/bin/env python3
"""Score Raysurfer one-shot eval logs with a 3-minute consistency metric."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    """A single eval task descriptor."""

    task_id: str
    title: str


@dataclass(frozen=True)
class RunRecord:
    """A single trial outcome for one task."""

    task_id: str
    completed: bool
    elapsed_seconds: float


@dataclass
class TaskAggregate:
    """Aggregated stats for a task across trials."""

    attempts: int = 0
    completed: int = 0
    completed_within_sla: int = 0

    def consistency(self) -> float:
        """Return completed-within-SLA ratio."""
        if self.attempts == 0:
            return 0.0
        return self.completed_within_sla / self.attempts


@dataclass
class ScoreReport:
    """Whole-run score details."""

    label: str
    per_task: dict[str, TaskAggregate]
    total_attempts: int
    total_completed: int
    total_completed_within_sla: int
    tasks_with_attempts: int

    def overall_consistency(self) -> float:
        """Return weighted overall consistency across all attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_completed_within_sla / self.total_attempts


def load_tasks(path: Path) -> dict[str, TaskSpec]:
    """Load task metadata from tasks JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list):
        raise ValueError("tasks file must include a 'tasks' array")

    tasks: dict[str, TaskSpec] = {}
    for item in tasks_raw:
        if not isinstance(item, dict):
            raise ValueError("each task must be an object")
        task_id = item.get("id")
        title = item.get("title")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("each task must have a non-empty string 'id'")
        if not isinstance(title, str) or not title:
            raise ValueError(f"task {task_id!r} must have a non-empty string 'title'")
        if task_id in tasks:
            raise ValueError(f"duplicate task id found: {task_id}")
        tasks[task_id] = TaskSpec(task_id=task_id, title=title)

    if not tasks:
        raise ValueError("tasks file contains no tasks")
    return tasks


def load_run_records(path: Path, valid_task_ids: set[str]) -> tuple[str, list[RunRecord]]:
    """Load a run log file and validate its records."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    label = raw.get("label")
    runs = raw.get("runs")

    if not isinstance(label, str) or not label:
        label = path.stem
    if not isinstance(runs, list):
        raise ValueError(f"{path}: run log must include a 'runs' array")

    records: list[RunRecord] = []
    for index, item in enumerate(runs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: run #{index} is not an object")

        task_id = item.get("task_id")
        completed = item.get("completed")
        elapsed_seconds = item.get("elapsed_seconds")

        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path}: run #{index} missing valid 'task_id'")
        if task_id not in valid_task_ids:
            raise ValueError(f"{path}: run #{index} has unknown task_id {task_id!r}")
        if not isinstance(completed, bool):
            raise ValueError(f"{path}: run #{index} must include boolean 'completed'")
        if not isinstance(elapsed_seconds, (int, float)):
            raise ValueError(f"{path}: run #{index} must include numeric 'elapsed_seconds'")
        if elapsed_seconds < 0:
            raise ValueError(f"{path}: run #{index} has negative 'elapsed_seconds'")

        records.append(
            RunRecord(
                task_id=task_id,
                completed=completed,
                elapsed_seconds=float(elapsed_seconds),
            )
        )

    return label, records


def compute_report(
    *,
    label: str,
    tasks: dict[str, TaskSpec],
    runs: list[RunRecord],
    sla_seconds: float,
) -> ScoreReport:
    """Aggregate run records into a score report."""
    per_task: dict[str, TaskAggregate] = {
        task_id: TaskAggregate() for task_id in tasks
    }

    for run in runs:
        aggregate = per_task[run.task_id]
        aggregate.attempts += 1
        if run.completed:
            aggregate.completed += 1
            if run.elapsed_seconds <= sla_seconds:
                aggregate.completed_within_sla += 1

    total_attempts = sum(item.attempts for item in per_task.values())
    total_completed = sum(item.completed for item in per_task.values())
    total_completed_within_sla = sum(item.completed_within_sla for item in per_task.values())
    tasks_with_attempts = sum(1 for item in per_task.values() if item.attempts > 0)

    return ScoreReport(
        label=label,
        per_task=per_task,
        total_attempts=total_attempts,
        total_completed=total_completed,
        total_completed_within_sla=total_completed_within_sla,
        tasks_with_attempts=tasks_with_attempts,
    )


def print_report(report: ScoreReport, tasks: dict[str, TaskSpec], sla_seconds: float) -> None:
    """Print a human-readable score summary."""
    total_tasks = len(tasks)
    overall = report.overall_consistency()

    print(f"\\n=== {report.label} ===")
    print(f"overall_consistency_within_{int(sla_seconds)}s: {overall:.2%}")
    print(
        "completed_within_sla/attempts: "
        f"{report.total_completed_within_sla}/{report.total_attempts}"
    )
    print(
        f"task_coverage: {report.tasks_with_attempts}/{total_tasks} "
        f"({(report.tasks_with_attempts / total_tasks):.2%})"
    )

    print("\\nPer-task consistency:")
    print("task_id | attempts | within_sla | consistency | title")
    print("--- | ---: | ---: | ---: | ---")

    for task_id in sorted(tasks):
        task = tasks[task_id]
        aggregate = report.per_task[task_id]
        print(
            f"{task_id} | {aggregate.attempts} | {aggregate.completed_within_sla} "
            f"| {aggregate.consistency():.2%} | {task.title}"
        )


def print_comparison(
    baseline: ScoreReport,
    raysurfer: ScoreReport,
    tasks: dict[str, TaskSpec],
    sla_seconds: float,
) -> None:
    """Print baseline-vs-Raysurfer delta metrics."""
    baseline_overall = baseline.overall_consistency()
    raysurfer_overall = raysurfer.overall_consistency()
    delta = raysurfer_overall - baseline_overall

    print("\\n=== Comparison ===")
    print(
        f"baseline_consistency_within_{int(sla_seconds)}s: {baseline_overall:.2%}"
    )
    print(
        f"raysurfer_consistency_within_{int(sla_seconds)}s: {raysurfer_overall:.2%}"
    )
    print(f"delta: {delta:+.2%}")

    print("\\nPer-task delta (Raysurfer - Baseline):")
    print("task_id | baseline | raysurfer | delta | title")
    print("--- | ---: | ---: | ---: | ---")

    for task_id in sorted(tasks):
        title = tasks[task_id].title
        base_score = baseline.per_task[task_id].consistency()
        rs_score = raysurfer.per_task[task_id].consistency()
        task_delta = rs_score - base_score
        print(
            f"{task_id} | {base_score:.2%} | {rs_score:.2%} "
            f"| {task_delta:+.2%} | {title}"
        )


def maybe_write_json(
    *,
    output_path: Path,
    tasks: dict[str, TaskSpec],
    raysurfer: ScoreReport,
    baseline: ScoreReport | None,
    sla_seconds: float,
) -> None:
    """Write machine-readable summary output."""
    result: dict[str, object] = {
        "sla_seconds": sla_seconds,
        "raysurfer": {
            "label": raysurfer.label,
            "overall_consistency": raysurfer.overall_consistency(),
            "total_attempts": raysurfer.total_attempts,
            "total_completed": raysurfer.total_completed,
            "total_completed_within_sla": raysurfer.total_completed_within_sla,
        },
        "per_task": {},
    }

    per_task: dict[str, object] = {}
    for task_id in sorted(tasks):
        rs_aggregate = raysurfer.per_task[task_id]
        item: dict[str, object] = {
            "title": tasks[task_id].title,
            "raysurfer_consistency": rs_aggregate.consistency(),
            "raysurfer_attempts": rs_aggregate.attempts,
            "raysurfer_within_sla": rs_aggregate.completed_within_sla,
        }

        if baseline is not None:
            base_aggregate = baseline.per_task[task_id]
            item["baseline_consistency"] = base_aggregate.consistency()
            item["baseline_attempts"] = base_aggregate.attempts
            item["baseline_within_sla"] = base_aggregate.completed_within_sla
            item["delta"] = rs_aggregate.consistency() - base_aggregate.consistency()

        per_task[task_id] = item

    result["per_task"] = per_task

    if baseline is not None:
        result["baseline"] = {
            "label": baseline.label,
            "overall_consistency": baseline.overall_consistency(),
            "total_attempts": baseline.total_attempts,
            "total_completed": baseline.total_completed,
            "total_completed_within_sla": baseline.total_completed_within_sla,
            "delta": raysurfer.overall_consistency() - baseline.overall_consistency(),
        }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Build CLI args."""
    parser = argparse.ArgumentParser(
        description="Score 3-minute consistency for Raysurfer one-shot eval runs"
    )
    parser.add_argument(
        "--tasks",
        required=True,
        type=Path,
        help="Path to tasks.json",
    )
    parser.add_argument(
        "--raysurfer-runs",
        required=True,
        type=Path,
        help="Run log for the Raysurfer-enabled mode",
    )
    parser.add_argument(
        "--baseline-runs",
        type=Path,
        help="Optional run log for baseline mode (no Raysurfer)",
    )
    parser.add_argument(
        "--sla-seconds",
        type=float,
        default=180.0,
        help="SLA threshold in seconds (default: 180)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for machine-readable summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Run the eval scorer."""
    args = parse_args()

    tasks = load_tasks(args.tasks)
    task_ids = set(tasks.keys())

    raysurfer_label, raysurfer_runs = load_run_records(args.raysurfer_runs, task_ids)
    raysurfer_report = compute_report(
        label=raysurfer_label,
        tasks=tasks,
        runs=raysurfer_runs,
        sla_seconds=args.sla_seconds,
    )

    baseline_report: ScoreReport | None = None
    if args.baseline_runs is not None:
        baseline_label, baseline_runs = load_run_records(args.baseline_runs, task_ids)
        baseline_report = compute_report(
            label=baseline_label,
            tasks=tasks,
            runs=baseline_runs,
            sla_seconds=args.sla_seconds,
        )

    print_report(raysurfer_report, tasks, args.sla_seconds)
    if baseline_report is not None:
        print_report(baseline_report, tasks, args.sla_seconds)
        print_comparison(baseline_report, raysurfer_report, tasks, args.sla_seconds)

    if args.json_out is not None:
        maybe_write_json(
            output_path=args.json_out,
            tasks=tasks,
            raysurfer=raysurfer_report,
            baseline=baseline_report,
            sla_seconds=args.sla_seconds,
        )
        print(f"\\nWrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
