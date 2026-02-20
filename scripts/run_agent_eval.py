#!/usr/bin/env python3
"""Run one-shot eval tasks with and without Raysurfer and emit run logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import py_compile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, ToolUseBlock
from raysurfer import AsyncRaySurfer


Mode = Literal["baseline", "raysurfer"]


@dataclass(frozen=True)
class EvalTask:
    """A single benchmark task."""

    task_id: str
    title: str
    one_shot_prompt: str


@dataclass
class TaskRunResult:
    """A measured outcome for one task attempt."""

    task_id: str
    trial: int
    completed: bool
    elapsed_seconds: float
    timestamp_utc: str
    details: str


def load_env_from_file(path: Path) -> None:
    """Load environment variables from a dotenv-like file if present."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    """Load local and workspace-level .env files."""
    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parents[1] / ".env",
        script_path.parents[3] / ".env",
    ]
    for candidate in candidates:
        load_env_from_file(candidate)


def load_tasks(path: Path, limit: int | None = None) -> list[EvalTask]:
    """Load task definitions from tasks JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks file must include a 'tasks' array")

    tasks: list[EvalTask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise ValueError("each task must be an object")
        task_id = item.get("id")
        title = item.get("title")
        one_shot_prompt = item.get("one_shot_prompt")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task missing string 'id'")
        if not isinstance(title, str) or not title:
            raise ValueError(f"{task_id}: missing string 'title'")
        if not isinstance(one_shot_prompt, str) or not one_shot_prompt:
            raise ValueError(f"{task_id}: missing string 'one_shot_prompt'")

        tasks.append(EvalTask(task_id=task_id, title=title, one_shot_prompt=one_shot_prompt))

    if limit is not None:
        return tasks[:limit]
    return tasks


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 Z format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_prompt(task: EvalTask) -> str:
    """Create a deterministic execution prompt from the task spec."""
    file_name = f"{task.task_id.lower()}_solution.py"
    return (
        "Benchmark key: RS_ONESHOT_EVAL_2026_02_20_V1\\n"
        f"Benchmark task id: {task.task_id}\\n\\n"
        "Implement this exactly once and avoid back-and-forth.\\n\\n"
        f"Task: {task.one_shot_prompt}\\n\\n"
        "Constraints:\\n"
        "- Write Python code only.\\n"
        "- Save everything into one file.\\n"
        f"- File path: {file_name}\\n"
        "- If the file already exists, reuse it and avoid rewriting unless compile fails.\\n"
        "- The file must include the literal marker: # VERIFIED_EVAL_SNIPPET\\n"
        "- The file must be at least 250 lines long.\\n"
        "- Keep code deterministic and production-style.\\n"
        "- Include type hints.\\n"
        "- Run this exact validation command and pass it before finishing:\\n"
        "  python - <<'PY'\\n"
        f"from pathlib import Path\\ntext = Path('{file_name}').read_text()\\n"
        "assert '# VERIFIED_EVAL_SNIPPET' in text\\n"
        "assert len(text.splitlines()) >= 250\\n"
        "print('validated')\\n"
        "PY\\n"
        "- Run `python -m py_compile <file>` and fix compile errors before stopping.\\n"
        "- After completion, reply with exactly: DONE\\n"
    )


def build_options(mode: Mode, task_workdir: Path, model: str, max_turns: int) -> ClaudeAgentOptions:
    """Build common agent options for both modes."""
    return ClaudeAgentOptions(
        allowed_tools=["Write", "Read", "Bash", "Edit"],
        permission_mode="bypassPermissions",
        model=model,
        max_turns=max_turns,
        cwd=str(task_workdir),
        system_prompt=(
            "You are running a timed coding benchmark."
            " Complete the task in one pass and stop as soon as it compiles."
        ),
    )


async def run_baseline_task(task: EvalTask, options: ClaudeAgentOptions) -> tuple[bool, int, str]:
    """Run one task with plain Claude Agent SDK."""
    tool_calls = 0
    final_status = "no_result"

    saw_result = False
    result_success = False

    async with ClaudeSDKClient(options=options) as client:
        await client.query(build_prompt(task))

        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        tool_calls += 1
            elif isinstance(msg, ResultMessage):
                saw_result = True
                final_status = msg.subtype
                result_success = msg.subtype == "success"

    return (saw_result and result_success), tool_calls, final_status


async def run_raysurfer_task(task: EvalTask, task_workdir: Path) -> tuple[bool, int, str, int]:
    """Run one task via direct Raysurfer retrieval + local execution checks."""
    api_key = os.getenv("RAYSURFER_API_KEY")
    base_url = os.getenv("RAYSURFER_BASE_URL", "https://api.raysurfer.com")
    if not api_key:
        return False, 0, "missing_api_key", 0

    query_text = build_prompt(task)
    async with AsyncRaySurfer(api_key=api_key, base_url=base_url) as rs:
        response = await rs.get_code_files(
            task=query_text,
            top_k=3,
            min_verdict_score=0.0,
            prefer_complete=False,
            cache_dir=str(task_workdir / ".raysurfer_code"),
        )

    if not response.files:
        return False, 0, "cache_miss", 0

    written_paths: list[Path] = []
    for index, file in enumerate(response.files, start=1):
        file_path = task_workdir / f"{index:02d}_{file.filename}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file.source, encoding="utf-8")
        written_paths.append(file_path)

    try:
        for file_path in written_paths:
            if file_path.suffix == ".py":
                py_compile.compile(str(file_path), doraise=True)
    except py_compile.PyCompileError:
        return False, 0, "compile_error", len(response.files)

    for file_path in written_paths:
        content = file_path.read_text(encoding="utf-8")
        has_marker = "# VERIFIED_EVAL_SNIPPET" in content
        has_line_budget = len(content.splitlines()) >= 250
        if has_marker and has_line_budget:
            return True, 0, "success", len(response.files)

    return False, 0, "validation_failed", len(response.files)


async def run_task_with_timeout(
    *,
    task: EvalTask,
    mode: Mode,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    trial: int,
    work_root: Path,
) -> TaskRunResult:
    """Execute one task and return measured result."""
    task_workdir = work_root / mode / task.task_id
    task_workdir.mkdir(parents=True, exist_ok=True)

    options = build_options(mode, task_workdir, model, max_turns)
    start = time.perf_counter()

    try:
        if timeout_seconds > 0:
            if mode == "baseline":
                completed, tool_calls, final_status = await asyncio.wait_for(
                    run_baseline_task(task, options),
                    timeout=timeout_seconds,
                )
                prefetched = 0
            else:
                completed, tool_calls, final_status, prefetched = await asyncio.wait_for(
                    run_raysurfer_task(task, task_workdir),
                    timeout=timeout_seconds,
                )
        else:
            if mode == "baseline":
                completed, tool_calls, final_status = await run_baseline_task(task, options)
                prefetched = 0
            else:
                completed, tool_calls, final_status, prefetched = await run_raysurfer_task(
                    task,
                    task_workdir,
                )

        elapsed = time.perf_counter() - start
        details = (
            f"status={final_status};"
            f"tools={tool_calls};"
            f"prefetched_files={prefetched};"
            f"workdir={task_workdir}"
        )
        return TaskRunResult(
            task_id=task.task_id,
            trial=trial,
            completed=completed,
            elapsed_seconds=elapsed,
            timestamp_utc=now_utc_iso(),
            details=details,
        )

    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        return TaskRunResult(
            task_id=task.task_id,
            trial=trial,
            completed=False,
            elapsed_seconds=elapsed,
            timestamp_utc=now_utc_iso(),
            details=f"status=timeout;tools=0;prefetched_files=0;workdir={task_workdir}",
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        message = str(exc).replace("\n", " ")
        return TaskRunResult(
            task_id=task.task_id,
            trial=trial,
            completed=False,
            elapsed_seconds=elapsed,
            timestamp_utc=now_utc_iso(),
            details=f"status=exception;error={message};workdir={task_workdir}",
        )


def write_results(path: Path, label: str, runs: list[TaskRunResult], notes: str) -> None:
    """Write run results in scorer-compatible format."""
    payload = {
        "label": label,
        "notes": notes,
        "runs": [
            {
                "task_id": item.task_id,
                "trial": item.trial,
                "completed": item.completed,
                "elapsed_seconds": round(item.elapsed_seconds, 3),
                "timestamp_utc": item.timestamp_utc,
                "details": item.details,
            }
            for item in runs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def run_eval(
    *,
    mode: Mode,
    tasks: list[EvalTask],
    out_path: Path,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    work_root: Path,
) -> None:
    """Run the task list sequentially in one mode."""
    results: list[TaskRunResult] = []

    for index, task in enumerate(tasks, start=1):
        print(f"[{mode}] {index}/{len(tasks)} {task.task_id} - {task.title}", flush=True)
        result = await run_task_with_timeout(
            task=task,
            mode=mode,
            model=model,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            trial=1,
            work_root=work_root,
        )
        results.append(result)

        marker = "OK" if result.completed else "FAIL"
        print(
            f"  -> {marker} elapsed={result.elapsed_seconds:.1f}s details={result.details}",
            flush=True,
        )

    notes = (
        f"mode={mode};model={model};max_turns={max_turns};"
        f"timeout_seconds={timeout_seconds};date={now_utc_iso()}"
    )
    write_results(out_path, mode, results, notes)
    print(f"wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run eval tasks in baseline or Raysurfer mode")
    parser.add_argument("--tasks", type=Path, default=Path("tasks/tasks.json"))
    parser.add_argument("--mode", choices=["baseline", "raysurfer"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("runs/workdirs"),
        help="Where per-task agent workdirs are created",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    load_env()

    tasks = load_tasks(args.tasks, args.limit)
    if not tasks:
        raise ValueError("No tasks to run")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Set it in env or /Users/raymondxu/raysurfer/.env"
        )
    if args.mode == "raysurfer" and not os.getenv("RAYSURFER_API_KEY"):
        raise RuntimeError(
            "RAYSURFER_API_KEY is missing. Set it in env or /Users/raymondxu/raysurfer/.env"
        )

    asyncio.run(
        run_eval(
            mode=args.mode,
            tasks=tasks,
            out_path=args.out,
            model=args.model,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            work_root=args.work_root,
        )
    )


if __name__ == "__main__":
    main()
