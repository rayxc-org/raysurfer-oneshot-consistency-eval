#!/usr/bin/env python3
"""Run rotating one-shot eval with Claude SDK baseline vs Raysurfer drop-in replacement."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import py_compile
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock, ToolUseBlock
from raysurfer import RaysurferClient


Mode = Literal["baseline", "raysurfer"]


ROTATING_VARIANTS = [
    "Variant focus: deterministic helper decomposition with strict naming.",
    "Variant focus: robust input guards before core workflow logic.",
    "Variant focus: stable output formatting and explicit edge-case handling.",
    "Variant focus: minimal branching and repeatable control flow.",
]


@dataclass(frozen=True)
class EvalTask:
    """A single one-shot benchmark task."""

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


def build_rotating_prompt(task: EvalTask, round_index: int, position_index: int) -> tuple[str, str]:
    """Create a rotated benchmark prompt with cache-first review requirements."""
    variant_index = (round_index + position_index) % len(ROTATING_VARIANTS)
    variant_label = f"round_{round_index + 1:02d}_variant_{variant_index + 1:02d}"
    file_name = f"{task.task_id.lower()}_solution.py"

    prompt = (
        "Benchmark key: RS_ONESHOT_EVAL_2026_02_20_ROTATING_V1\n"
        f"Benchmark task id: {task.task_id}\n"
        f"Rotation label: {variant_label}\n"
        f"{ROTATING_VARIANTS[variant_index]}\n\n"
        f"Task: {task.one_shot_prompt}\n\n"
        "Constraints:\n"
        "- Write Python code only.\n"
        f"- Save everything into one file at {file_name}.\n"
        "- The file must include the literal marker: # VERIFIED_EVAL_SNIPPET\n"
        "- The file must be at least 250 lines long.\n"
        "- Include type hints and deterministic structure.\n"
        "- If .raysurfer_code exists and has files, inspect cache files + reputation notes before coding.\n"
        "- If cache files are present, write cache_review.json with keys: reviewed_files, selected_file, rationale.\n"
        "- Run python -m py_compile on the final file before stopping.\n"
        "- Reply with DONE after validations pass.\n"
    )
    return prompt, variant_label


def build_baseline_options(task_workdir: Path, model: str, max_turns: int) -> ClaudeAgentOptions:
    """Build options for plain Claude Agent SDK runs."""
    return ClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        sandbox={"enabled": True, "autoAllowBashIfSandboxed": True},
        permission_mode="bypassPermissions",
        model=model,
        max_turns=max_turns,
        cwd=str(task_workdir),
        system_prompt=(
            "You are running a rotating coding benchmark. Compile-check output and stop once valid."
        ),
    )


def build_raysurfer_options(task_workdir: Path, model: str, max_turns: int) -> ClaudeAgentOptions:
    """Build options for Raysurfer drop-in runs (defaults fill tools+sandbox)."""
    return ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model=model,
        max_turns=max_turns,
        cwd=str(task_workdir),
        system_prompt=(
            "You are running a rotating coding benchmark. Review cached code/reputation first when available."
        ),
    )


def extract_python_source(text: str) -> str | None:
    """Extract Python source from assistant markdown text."""
    python_block = re.search(r"```python\\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if python_block:
        source = python_block.group(1).strip()
        return source if source else None

    generic_block = re.search(r"```\\s*(.*?)```", text, flags=re.DOTALL)
    if generic_block:
        source = generic_block.group(1).strip()
        return source if source else None

    stripped = text.strip()
    if stripped.startswith("def ") or stripped.startswith("from ") or stripped.startswith("import "):
        return stripped
    return None


def validate_file(path: Path) -> tuple[bool, str]:
    """Validate one-shot file requirements."""
    if not path.exists():
        return False, "missing_file"

    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError:
        return False, "compile_error"

    content = path.read_text(encoding="utf-8")
    if "# VERIFIED_EVAL_SNIPPET" not in content:
        return False, "missing_marker"
    if len(content.splitlines()) < 250:
        return False, "line_budget"

    return True, "passed"


async def run_task_once(
    *,
    task: EvalTask,
    mode: Mode,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    round_index: int,
    position_index: int,
    work_root: Path,
) -> TaskRunResult:
    """Execute one one-shot task with timeout and validation."""
    task_workdir = work_root / mode / f"round_{round_index + 1:02d}" / task.task_id
    if task_workdir.exists():
        shutil.rmtree(task_workdir)
    task_workdir.mkdir(parents=True, exist_ok=True)

    prompt, variant_label = build_rotating_prompt(task, round_index, position_index)
    options = (
        build_baseline_options(task_workdir, model, max_turns)
        if mode == "baseline"
        else build_raysurfer_options(task_workdir, model, max_turns)
    )

    start = time.perf_counter()
    tool_calls = 0
    tool_names: set[str] = set()
    final_status = "no_result"
    saw_success = False
    assistant_text_chunks: list[str] = []

    async def _execute() -> bool:
        nonlocal tool_calls, final_status, saw_success
        if mode == "baseline":
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                tool_calls += 1
                                tool_names.add(block.name)
                            elif isinstance(block, TextBlock):
                                assistant_text_chunks.append(block.text)
                    elif isinstance(msg, ResultMessage):
                        final_status = msg.subtype
                        saw_success = msg.subtype == "success"
                return bool(getattr(client, "_cached_code_blocks", []))

        async with RaysurferClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls += 1
                            tool_names.add(block.name)
                        elif isinstance(block, TextBlock):
                            assistant_text_chunks.append(block.text)
                elif isinstance(msg, ResultMessage):
                    final_status = msg.subtype
                    saw_success = msg.subtype == "success"
            return len(getattr(client, "_cached_code_blocks", [])) > 0

    try:
        cache_hit = await asyncio.wait_for(_execute(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        return TaskRunResult(
            task_id=task.task_id,
            trial=round_index + 1,
            completed=False,
            elapsed_seconds=round(elapsed, 3),
            timestamp_utc=now_utc_iso(),
            details=(
                f"status=timeout;mode={mode};tools={tool_calls};"
                f"round={round_index + 1};variant={variant_label}"
            ),
        )

    file_name = f"{task.task_id.lower()}_solution.py"
    solution_path = task_workdir / file_name
    if not solution_path.exists():
        candidates = sorted(
            [path for path in task_workdir.glob("*.py") if path.name != "__init__.py"],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            solution_path = candidates[0]
        else:
            extracted = extract_python_source("\n\n".join(assistant_text_chunks))
            if extracted:
                solution_path = task_workdir / file_name
                solution_path.write_text(extracted, encoding="utf-8")

    valid, validation_reason = validate_file(solution_path)
    cache_review_exists = (task_workdir / "cache_review.json").exists()
    completed = saw_success and valid
    elapsed = time.perf_counter() - start

    return TaskRunResult(
        task_id=task.task_id,
        trial=round_index + 1,
        completed=completed,
        elapsed_seconds=round(elapsed, 3),
        timestamp_utc=now_utc_iso(),
        details=(
            f"status={final_status};mode={mode};validation={validation_reason};tools={tool_calls};"
            f"tool_names={','.join(sorted(tool_names))};cache_hit={int(cache_hit)};"
            f"cache_review={int(cache_review_exists)};round={round_index + 1};variant={variant_label}"
        ),
    )


def rotate_tasks(tasks: list[EvalTask], round_index: int) -> list[EvalTask]:
    """Rotate ordering each round to stress persistence across varied sequence."""
    if not tasks:
        return []
    offset = round_index % len(tasks)
    return tasks[offset:] + tasks[:offset]


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
    rounds: int,
    out_path: Path,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    work_root: Path,
) -> None:
    """Run rotating evaluation tasks sequentially."""
    results: list[TaskRunResult] = []

    for round_index in range(rounds):
        ordered = rotate_tasks(tasks, round_index)
        for position_index, task in enumerate(ordered, start=1):
            print(
                f"[{mode}] round={round_index + 1}/{rounds} "
                f"{position_index}/{len(ordered)} {task.task_id} - {task.title}",
                flush=True,
            )
            result = await run_task_once(
                task=task,
                mode=mode,
                model=model,
                max_turns=max_turns,
                timeout_seconds=timeout_seconds,
                round_index=round_index,
                position_index=position_index - 1,
                work_root=work_root,
            )
            results.append(result)
            marker = "OK" if result.completed else "FAIL"
            print(
                f"  -> {marker} elapsed={result.elapsed_seconds:.1f}s details={result.details}",
                flush=True,
            )

    notes = (
        f"mode={mode};model={model};max_turns={max_turns};timeout_seconds={timeout_seconds};"
        f"rounds={rounds};rotation=deterministic;date={now_utc_iso()}"
    )
    write_results(out_path, mode, results, notes)
    print(f"wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run rotating one-shot eval in baseline or Raysurfer mode",
    )
    parser.add_argument("--tasks", type=Path, default=Path("tasks/tasks.json"))
    parser.add_argument("--mode", choices=["baseline", "raysurfer"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("runs/workdirs_rotating"),
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
            rounds=args.rounds,
            out_path=args.out,
            model=args.model,
            max_turns=args.max_turns,
            timeout_seconds=args.timeout_seconds,
            work_root=args.work_root,
        )
    )


if __name__ == "__main__":
    main()
