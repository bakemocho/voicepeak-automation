from __future__ import annotations

import argparse
from pathlib import Path

from voicepeak_automation.runner import RunResult, run_task
from voicepeak_automation.task import TaskValidationError, parse_task


def _format_run_summary(result: RunResult) -> str:
    lines = [
        f"project={result.task_project}",
        f"dry_run={result.dry_run}",
        f"chunks={len(result.chunk_results)}",
    ]
    for chunk in result.chunk_results:
        lines.append(
            f"- {chunk.item_id}#{chunk.chunk_index}: {chunk.output_wav}"
        )
    return "\n".join(lines)


def cmd_validate(args: argparse.Namespace) -> int:
    task = parse_task(Path(args.task))
    print(f"[ok] valid task: {task.source_path}")
    print(f"project={task.project} items={len(task.items)}")
    print(f"output_dir={task.settings.output_dir}")
    print(f"formula_mode={task.settings.formula_mode}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    task = parse_task(Path(args.task))
    result = run_task(task=task, dry_run=bool(args.dry_run))
    print(_format_run_summary(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicepeak-automation")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate task JSON")
    validate.add_argument("task", help="path to task json")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run synthesis task")
    run.add_argument("--task", required=True, help="path to task json")
    run.add_argument("--dry-run", action="store_true", help="build chunk plan without invoking voicepeak")
    run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except (FileNotFoundError, TaskValidationError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
