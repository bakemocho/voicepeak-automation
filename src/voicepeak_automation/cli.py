from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_task(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("task file must be a JSON object")
    return data


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.task)
    if not path.exists():
        raise FileNotFoundError(f"task file not found: {path}")

    data = load_task(path)
    required = ["project", "items"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")

    items = data.get("items")
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("items must be a non-empty list")

    print(f"[ok] valid task: {path}")
    print(f"project={data.get('project')} items={len(items)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.task)
    data = load_task(path)
    print(f"[dry-run={args.dry_run}] run task: {path}")
    print(f"project={data.get('project')} items={len(data.get('items', []))}")
    print("runner implementation is pending")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicepeak-automation")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate task JSON")
    validate.add_argument("task", help="path to task json")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run task (currently stub)")
    run.add_argument("--task", required=True, help="path to task json")
    run.add_argument("--dry-run", action="store_true", default=True)
    run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
