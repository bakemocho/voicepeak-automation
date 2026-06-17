from __future__ import annotations

import argparse
from pathlib import Path

from voicepeak_automation.dic import (
    DEFAULT_DIC_PATH,
    DicEntry,
    DicError,
    add_entry,
    load_dic,
    remove_entry,
    save_dic,
    validate_entry,
)
from voicepeak_automation.runner import RunResult, run_task
from voicepeak_automation.task import TaskValidationError, parse_task
from voicepeak_automation.vpp import VppParams, generate_vpp, write_vpp


def _format_run_summary(result: RunResult) -> str:
    lines = [
        f"project={result.task_project}",
        f"dry_run={result.dry_run}",
        f"chunks={len(result.chunk_results)}",
        f"warnings={len(result.warnings)}",
        f"errors={len(result.errors)}",
    ]
    for chunk in result.chunk_results:
        lines.append(
            f"- {chunk.item_id}#{chunk.chunk_index}: {chunk.output_wav}"
        )
    if result.warnings:
        lines.append("[warnings]")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    if result.errors:
        lines.append("[errors]")
        for error in result.errors:
            lines.append(f"- {error}")
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
    if result.errors:
        return 2
    return 0


def cmd_dic_list(args: argparse.Namespace) -> int:
    path = Path(args.dic_path) if args.dic_path else DEFAULT_DIC_PATH
    entries = load_dic(path)
    if not entries:
        print(f"[ok] dic empty: {path}")
        return 0
    print(f"[ok] {len(entries)} entries: {path}")
    for e in entries:
        errors = validate_entry(e)
        flag = " [WARN:" + ",".join(errors) + "]" if errors else ""
        print(f"  sur={e.sur!r} pron={e.pron!r} pos={e.pos} accent={e.accentType}{flag}")
    return 0


def cmd_dic_add(args: argparse.Namespace) -> int:
    path = Path(args.dic_path) if args.dic_path else DEFAULT_DIC_PATH
    entry = DicEntry(
        sur=args.sur,
        pron=args.pron,
        pos=args.pos,
        priority=args.priority,
        accentType=args.accent_type,
        lang="ja",
    )
    errors = validate_entry(entry)
    if errors:
        for e in errors:
            print(f"[error] {e}")
        return 1
    entries = load_dic(path)
    entries, replaced = add_entry(entries, entry)
    save_dic(entries, path)
    action = "replaced" if replaced else "added"
    print(f"[ok] {action}: sur={entry.sur!r} pron={entry.pron!r} accent={entry.accentType}")
    print("note: restart VOICEPEAK to apply changes")
    return 0


def cmd_dic_remove(args: argparse.Namespace) -> int:
    path = Path(args.dic_path) if args.dic_path else DEFAULT_DIC_PATH
    entries = load_dic(path)
    entries, found = remove_entry(entries, args.sur)
    if not found:
        print(f"[warn] not found: sur={args.sur!r}")
        return 1
    save_dic(entries, path)
    print(f"[ok] removed: sur={args.sur!r}")
    print("note: restart VOICEPEAK to apply changes")
    return 0


def cmd_dic_validate(args: argparse.Namespace) -> int:
    path = Path(args.dic_path) if args.dic_path else DEFAULT_DIC_PATH
    try:
        entries = load_dic(path)
    except DicError as exc:
        print(f"[error] {exc}")
        return 1
    issues: list[str] = []
    for e in entries:
        for err in validate_entry(e):
            issues.append(f"sur={e.sur!r}: {err}")
    if issues:
        for issue in issues:
            print(f"[warn] {issue}")
        return 1
    print(f"[ok] {len(entries)} entries valid: {path}")
    return 0


def cmd_vpp_generate(args: argparse.Namespace) -> int:
    text = args.text or (Path(args.text_file).read_text(encoding="utf-8") if args.text_file else None)
    if not text:
        print("[error] provide --text or --text-file")
        return 1

    params = VppParams(
        narrator=args.narrator,
        speed=args.speed,
        pitch=args.pitch,
        pause_scale=args.pause_scale,
        comma_pause_d=args.comma_pause,
        period_pause_d=args.period_pause,
    )

    dic_path = Path(args.dic_path) if args.dic_path else None
    data = generate_vpp(text, params=params, **({"dic_path": dic_path} if dic_path else {}))

    out = Path(args.output)
    write_vpp(data, out)
    n_blocks = len(data["project"]["blocks"])
    n_sents = sum(len(b["sentence-list"]) for b in data["project"]["blocks"])
    print(f"[ok] wrote {out} ({n_blocks} block, {n_sents} sentences)")
    print("note: open in VOICEPEAK GUI to synthesize and export audio")
    return 0


def cmd_vpp_synth(args: argparse.Namespace) -> int:
    from voicepeak_automation.synth import synthesize_vpp
    vpp_path = Path(args.vpp)
    output_dir = Path(args.output_dir)
    try:
        synthesize_vpp(vpp_path, output_dir, timeout=args.timeout)
        print(f"[ok] export triggered → {output_dir}")
    except Exception as exc:
        print(f"[error] {exc}")
        return 1
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

    _dic_path_kwargs = dict(default=None, metavar="PATH", help="path to dic.json (default: VOICEPEAK settings)")

    dic_list = sub.add_parser("dic-list", help="list user dictionary entries")
    dic_list.add_argument("--dic-path", **_dic_path_kwargs)
    dic_list.set_defaults(func=cmd_dic_list)

    dic_add = sub.add_parser("dic-add", help="add or replace a user dictionary entry")
    dic_add.add_argument("--sur", required=True, help="surface form (the word as written)")
    dic_add.add_argument("--pron", required=True, help="katakana reading")
    dic_add.add_argument("--accent-type", required=True, type=int, metavar="N",
                         help="pitch accent nucleus position (0=flat, 1=drop after 1st mora, …)")
    dic_add.add_argument("--pos", default="Japanese_Koyuumeishi_ippan",
                         help="part of speech (default: Japanese_Koyuumeishi_ippan)")
    dic_add.add_argument("--priority", type=int, default=5, metavar="N", help="priority 1-9 (default: 5)")
    dic_add.add_argument("--dic-path", **_dic_path_kwargs)
    dic_add.set_defaults(func=cmd_dic_add)

    dic_remove = sub.add_parser("dic-remove", help="remove a user dictionary entry by sur")
    dic_remove.add_argument("--sur", required=True, help="surface form to remove")
    dic_remove.add_argument("--dic-path", **_dic_path_kwargs)
    dic_remove.set_defaults(func=cmd_dic_remove)

    dic_validate = sub.add_parser("dic-validate", help="validate all entries in dic.json")
    dic_validate.add_argument("--dic-path", **_dic_path_kwargs)
    dic_validate.set_defaults(func=cmd_dic_validate)

    vpp_gen = sub.add_parser("vpp-generate", help="generate a .vpp project file from Japanese text")
    vpp_gen.add_argument("--text", default=None, help="text to synthesize (inline)")
    vpp_gen.add_argument("--text-file", default=None, metavar="FILE", help="read text from file")
    vpp_gen.add_argument("--output", required=True, metavar="FILE.vpp", help="output .vpp path")
    vpp_gen.add_argument("--narrator", default="Koharu Rikka", help="narrator name (default: Koharu Rikka)")
    vpp_gen.add_argument("--speed", type=float, default=1.0, help="speed multiplier 0.5-2.0 (default: 1.0)")
    vpp_gen.add_argument("--pitch", type=float, default=0.0, help="pitch shift (default: 0.0)")
    vpp_gen.add_argument("--pause-scale", type=float, default=1.0, help="global pause scale (default: 1.0)")
    vpp_gen.add_argument("--comma-pause", type=float, default=1.0, metavar="D",
                         help="pause duration multiplier at 、 (default: 1.0)")
    vpp_gen.add_argument("--period-pause", type=float, default=1.5, metavar="D",
                         help="pause duration multiplier at 。 (default: 1.5)")
    vpp_gen.add_argument("--dic-path", **_dic_path_kwargs)
    vpp_gen.set_defaults(func=cmd_vpp_generate)

    vpp_synth = sub.add_parser("vpp-synth", help="open .vpp in VOICEPEAK GUI and trigger export (macOS only)")
    vpp_synth.add_argument("--vpp", required=True, metavar="FILE.vpp", help="input .vpp path")
    vpp_synth.add_argument("--output-dir", required=True, metavar="DIR", help="export output directory")
    vpp_synth.add_argument("--timeout", type=int, default=60, help="seconds to wait for export (default: 60)")
    vpp_synth.set_defaults(func=cmd_vpp_synth)

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
