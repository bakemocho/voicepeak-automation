#!/usr/bin/env python3
"""Situation-based parameter case library for chunk synthesis.

Accumulates hand-tuned chunk params alongside situation context so past
examples can be retrieved as starting baselines for new scripts.

Usage:
    # Add an adopted case
    python tools/case_library.py add \\
        --chunks adopted.json \\
        --situation "主人公が恐怖で足がすくんでいる場面" \\
        --scene-mode dramatic_fear \\
        --narrator "Koharu Rikka" \\
        --mos 2.47 \\
        --source "param_search param-search-004 cand1" \\
        --notes "chunk3が平坦だったのでlam 0.5→0.9に。指示: もっと怖そうに、足がすくむ感じで"

    # Search by keyword (checks situation + scene_mode + notes)
    python tools/case_library.py search "主人公 恐怖"
    python tools/case_library.py search "もっと怖そう"

    # Filter by scene_mode
    python tools/case_library.py search --scene-mode dramatic_fear

    # List all cases
    python tools/case_library.py list

    # Export a case's chunks.json as a new baseline
    python tools/case_library.py export case-20260618-001 --out baseline.json

Default library path: ./case_library/cases.jsonl
Override with --library or CASE_LIBRARY_PATH env var.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

_DEFAULT_LIBRARY = Path(__file__).parent.parent / "case_library" / "cases.jsonl"


def _library_path(args: argparse.Namespace) -> Path:
    env = os.environ.get("CASE_LIBRARY_PATH")
    if env:
        return Path(env)
    return args.library


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return cases


def _save_append(path: Path, case: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


def _next_id(cases: list[dict]) -> str:
    today = date.today().strftime("%Y%m%d")
    prefix = f"case-{today}-"
    existing = [c["id"] for c in cases if c.get("id", "").startswith(prefix)]
    n = max((int(e.split("-")[-1]) for e in existing if e.split("-")[-1].isdigit()), default=0)
    return f"{prefix}{n+1:03d}"


def _match(case: dict, tokens: list[str], scene_mode: str | None, narrator: str | None) -> bool:
    if scene_mode and case.get("scene_mode", "").lower() != scene_mode.lower():
        return False
    if narrator and case.get("narrator", "").lower() != narrator.lower():
        return False
    if tokens:
        haystack = " ".join([
            case.get("situation", ""),
            case.get("scene_mode", ""),
            case.get("notes", ""),
        ]).lower()
        return all(t.lower() in haystack for t in tokens)
    return True


def _fmt_case(case: dict, verbose: bool = False) -> str:
    mos_s = f"MOS={case['mos']:.3f}" if "mos" in case else "MOS=—"
    n_chunks = len(case.get("chunks", []))
    src = case.get("source", "")
    notes = case.get("notes", "")
    line = (
        f"  {case['id']}  [{case.get('scene_mode','—'):20s}]  {mos_s}  "
        f"{n_chunks}chunks  {case.get('date','—')}\n"
        f"    {case.get('situation','')}"
    )
    if notes:
        line += f"\n    notes: {notes}"
    if verbose and src:
        line += f"\n    source: {src}"
    return line


def cmd_add(args: argparse.Namespace) -> None:
    lib = _library_path(args)
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))

    cases = _load(lib)
    case_id = _next_id(cases)
    case: dict = {
        "id": case_id,
        "date": date.today().isoformat(),
        "situation": args.situation,
        "scene_mode": args.scene_mode or "",
        "narrator": args.narrator,
        "chunks": chunks,
    }
    if args.mos is not None:
        case["mos"] = round(args.mos, 4)
    if args.source:
        case["source"] = args.source
    if args.notes:
        case["notes"] = args.notes

    _save_append(lib, case)
    print(f"added: {case_id}  →  {lib}")
    print(f"  situation : {args.situation}")
    print(f"  scene_mode: {args.scene_mode or '(none)'}")
    print(f"  chunks    : {len(chunks)}")
    if args.mos is not None:
        print(f"  MOS       : {args.mos:.3f}")
    if args.notes:
        print(f"  notes     : {args.notes}")


def cmd_search(args: argparse.Namespace) -> None:
    lib = _library_path(args)
    cases = _load(lib)
    if not cases:
        print(f"Library is empty: {lib}")
        return

    tokens = args.query.split() if args.query else []
    results = [c for c in cases if _match(c, tokens, args.scene_mode, args.narrator)]

    if not results:
        print("No matches.")
        return

    # Sort by MOS descending (cases without MOS go last)
    results.sort(key=lambda c: c.get("mos", -1.0), reverse=True)
    top = results[: args.top_k]

    print(f"\n{len(top)} result(s) (of {len(results)} matches, {len(cases)} total):\n")
    for c in top:
        print(_fmt_case(c, verbose=args.verbose))
        print()


def cmd_list(args: argparse.Namespace) -> None:
    lib = _library_path(args)
    cases = _load(lib)
    if not cases:
        print(f"Library is empty: {lib}")
        return

    print(f"\n{len(cases)} case(s) in {lib}:\n")
    for c in cases:
        print(_fmt_case(c))
        print()


def cmd_export(args: argparse.Namespace) -> None:
    lib = _library_path(args)
    cases = _load(lib)
    matches = [c for c in cases if c["id"] == args.case_id]
    if not matches:
        print(f"Case not found: {args.case_id}")
        sys.exit(1)
    case = matches[0]
    out = args.out or Path(f"{args.case_id}.json")
    Path(out).write_text(
        json.dumps(case["chunks"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"exported chunks → {out}")
    print(f"  situation : {case.get('situation','')}")
    print(f"  scene_mode: {case.get('scene_mode','')}")
    print(f"  chunks    : {len(case['chunks'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Situation-based parameter case library")
    parser.add_argument(
        "--library", type=Path, default=_DEFAULT_LIBRARY,
        help=f"Path to cases.jsonl (default: {_DEFAULT_LIBRARY})",
    )
    sub = parser.add_subparsers(dest="cmd")

    # add
    p_add = sub.add_parser("add", help="Add a case from adopted chunks.json")
    p_add.add_argument("--chunks", type=Path, required=True, help="Adopted chunks JSON")
    p_add.add_argument("--situation", required=True, help="Free-text situation description (Japanese OK)")
    p_add.add_argument("--scene-mode", help="Structured tag e.g. dramatic_fear, calm_narration")
    p_add.add_argument("--narrator", default="Koharu Rikka")
    p_add.add_argument("--mos", type=float, help="MOS score if known")
    p_add.add_argument("--source", help="Provenance note e.g. 'param_search param-search-004 cand1'")
    p_add.add_argument("--notes", help="Editorial notes: correction history, instructions given, what was wrong")

    # search
    p_search = sub.add_parser("search", help="Search cases by keyword / tag")
    p_search.add_argument("query", nargs="?", default="", help="Space-separated keywords")
    p_search.add_argument("--scene-mode", help="Filter by scene_mode tag")
    p_search.add_argument("--narrator", help="Filter by narrator")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--verbose", action="store_true")

    # list
    sub.add_parser("list", help="List all cases")

    # export
    p_exp = sub.add_parser("export", help="Export a case's chunks.json as baseline")
    p_exp.add_argument("case_id", help="Case ID e.g. case-20260618-001")
    p_exp.add_argument("--out", type=Path, help="Output path (default: <case_id>.json)")

    args = parser.parse_args()

    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "export":
        cmd_export(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
