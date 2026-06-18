#!/usr/bin/env python3
"""Analyze VOICEPEAK call logs from logs/vp_calls.jsonl.

Usage:
    python tools/vp_log_show.py              # summary
    python tools/vp_log_show.py --errors     # failures only
    python tools/vp_log_show.py --gaps       # call_n gaps between failures
    python tools/vp_log_show.py --last N     # last N records
    python tools/vp_log_show.py --clear      # truncate both log files
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_CALLS_LOG = _LOG_DIR / "vp_calls.jsonl"
_ERRORS_LOG = _LOG_DIR / "vp_errors.jsonl"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _fmt_row(r: dict) -> str:
    ok_s = "ok  " if r["ok"] else "FAIL"
    return (
        f"  [{r['call_n']:>4}] {ok_s}  rc={r['rc']}  {r['duration_ms']:>5}ms"
        f"  spd={r['speed']:>3} pit={r['pitch']:>4} em={r['emotion']:<20}"
        f"  {r['text'][:30]!r}"
    )


def cmd_summary(args: argparse.Namespace) -> None:
    rows = _load(_CALLS_LOG)
    if not rows:
        print(f"No calls logged yet ({_CALLS_LOG})")
        return

    total = len(rows)
    fails = [r for r in rows if not r["ok"]]
    ok_dur = [r["duration_ms"] for r in rows if r["ok"]]
    fail_call_ns = [r["call_n"] for r in fails]

    print(f"Log: {_CALLS_LOG}  ({total} calls)")
    print(f"  ok   : {total - len(fails)}")
    print(f"  FAIL : {len(fails)}  ({100*len(fails)/total:.1f}%)")
    if ok_dur:
        print(f"  avg duration (ok): {sum(ok_dur)//len(ok_dur)}ms")
    if fail_call_ns:
        print(f"  failure call_n   : {fail_call_ns}")
        gaps = [b - a for a, b in zip(fail_call_ns, fail_call_ns[1:])]
        if gaps:
            print(f"  gaps between failures: {gaps}  (mean={sum(gaps)/len(gaps):.1f})")
    if args.last:
        print(f"\nLast {args.last} calls:")
        for r in rows[-args.last:]:
            print(_fmt_row(r))


def cmd_errors(args: argparse.Namespace) -> None:
    rows = _load(_ERRORS_LOG)
    if not rows:
        print(f"No failures logged ({_ERRORS_LOG})")
        return
    print(f"{len(rows)} failures in {_ERRORS_LOG}\n")
    for r in rows:
        print(_fmt_row(r))
        # Crash report
        if r.get("crash_report"):
            print(f"    crash_report: {r['crash_report']}")
        if r.get("crash_summary"):
            cs = r["crash_summary"]
            print(f"    exception: {cs.get('exception_type','')} {cs.get('exception_signal','')}  codes: {cs.get('exception_codes','')}")
            for i, frame in enumerate(cs.get("crashed_frames", [])[:6]):
                print(f"      #{i} {frame}")
        elif r.get("stderr"):
            tail = r["stderr"].strip().splitlines()
            interesting = [l for l in tail if "iconv_open" not in l and l.strip()][-5:]
            for l in interesting:
                print(f"    stderr: {l}")
        print()


def cmd_gaps(args: argparse.Namespace) -> None:
    rows = _load(_CALLS_LOG)
    fail_ns = [r["call_n"] for r in rows if not r["ok"]]
    if not fail_ns:
        print("No failures found.")
        return
    print(f"Failure call_n values: {fail_ns}")
    if len(fail_ns) > 1:
        gaps = [b - a for a, b in zip(fail_ns, fail_ns[1:])]
        print(f"Gaps between consecutive failures: {gaps}")
        print(f"  min={min(gaps)}  max={max(gaps)}  mean={sum(gaps)/len(gaps):.1f}")
    # Also show call_n mod small numbers to look for periodicity
    for period in [8, 9, 10, 11, 12]:
        mods = [n % period for n in fail_ns]
        if len(set(mods)) <= 2:
            print(f"  Possible period {period}: call_n mod {period} = {mods}")


def cmd_clear(args: argparse.Namespace) -> None:
    for p in [_CALLS_LOG, _ERRORS_LOG]:
        if p.exists():
            p.write_text("", encoding="utf-8")
            print(f"cleared: {p}")
        else:
            print(f"not found: {p}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze VOICEPEAK call logs")
    parser.add_argument("--errors", action="store_true", help="Show failures only")
    parser.add_argument("--gaps", action="store_true", help="Analyze call_n gaps between failures")
    parser.add_argument("--last", type=int, metavar="N", help="Show last N records in summary")
    parser.add_argument("--clear", action="store_true", help="Truncate log files")
    args = parser.parse_args()

    if args.clear:
        cmd_clear(args)
    elif args.errors:
        cmd_errors(args)
    elif args.gaps:
        cmd_gaps(args)
    else:
        cmd_summary(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
