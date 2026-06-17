#!/usr/bin/env python3
"""Estimate inter-chunk pause duration using MeCab morphological analysis.

Analyzes the final token of each chunk text and assigns gap_after_ms based
on boundary strength: sentence-end punctuation > clause punctuation >
sentence-final particles > copula/auxiliary > other.

Input JSON (same as annotate_script.py / chunk_synth.py):
    [{"text": "怖いですね。", ...}, {"text": "本当に。", ...}]

Output: same JSON with gap_after_ms added to each entry.
The final chunk always gets gap_after_ms=0.

Usage:
    python tools/estimate_gaps.py --chunks chunks.json
    python tools/estimate_gaps.py --chunks chunks.json --out chunks_with_gaps.json
    annotate_script.py ... | estimate_gaps.py | chunk_synth.py --out-dir out/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# --- Gap rules (evaluated in order, first match wins) ---
# Each rule: (label, test_fn(pos1, pos2, surface), gap_ms)
_RULES: list[tuple[str, object, int]] = [
    # Strong sentence boundary: 。！？
    ("句点",      lambda p1, p2, s: p1 == "記号" and p2 in ("句点", "感嘆符", "疑問符"),  400),
    ("感嘆符",    lambda p1, p2, s: s in ("！", "!"),                                     400),
    ("疑問符",    lambda p1, p2, s: s in ("？", "?"),                                     400),
    # Clause boundary: 、
    ("読点",      lambda p1, p2, s: p1 == "記号" and p2 == "読点",                        200),
    # Sentence-final particle: ね よ か な わ ぞ ぜ
    ("終助詞",    lambda p1, p2, s: p1 == "助詞" and p2 == "終助詞",                      250),
    # Copula / auxiliary at utterance end (です ます た だ)
    ("助動詞",    lambda p1, p2, s: p1 == "助動詞",                                       150),
    # Noun / verb end without punctuation
    ("その他",    lambda p1, p2, s: True,                                                   80),
]


def _parse_last_token(text: str) -> tuple[str, str, str]:
    """Return (pos1, pos2, surface) of the final non-EOS token in text."""
    proc = subprocess.run(
        ["mecab"],
        input=text.strip() + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    last_pos1, last_pos2, last_surface = "", "", ""
    for line in proc.stdout.splitlines():
        if line in ("EOS", ""):
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        surface = parts[0]
        features = parts[1].split(",")
        pos1 = features[0] if len(features) > 0 else ""
        pos2 = features[1] if len(features) > 1 else ""
        last_surface, last_pos1, last_pos2 = surface, pos1, pos2
    return last_pos1, last_pos2, last_surface


def estimate_gap(text: str) -> tuple[int, str]:
    """Return (gap_ms, rule_label) for a chunk text."""
    pos1, pos2, surface = _parse_last_token(text)
    for label, test, gap_ms in _RULES:
        if test(pos1, pos2, surface):
            return gap_ms, label
    return 80, "その他"


def add_gaps(chunks: list[dict]) -> list[dict]:
    """Add gap_after_ms to each chunk. Last chunk always gets 0."""
    result = []
    for i, entry in enumerate(chunks):
        entry = dict(entry)
        if i == len(chunks) - 1:
            entry["gap_after_ms"] = 0
        else:
            gap_ms, label = estimate_gap(entry["text"])
            entry["gap_after_ms"] = gap_ms
            entry["_gap_rule"] = label
        result.append(entry)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate inter-chunk pause duration via MeCab boundary analysis"
    )
    parser.add_argument("--chunks", type=Path, help="Input chunk JSON (default: stdin)")
    parser.add_argument("--out", type=Path, help="Output JSON path (default: stdout)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-chunk gap estimates to stderr")
    args = parser.parse_args()

    if args.chunks:
        chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    elif not sys.stdin.isatty():
        chunks = json.loads(sys.stdin.read())
    else:
        parser.error("Provide --chunks or pipe JSON via stdin")
        return 1

    result = add_gaps(chunks)

    if args.verbose:
        for entry in result:
            gap = entry.get("gap_after_ms", 0)
            rule = entry.get("_gap_rule", "—")
            sys.stderr.write(f"  {entry['text']!r:30s}  → {gap:3d}ms  [{rule}]\n")

    # Strip internal _gap_rule before output
    for entry in result:
        entry.pop("_gap_rule", None)

    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(out_json, encoding="utf-8")
        sys.stderr.write(f"wrote → {args.out}\n")
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
