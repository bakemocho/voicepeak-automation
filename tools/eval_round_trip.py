#!/usr/bin/env python3
"""Round-trip evaluation: text → VOICEPEAK synthesis → mjo transcription → match check.

Usage:
    python tools/eval_round_trip.py
    python tools/eval_round_trip.py --text "テスト文" --narrator "Koharu Rikka"

Exit codes: 0=all pass, 1=mismatch/error
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

VOICEPEAK_BIN = "/Applications/voicepeak.app/Contents/MacOS/voicepeak"
MJO_ALIAS = "mjo"

CASES = [
    {"text": "田中さんは東京に行きました。", "narrator": "Koharu Rikka"},
    {"text": "テスト、です。確認用の音声。", "narrator": "Koharu Rikka"},
]


def synthesize(text: str, narrator: str, out_wav: Path) -> bool:
    proc = subprocess.run(
        [VOICEPEAK_BIN, "-s", text, "--narrator", narrator, "-o", str(out_wav)],
        capture_output=True,
        text=True,
    )
    return out_wav.exists()


def transcribe(wav_path: Path, out_dir: Path) -> str | None:
    # Run mjo from /tmp so its log files never land in the calling repo's CWD.
    proc = subprocess.run(
        ["zsh", "-ic", f"mjo {wav_path} --output-dir {out_dir} --language ja"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd="/tmp",
    )
    stdout = proc.stdout
    # mjo stdout: log lines, then one JSON object, then one more log line.
    # Use raw_decode to parse the first complete JSON object.
    decoder = json.JSONDecoder()
    idx = stdout.find("{")
    if idx == -1:
        return None
    try:
        obj, _ = decoder.raw_decode(stdout, idx)
    except json.JSONDecodeError:
        return None
    # Shape: { "<wav_path>": { "entries": [{ "transcribe_result": { "full_text": ... } }] } }
    for source_val in obj.values():
        entries = source_val.get("entries", [])
        if entries:
            return entries[0].get("transcribe_result", {}).get("full_text")
    return None


_PUNCT = str.maketrans("", "", "、。！？…・ 　,.")


def normalize(text: str, strip_punct: bool = False) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKC", text).strip()
    if strip_punct:
        t = t.translate(_PUNCT)
    return t


def run_cases(cases: list[dict], narrator_override: str | None = None) -> bool:
    all_pass = True
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, case in enumerate(cases):
            text = case["text"]
            narrator = narrator_override or case["narrator"]
            wav = tmpdir / f"case_{i}.wav"
            tr_dir = tmpdir / f"tr_{i}"
            tr_dir.mkdir()

            # Step 1: synthesize
            ok = synthesize(text, narrator, wav)
            if not ok:
                print(f"[FAIL] case {i}: synthesis failed — {text!r}")
                all_pass = False
                continue

            # Step 2: transcribe
            transcript = transcribe(wav, tr_dir)
            if transcript is None:
                print(f"[FAIL] case {i}: transcription returned nothing — {text!r}")
                all_pass = False
                continue

            # Step 3: strict compare (NFKC normalized)
            expected = normalize(text)
            got = normalize(transcript)
            if expected == got:
                print(f"[PASS] case {i}: {text!r}")
            else:
                # Step 3b: fuzzy compare (strip punctuation+spaces)
                exp_f = normalize(text, strip_punct=True)
                got_f = normalize(transcript, strip_punct=True)
                if exp_f == got_f:
                    print(f"[PASS-FUZZY] case {i}: words match (punct differs) — {text!r}")
                    print(f"           got: {got!r}")
                else:
                    print(f"[FAIL] case {i}: expected={expected!r} got={got!r}")
                    all_pass = False

    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Round-trip TTS evaluation")
    parser.add_argument("--text", default=None, help="Single text to test")
    parser.add_argument("--narrator", default="Koharu Rikka", help="Narrator name")
    args = parser.parse_args()

    if args.text:
        cases = [{"text": args.text, "narrator": args.narrator}]
    else:
        cases = CASES

    return 0 if run_cases(cases, narrator_override=args.narrator if args.text else None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
