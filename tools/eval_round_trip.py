#!/usr/bin/env python3
"""Round-trip evaluation: text → VOICEPEAK synthesis → mjo transcription + scoring.

Usage:
    python tools/eval_round_trip.py
    python tools/eval_round_trip.py --text "テスト文" --narrator "Koharu Rikka"
    python tools/eval_round_trip.py --score          # also run MOS/emotion/VAD scoring
    python tools/eval_round_trip.py --json           # output JSON

Requires for --score: source .venv-eval/bin/activate first.

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

CASES = [
    {"text": "田中さんは東京に行きました。", "narrator": "Koharu Rikka"},
    {"text": "テスト、です。確認用の音声。", "narrator": "Koharu Rikka"},
]


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #

def synthesize(text: str, narrator: str, out_wav: Path) -> bool:
    proc = subprocess.run(
        [VOICEPEAK_BIN, "-s", text, "--narrator", narrator, "-o", str(out_wav)],
        capture_output=True,
        text=True,
    )
    return out_wav.exists()


# --------------------------------------------------------------------------- #
# Transcription (mjo)
# --------------------------------------------------------------------------- #

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
    decoder = json.JSONDecoder()
    idx = stdout.find("{")
    if idx == -1:
        return None
    try:
        obj, _ = decoder.raw_decode(stdout, idx)
    except json.JSONDecodeError:
        return None
    for source_val in obj.values():
        entries = source_val.get("entries", [])
        if entries:
            return entries[0].get("transcribe_result", {}).get("full_text")
    return None


# --------------------------------------------------------------------------- #
# Scoring (optional — requires .venv-eval)
# --------------------------------------------------------------------------- #

def _try_score(wav_path: Path) -> dict | None:
    """Run score_wav via subprocess using .venv-eval python. Returns dict or None."""
    venv_python = Path(__file__).parent.parent / ".venv-eval" / "bin" / "python"
    score_script = Path(__file__).parent / "score_wav.py"
    if not venv_python.exists() or not score_script.exists():
        return None
    proc = subprocess.run(
        [str(venv_python), str(score_script), str(wav_path), "--json"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="/tmp",
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Text comparison
# --------------------------------------------------------------------------- #

_PUNCT = str.maketrans("", "", "、。！？…・ 　,.")


def normalize(text: str, strip_punct: bool = False) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKC", text).strip()
    if strip_punct:
        t = t.translate(_PUNCT)
    return t


def _match(expected: str, got: str) -> str:
    """Return 'strict', 'fuzzy', or 'fail'."""
    if normalize(expected) == normalize(got):
        return "strict"
    if normalize(expected, strip_punct=True) == normalize(got, strip_punct=True):
        return "fuzzy"
    return "fail"


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #

def run_cases(
    cases: list[dict],
    narrator_override: str | None = None,
    with_score: bool = False,
    as_json: bool = False,
) -> bool:
    all_pass = True
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i, case in enumerate(cases):
            text = case["text"]
            narrator = narrator_override or case["narrator"]
            wav = tmpdir / f"case_{i}.wav"
            tr_dir = tmpdir / f"tr_{i}"
            tr_dir.mkdir()

            entry: dict = {"case": i, "text": text}

            # Step 1: synthesize
            if not synthesize(text, narrator, wav):
                entry["result"] = "FAIL"
                entry["reason"] = "synthesis failed"
                results.append(entry)
                all_pass = False
                continue

            # Step 2: transcribe
            transcript = transcribe(wav, tr_dir)
            if transcript is None:
                entry["result"] = "FAIL"
                entry["reason"] = "transcription returned nothing"
                results.append(entry)
                all_pass = False
                continue

            entry["transcript"] = transcript

            # Step 3: text match
            match = _match(text, transcript)
            if match == "strict":
                entry["result"] = "PASS"
            elif match == "fuzzy":
                entry["result"] = "PASS-FUZZY"
                entry["got"] = normalize(transcript)
            else:
                entry["result"] = "FAIL"
                entry["got"] = normalize(transcript)
                all_pass = False

            # Step 4 (optional): quality scoring
            if with_score:
                scores = _try_score(wav)
                if scores:
                    entry["mos"] = scores.get("mos")
                    entry["emotion"] = scores.get("emotion")
                    entry["arousal"] = scores.get("arousal")
                    entry["valence"] = scores.get("valence")
                else:
                    entry["score_error"] = "scorer unavailable"

            results.append(entry)

    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for e in results:
            r = e["result"]
            t = e["text"]
            prefix = f"[{r}] case {e['case']}: {t!r}"
            if r == "PASS-FUZZY":
                print(f"{prefix}  (got: {e.get('got')!r})")
            elif r == "FAIL":
                print(f"{prefix}  reason={e.get('reason','mismatch')}  got={e.get('got')!r}")
            else:
                print(prefix)
            if "mos" in e:
                print(f"       MOS={e['mos']:.3f}  emotion={e['emotion']}  arousal={e['arousal']:.3f}  valence={e['valence']:.3f}")

    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Round-trip TTS evaluation")
    parser.add_argument("--text", default=None, help="Single text to test")
    parser.add_argument("--narrator", default="Koharu Rikka", help="Narrator name")
    parser.add_argument("--score", action="store_true", help="Run MOS/emotion/VAD scoring (needs .venv-eval)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    cases = [{"text": args.text, "narrator": args.narrator}] if args.text else CASES
    narrator_override = args.narrator if args.text else None

    ok = run_cases(cases, narrator_override=narrator_override, with_score=args.score, as_json=args.as_json)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
