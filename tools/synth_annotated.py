#!/usr/bin/env python3
"""Synthesize WAVs from annotate_script.py JSON output.

Reads annotation JSON (from stdin or --annotations) and synthesizes each
sentence with VOICEPEAK CLI using emotion/speed/pitch from the annotations.

Usage:
    python tools/annotate_script.py --text "..." | python tools/synth_annotated.py --out-dir out/
    python tools/synth_annotated.py --annotations annotations.json --out-dir out/ --narrator "Koharu Rikka"

Output:
    out/001.wav, out/002.wav, ... — one WAV per annotation entry
    out/manifest.json              — {index, text, emotion, files, ...}

VOICEPEAK emotion range: 0–100 (integer).
Annotation emotion range: 0.0–1.0 (float) → multiplied by 100.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VOICEPEAK_BIN = "/Applications/voicepeak.app/Contents/MacOS/voicepeak"

# Speed/pitch annotation range → CLI range mapping
# annotation speed 1.0 → CLI 100; 0.5 → 50; 2.0 → 200
# annotation pitch 0.0 → CLI 0; semitone values map linearly
_SPEED_SCALE = 100   # multiply annotation speed by this
_PITCH_SCALE = 10    # multiply annotation pitch by this (semitone → VOICEPEAK units)

# Generic emotion → narrator-specific emotion mapping.
# Generic labels from LLM: happy, sad, angry, calm, excited, fearful, disgusted, surprised
# Query narrator-specific labels: voicepeak --list-emotion "<narrator>"
_EMOTION_MAP: dict[str, dict[str, str]] = {
    "Koharu Rikka": {
        "happy":     "hightension",
        "excited":   "hightension",
        "surprised": "hightension",
        "sad":       "lamenting",
        "fearful":   "lamenting",
        "angry":     "livid",
        "disgusted": "despising",
        "calm":      "narration",
        "neutral":   "narration",
    },
}
_EMOTION_MAP_DEFAULT = {
    "happy": "happy", "sad": "sad", "angry": "angry", "calm": "calm",
}


def _map_emotions(emotion: dict[str, float], narrator: str) -> dict[str, float]:
    """Translate generic emotion labels to narrator-specific ones, combining duplicates."""
    mapping = _EMOTION_MAP.get(narrator, _EMOTION_MAP_DEFAULT)
    out: dict[str, float] = {}
    for generic, val in emotion.items():
        specific = mapping.get(generic, generic)
        out[specific] = max(out.get(specific, 0.0), val)
    return out


def _emotion_arg(emotion: dict[str, float]) -> str | None:
    """Convert {happy: 0.5, sad: 0.2} → 'happy=50,sad=20'. Returns None if all zero."""
    parts = []
    for name, val in emotion.items():
        int_val = max(0, min(100, round(val * 100)))
        if int_val > 0:
            parts.append(f"{name}={int_val}")
    return ",".join(parts) if parts else None


def synthesize_one(
    text: str,
    out_wav: Path,
    narrator: str,
    emotion: dict[str, float] | None = None,
    speed: float = 1.0,
    pitch: float = 0.0,
) -> bool:
    cmd = [VOICEPEAK_BIN, "-s", text, "--narrator", narrator, "-o", str(out_wav)]

    speed_int = max(50, min(200, round(speed * _SPEED_SCALE)))
    if speed_int != 100:
        cmd += ["--speed", str(speed_int)]

    pitch_int = max(-300, min(300, round(pitch * _PITCH_SCALE)))
    if pitch_int != 0:
        cmd += ["--pitch", str(pitch_int)]

    if emotion:
        mapped = _map_emotions(emotion, narrator)
        em_arg = _emotion_arg(mapped)
        if em_arg:
            cmd += ["--emotion", em_arg]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return out_wav.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize WAVs from annotation JSON")
    parser.add_argument("--annotations", type=Path, help="Annotation JSON file (default: stdin)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for WAVs")
    parser.add_argument("--narrator", default="Koharu Rikka", help="VOICEPEAK narrator")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without synthesizing")
    args = parser.parse_args()

    if args.annotations:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    elif not sys.stdin.isatty():
        annotations = json.loads(sys.stdin.read())
    else:
        parser.error("Provide --annotations or pipe annotation JSON via stdin")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    all_ok = True

    for i, entry in enumerate(annotations):
        text = entry["text"]
        emotion = entry.get("emotion", {})
        speed = entry.get("speed", 1.0)
        pitch = entry.get("pitch", 0.0)
        wav = args.out_dir / f"{i+1:03d}.wav"

        mapped_emotion = _map_emotions(emotion, args.narrator) if emotion else {}
        em_arg = _emotion_arg(mapped_emotion) if mapped_emotion else None
        speed_int = max(50, min(200, round(speed * _SPEED_SCALE)))
        pitch_int = max(-300, min(300, round(pitch * _PITCH_SCALE)))

        if args.dry_run:
            cmd_preview = f"voicepeak -s {text!r} --narrator {args.narrator!r} -o {wav}"
            if em_arg:
                cmd_preview += f" --emotion {em_arg}"
            if speed_int != 100:
                cmd_preview += f" --speed {speed_int}"
            if pitch_int != 0:
                cmd_preview += f" --pitch {pitch_int}"
            print(f"[{i+1:03d}] {cmd_preview}")
            record = {"index": i + 1, "text": text, "wav": str(wav), "status": "dry-run"}
        else:
            ok = synthesize_one(text, wav, args.narrator, emotion, speed, pitch)
            status = "ok" if ok else "FAIL"
            if not ok:
                all_ok = False
                sys.stderr.write(f"[FAIL] {i+1:03d}: {text!r}\n")
            else:
                print(f"[{i+1:03d}] {status}  {wav.name}  {text[:40]!r}")
            record = {"index": i + 1, "text": text, "wav": str(wav), "status": status}

        if em_arg:
            record["emotion"] = em_arg
        if speed_int != 100:
            record["speed"] = speed_int
        if pitch_int != 0:
            record["pitch"] = pitch_int
        manifest.append(record)

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest → {manifest_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
