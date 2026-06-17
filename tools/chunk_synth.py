#!/usr/bin/env python3
"""Chunk-based synthesis: synthesize each annotated segment separately, crossfade, and merge.

Workaround for VOICEPEAK's global conditioning (V94-026): per-word emotion
parameters are applied by synthesizing N chunks and concatenating with crossfade.

Input JSON format (same as annotate_script.py output):
    [{"text": "怖いです", "emotion": {"lamenting": 0.2}, "speed": 1.0, "pitch": 0.0},
     {"text": "ね。",    "emotion": {"lamenting": 0.8}, "speed": 0.9, "pitch": 0.0}]

Output:
    <out-dir>/chunks/001.wav, 002.wav, ...  — per-chunk WAVs (after trim)
    <out-dir>/merged.wav                    — crossfaded merge
    <out-dir>/manifest.json                 — metadata

Usage:
    python tools/chunk_synth.py --chunks chunks.json --out-dir out/
    python tools/chunk_synth.py --chunks chunks.json --out-dir out/ --baseline
    python tools/chunk_synth.py --chunks chunks.json --out-dir out/ --crossfade-ms 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Reuse VOICEPEAK invocation helpers from sibling module
_tools_dir = Path(__file__).parent
sys.path.insert(0, str(_tools_dir))
from synth_annotated import (  # noqa: E402
    _emotion_arg,
    _map_emotions,
    synthesize_one,
)

TOP_DB = 40          # librosa.effects.trim silence threshold


def _trim_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load WAV and trim leading/trailing silence."""
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return y_trimmed, sr


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    """Overlap-add crossfade between two mono arrays."""
    fade_n = min(int(fade_ms / 1000 * sr), len(a), len(b))
    if fade_n <= 0:
        return np.concatenate([a, b])

    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    out = np.empty(len(a) + len(b) - fade_n, dtype=np.float32)
    out[: len(a) - fade_n] = a[: len(a) - fade_n]
    out[len(a) - fade_n : len(a)] = a[len(a) - fade_n :] * (1 - ramp) + b[:fade_n] * ramp
    out[len(a) :] = b[fade_n:]
    return out


def merge_wavs(
    chunk_paths: list[Path],
    sr: int,
    crossfade_ms: int,
) -> np.ndarray:
    """Crossfade-merge pre-trimmed chunk WAVs into a single array."""
    arrays = []
    for p in chunk_paths:
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        arrays.append(y)

    merged = arrays[0]
    for nxt in arrays[1:]:
        merged = _crossfade(merged, nxt, sr, crossfade_ms)
    return merged


def synthesize_baseline(
    chunks: list[dict],
    out_wav: Path,
    narrator: str,
) -> bool:
    """Single VOICEPEAK call with all text concatenated (baseline comparison)."""
    combined_text = "".join(c["text"] for c in chunks)
    # Use first chunk's params for baseline
    first = chunks[0]
    emotion = first.get("emotion", {})
    speed = first.get("speed", 1.0)
    pitch = first.get("pitch", 0.0)
    return synthesize_one(combined_text, out_wav, narrator, emotion, speed, pitch)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk-based VOICEPEAK synthesis with crossfade")
    parser.add_argument("--chunks", type=Path, help="Chunk annotation JSON (default: stdin)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--narrator", default="Koharu Rikka", help="VOICEPEAK narrator")
    parser.add_argument("--crossfade-ms", type=int, default=20, metavar="MS",
                        help="Crossfade duration in ms (default: 20)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also synthesize a single-call baseline for comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print synthesis commands without running VOICEPEAK")
    parser.add_argument("--play", action="store_true",
                        help="Play merged.wav with afplay after synthesis")
    args = parser.parse_args()

    if args.chunks:
        chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    elif not sys.stdin.isatty():
        chunks = json.loads(sys.stdin.read())
    else:
        parser.error("Provide --chunks or pipe JSON via stdin")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    # --- Dry run ---
    if args.dry_run:
        for i, entry in enumerate(chunks):
            text = entry["text"]
            emotion = _map_emotions(entry.get("emotion", {}), args.narrator)
            em_arg = _emotion_arg(emotion)
            speed = round(entry.get("speed", 1.0) * 100)
            pitch = round(entry.get("pitch", 0.0) * 10)
            print(f"[{i+1:03d}] voicepeak -s {text!r}"
                  f"{' --emotion ' + em_arg if em_arg else ''}"
                  f"{' --speed ' + str(speed) if speed != 100 else ''}"
                  f"{' --pitch ' + str(pitch) if pitch != 0 else ''}")
        if args.baseline:
            combined = "".join(c["text"] for c in chunks)
            print(f"[baseline] voicepeak -s {combined!r}")
        return 0

    # --- Synthesize chunks ---
    chunk_wavs: list[Path] = []
    manifest: list[dict] = []
    all_ok = True
    sr_detected: int | None = None

    for i, entry in enumerate(chunks):
        text = entry["text"]
        emotion = entry.get("emotion", {})
        speed = entry.get("speed", 1.0)
        pitch = entry.get("pitch", 0.0)
        wav = chunks_dir / f"{i+1:03d}.wav"

        raw_wav = chunks_dir / f"_raw_{i+1:03d}.wav"
        ok = synthesize_one(text, raw_wav, args.narrator, emotion, speed, pitch)

        if not ok:
            sys.stderr.write(f"[FAIL] chunk {i+1:03d}: {text!r}\n")
            all_ok = False
            manifest.append({"index": i + 1, "text": text, "status": "FAIL"})
            continue

        # Trim and save
        y, sr = _trim_wav(raw_wav)
        if sr_detected is None:
            sr_detected = sr
        sf.write(str(wav), y, sr)
        raw_wav.unlink()  # remove untrimmed copy

        mapped_em = _map_emotions(emotion, args.narrator)
        em_arg = _emotion_arg(mapped_em)
        record = {
            "index": i + 1,
            "text": text,
            "wav": str(wav),
            "status": "ok",
            "duration_s": round(len(y) / sr, 3),
        }
        if em_arg:
            record["emotion"] = em_arg
        if speed != 1.0:
            record["speed"] = speed
        if pitch != 0.0:
            record["pitch"] = pitch
        manifest.append(record)
        chunk_wavs.append(wav)
        print(f"[{i+1:03d}] ok  {wav.name}  {text[:40]!r}  ({record['duration_s']}s)")

    # --- Merge ---
    if len(chunk_wavs) >= 2:
        sr = sr_detected or 44100
        merged = merge_wavs(chunk_wavs, sr, args.crossfade_ms)
        merged_path = args.out_dir / "merged.wav"
        sf.write(str(merged_path), merged, sr)
        print(f"\nmerged → {merged_path}  ({len(merged)/sr:.2f}s, crossfade={args.crossfade_ms}ms)")
        if args.play:
            import subprocess
            subprocess.run(["afplay", str(merged_path)])
    elif len(chunk_wavs) == 1:
        import shutil
        merged_path = args.out_dir / "merged.wav"
        shutil.copy2(chunk_wavs[0], merged_path)
        print(f"\nmerged → {merged_path}  (single chunk, no crossfade)")
    else:
        sys.stderr.write("No chunks synthesized successfully.\n")
        all_ok = False

    # --- Baseline ---
    if args.baseline:
        baseline_path = args.out_dir / "baseline.wav"
        ok = synthesize_baseline(chunks, baseline_path, args.narrator)
        status = "ok" if ok else "FAIL"
        print(f"baseline → {baseline_path}  [{status}]")
        manifest.append({"index": "baseline", "text": "".join(c["text"] for c in chunks),
                         "wav": str(baseline_path), "status": status})

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest → {manifest_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
