#!/usr/bin/env python3
"""Irodori-TTS oracle: generate reference WAVs and extract prosody for VOICEPEAK param estimation.

Synthesizes each chunk with Irodori-TTS (one model load via irodori_batch_runner.py),
extracts per-chunk F0 statistics, maps them to VOICEPEAK params (emotion intensity,
pitch offset), and outputs an updated chunk JSON ready for chunk_synth.py.

Oracle modes (--oracle-mode):
  no-ref      Plain Irodori synthesis, no ref-wav (default voice).
  j11         Irodori-11 anchor: uses 11_j11_less_childish.wav as ref-wav with
              the established J11 caption (600M-VoiceDesign checkpoint).
  voicepeak   First synthesizes a VOICEPEAK baseline, then uses that WAV as
              Irodori ref-wav to anchor prosody in VOICEPEAK timbre space.

F0 mapping (V94-030):
  f0_median (Hz)  → pitch offset (annotation semitones, deviation from global median)
  f0_range (st)   → emotion intensity (wide = more intense; saturates at 10 st)

Usage:
    python tools/oracle_synth.py --chunks chunks.json --out-dir out/oracle/
    python tools/oracle_synth.py --chunks chunks.json --out-dir out/ --oracle-mode j11
    python tools/oracle_synth.py --chunks chunks.json --out-dir out/ --oracle-mode voicepeak
    python tools/oracle_synth.py --chunks chunks.json --out-dir out/ | chunk_synth.py ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np

_IRODORI_ROOT = Path("/Users/bakemocho/gitwork_bk/Irodori-TTS")
_IRODORI_PYTHON = _IRODORI_ROOT / ".venv" / "bin" / "python"
_BATCH_RUNNER = Path(__file__).parent / "irodori_batch_runner.py"

_J11_REF_WAV = Path(
    "/Users/bakemocho/Library/Application Support/seimeido/tts-trials"
    "/irodori-tts/2026-06-15-bokukko-jitome/11_j11_less_childish.wav"
)
_J11_CAPTION = "若すぎない僕っこ女性キャラクターの声。少し低めで落ち着きがある。ジト目で淡々と、面倒そうに話す。子供っぽくせず、自然に聞こえる。"
_J11_CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"

# F0 mapping constants
_F0_RANGE_SATURATE_ST = 10.0   # semitone range → emotion 1.0
_PITCH_SCALE_ST = 3.0           # ±3 st deviation caps pitch at ±1.0


def _extract_f0_stats(wav_path: Path) -> dict:
    """Extract voiced F0 median, range, and RMS from a WAV file."""
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]

    if len(voiced_f0) < 5:
        return {"f0_median_hz": 150.0, "f0_range_st": 0.0, "rms": float(np.sqrt(np.mean(y**2)))}

    f0_midi = librosa.hz_to_midi(voiced_f0)
    return {
        "f0_median_hz": float(np.median(voiced_f0)),
        "f0_median_midi": float(np.median(f0_midi)),
        "f0_range_st": float(f0_midi.max() - f0_midi.min()),
        "rms": float(np.sqrt(np.mean(y**2))),
        "duration_sec": float(len(y) / sr),
    }


def _f0_to_voicepeak_params(
    stats: dict, global_f0_median_midi: float, base_emotion_type: str
) -> dict:
    """Map oracle F0 stats to VOICEPEAK annotation params."""
    f0_range = stats.get("f0_range_st", 0.0)
    f0_midi = stats.get("f0_median_midi", global_f0_median_midi)

    # emotion intensity: F0 range normalized, saturates at _F0_RANGE_SATURATE_ST
    intensity = min(f0_range / _F0_RANGE_SATURATE_ST, 1.0)

    # pitch offset: semitone deviation from global median, capped at ±_PITCH_SCALE_ST
    pitch_dev_st = f0_midi - global_f0_median_midi
    pitch_offset = float(np.clip(pitch_dev_st / _PITCH_SCALE_ST, -1.0, 1.0))

    return {
        "emotion": {base_emotion_type: round(intensity, 3)},
        "pitch": round(pitch_offset, 3),
    }


def _run_irodori_batch(task: dict) -> list[dict]:
    """Run irodori_batch_runner.py in Irodori's venv and return results."""
    task_json = json.dumps(task, ensure_ascii=False)
    proc = subprocess.run(
        [str(_IRODORI_PYTHON), str(_BATCH_RUNNER)],
        input=task_json,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"[oracle] batch runner failed:\n{proc.stderr}\n")
        return []
    # Print stderr (progress) to our stderr
    sys.stderr.write(proc.stderr)
    return json.loads(proc.stdout)


def _build_task(
    chunks: list[dict],
    oracle_dir: Path,
    oracle_mode: str,
    num_steps: int,
    seed: int,
) -> dict:
    """Build irodori_batch_runner task JSON for the given oracle mode."""
    task_chunks = [
        {"text": entry["text"], "out_wav": str(oracle_dir / f"oracle_{i+1:03d}.wav")}
        for i, entry in enumerate(chunks)
    ]

    if oracle_mode == "j11":
        return {
            "checkpoint": _J11_CHECKPOINT,
            "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "ref_wav": str(_J11_REF_WAV),
            "caption": _J11_CAPTION,
            "no_ref": False,
            "ref_normalize_db": -16.0,
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 3.2,
            "cfg_scale_speaker": 5.0,
            "num_steps": num_steps,
            "seed": seed,
            "chunks": task_chunks,
        }
    else:
        # no-ref (also base for voicepeak mode's Irodori pass)
        return {
            "checkpoint": "Aratako/Irodori-TTS-500M-v3",
            "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "no_ref": True,
            "num_steps": num_steps,
            "seed": seed,
            "chunks": task_chunks,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Irodori-TTS prosody oracle for VOICEPEAK params")
    parser.add_argument("--chunks", type=Path, help="Input chunk JSON (default: stdin)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--oracle-mode",
        choices=["no-ref", "j11", "voicepeak"],
        default="no-ref",
        help="Oracle mode: no-ref (default Irodori), j11 (Irodori-11 anchor), voicepeak (VOICEPEAK ref)",
    )
    parser.add_argument("--emotion-type", default="lamenting",
                        help="Base emotion type for intensity mapping (default: lamenting)")
    parser.add_argument("--num-steps", type=int, default=40,
                        help="Irodori inference steps (default: 40; use 4 for fast smoke)")
    parser.add_argument("--seed", type=int, default=20260618, help="Random seed")
    parser.add_argument("--narrator", default="Koharu Rikka",
                        help="VOICEPEAK narrator (for voicepeak mode baseline)")
    parser.add_argument("--out", type=Path, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    if args.chunks:
        chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    elif not sys.stdin.isatty():
        chunks = json.loads(sys.stdin.read())
    else:
        parser.error("Provide --chunks or pipe JSON via stdin")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = args.out_dir / "oracle_wavs"
    oracle_dir.mkdir(exist_ok=True)

    # --- voicepeak mode: generate VOICEPEAK baseline first as ref-wav ---
    if args.oracle_mode == "voicepeak":
        sys.stderr.write("[oracle] voicepeak mode: generating VOICEPEAK baseline first...\n")
        _tools_dir = Path(__file__).parent
        sys.path.insert(0, str(_tools_dir))
        from synth_annotated import synthesize_one

        vp_wavs = []
        for i, entry in enumerate(chunks):
            wav = oracle_dir / f"vp_ref_{i+1:03d}.wav"
            ok = synthesize_one(entry["text"], wav, args.narrator,
                                entry.get("emotion", {}), entry.get("speed", 1.0),
                                entry.get("pitch", 0.0))
            if not ok:
                sys.stderr.write(f"[oracle] VOICEPEAK FAIL chunk {i+1:03d}\n")
                vp_wavs.append(None)
            else:
                vp_wavs.append(wav)

        # For each chunk: use its VOICEPEAK wav as Irodori ref
        # Run Irodori once per chunk with different ref_wav (separate batch calls)
        oracle_results = []
        for i, entry in enumerate(chunks):
            ref = vp_wavs[i]
            if ref is None:
                oracle_results.append(None)
                continue
            task = {
                "checkpoint": "Aratako/Irodori-TTS-500M-v3",
                "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
                "ref_wav": str(ref),
                "no_ref": False,
                "ref_normalize_db": -16.0,
                "cfg_scale_text": 3.0,
                "cfg_scale_speaker": 3.0,
                "num_steps": args.num_steps,
                "seed": args.seed,
                "chunks": [{"text": entry["text"],
                             "out_wav": str(oracle_dir / f"oracle_{i+1:03d}.wav")}],
            }
            res = _run_irodori_batch(task)
            oracle_results.extend(res)
    else:
        task = _build_task(chunks, oracle_dir, args.oracle_mode, args.num_steps, args.seed)
        oracle_results = _run_irodori_batch(task)

    # --- Extract F0 stats from oracle WAVs ---
    stats_list = []
    for res in oracle_results:
        if res and res.get("ok"):
            stats = _extract_f0_stats(Path(res["out_wav"]))
            stats["oracle_wav"] = res["out_wav"]
        else:
            stats = {"f0_median_hz": 150.0, "f0_range_st": 0.0, "rms": 0.0}
        stats_list.append(stats)

    # Global F0 median (for pitch deviation reference)
    valid_midis = [s["f0_median_midi"] for s in stats_list if "f0_median_midi" in s]
    global_f0_midi = float(np.median(valid_midis)) if valid_midis else librosa.hz_to_midi(150.0)

    # --- Map to VOICEPEAK params and build output ---
    result_chunks = []
    for i, (entry, stats) in enumerate(zip(chunks, stats_list)):
        updated = dict(entry)
        vp_params = _f0_to_voicepeak_params(stats, global_f0_midi, args.emotion_type)

        # Override emotion intensity; keep user-set type if different from emotion_type
        updated["emotion"] = vp_params["emotion"]
        updated["pitch"] = vp_params["pitch"]

        # Attach oracle diagnostics
        updated["_oracle"] = {
            "mode": args.oracle_mode,
            "f0_median_hz": round(stats.get("f0_median_hz", 0), 1),
            "f0_range_st": round(stats.get("f0_range_st", 0), 2),
            "intensity": round(vp_params["emotion"].get(args.emotion_type, 0), 3),
        }
        if "oracle_wav" in stats:
            updated["_oracle"]["oracle_wav"] = stats["oracle_wav"]

        sys.stderr.write(
            f"  [{i+1:03d}] {entry['text']!r:30s}"
            f"  F0={stats.get('f0_median_hz', 0):.0f}Hz"
            f"  range={stats.get('f0_range_st', 0):.1f}st"
            f"  → {args.emotion_type}={updated['emotion'][args.emotion_type]:.3f}"
            f"  pitch={updated['pitch']:+.3f}\n"
        )
        result_chunks.append(updated)

    # Strip internal _oracle fields from final output (keep for verbose inspection)
    out_chunks = []
    for entry in result_chunks:
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        out_chunks.append(clean)

    out_json = json.dumps(out_chunks, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(out_json, encoding="utf-8")
        sys.stderr.write(f"[oracle] wrote → {args.out}\n")
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
