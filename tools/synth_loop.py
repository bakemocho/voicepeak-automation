#!/usr/bin/env python3
"""Interactive REPL for iterative per-chunk TTS improvement.

Usage:
    python tools/synth_loop.py --chunks chunks.json --out-dir out/

Commands at the prompt:
    play [N]            Play merged.wav or chunk N
    show                Show chunk params table
    edit N key=val      Patch chunk N params (speed, pitch, gap_after_ms, ...)
    edit N '{"k": v}'   JSON patch on chunk N
    caption N "text"    Update oracle_caption then re-oracle + re-synth chunk N
    oracle N            Re-run Irodori oracle for chunk N, update params, re-synth
    gap N MS            Set gap_after_ms for chunk N (shorthand)
    save [path]         Save working chunks.json
    done / quit         Exit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_tools_dir = Path(__file__).parent
sys.path.insert(0, str(_tools_dir))

from synth_annotated import synthesize_one  # noqa: E402
from oracle_synth import (  # noqa: E402
    _J11_CAPTION,
    _J11_CHECKPOINT,
    _J11_REF_WAV,
    _extract_f0_stats,
    _f0_to_voicepeak_params,
    _run_irodori_batch,
)

TOP_DB = 40
CROSSFADE_MS = 30


def _synth_chunk(entry: dict, wav_path: Path, narrator: str) -> bool:
    raw = wav_path.parent / f"_raw_{wav_path.stem}.wav"
    ok = synthesize_one(
        entry["text"], raw, narrator,
        entry.get("emotion", {}),
        entry.get("speed", 1.0),
        entry.get("pitch", 0.0),
    )
    if not ok:
        return False
    y, sr = librosa.load(str(raw), sr=None, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    sf.write(str(wav_path), y_trimmed, sr)
    raw.unlink(missing_ok=True)
    return True


def _merge_all(chunk_wavs: list[Path], chunks: list[dict], sr: int) -> np.ndarray:
    arrays = [librosa.load(str(p), sr=sr, mono=True)[0] for p in chunk_wavs if p.exists()]
    merged = arrays[0]
    for i, nxt in enumerate(arrays[1:]):
        gap_ms = chunks[i].get("gap_after_ms", 0)
        if gap_ms > 0:
            merged = np.concatenate([merged, np.zeros(int(gap_ms / 1000 * sr), dtype=np.float32)])
        fade_n = min(int(CROSSFADE_MS / 1000 * sr), len(merged), len(nxt))
        if fade_n > 0:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            out = np.empty(len(merged) + len(nxt) - fade_n, dtype=np.float32)
            out[: len(merged) - fade_n] = merged[: len(merged) - fade_n]
            out[len(merged) - fade_n : len(merged)] = merged[len(merged) - fade_n :] * (1 - ramp) + nxt[:fade_n] * ramp
            out[len(merged) :] = nxt[fade_n:]
            merged = out
        else:
            merged = np.concatenate([merged, nxt])
    return merged


def _oracle_single(
    entry: dict,
    oracle_wav: Path,
    oracle_mode: str,
    num_steps: int,
    seed: int,
    emotion_floor: float,
    global_f0_midi: float,
    emotion_type: str = "lamenting",
) -> dict | None:
    oracle_wav.parent.mkdir(parents=True, exist_ok=True)
    chunk_task = {"text": entry["text"], "out_wav": str(oracle_wav)}
    if "oracle_caption" in entry:
        chunk_task["caption"] = entry["oracle_caption"]

    if oracle_mode == "j11":
        task = {
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
            "chunks": [chunk_task],
        }
    else:
        task = {
            "checkpoint": "Aratako/Irodori-TTS-500M-v3",
            "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
            "no_ref": True,
            "num_steps": num_steps,
            "seed": seed,
            "chunks": [chunk_task],
        }

    results = _run_irodori_batch(task)
    if not results or not results[0].get("ok"):
        return None

    stats = _extract_f0_stats(oracle_wav)
    return _f0_to_voicepeak_params(stats, global_f0_midi, emotion_type, emotion_floor=emotion_floor)


def _show_table(chunks: list[dict]) -> None:
    print(f"\n  {'#':>3}  {'text':32s}  {'lam':>5}  {'pitch':>6}  {'speed':>5}  {'gap':>5}")
    print("  " + "─" * 62)
    for i, c in enumerate(chunks):
        lam = c.get("emotion", {}).get("lamenting", 0.0)
        pitch = c.get("pitch", 0.0)
        speed = c.get("speed", 1.0)
        gap = c.get("gap_after_ms", "—")
        print(f"  {i+1:>3}  {c['text'][:32]:32s}  {lam:>5.3f}  {pitch:>+6.3f}  {speed:>5.2f}  {gap!s:>5}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive REPL for per-chunk TTS improvement")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--narrator", default="Koharu Rikka")
    parser.add_argument("--oracle-mode", choices=["no-ref", "j11"], default="j11")
    parser.add_argument("--emotion-floor", type=float, default=0.4)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

    chunks: list[dict] = json.loads(args.chunks.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    oracle_dir = args.out_dir / "oracle"
    oracle_dir.mkdir(exist_ok=True)
    merged_path = args.out_dir / "merged.wav"

    # Compute global F0 reference from existing emotion/pitch params
    midis = []
    for c in chunks:
        p = c.get("pitch", 0.0)
        midis.append(librosa.hz_to_midi(150.0) + p * 3.0)
    global_f0_midi = float(np.mean(midis)) if midis else float(librosa.hz_to_midi(200.0))

    # Initial synthesis
    chunk_wavs: list[Path] = []
    sr_detected: int | None = None
    print("\n[loop] initial synthesis...")
    for i, entry in enumerate(chunks):
        wav = chunks_dir / f"{i+1:03d}.wav"
        chunk_wavs.append(wav)
        if wav.exists():
            sys.stderr.write(f"  [{i+1:03d}] cached\n")
            if sr_detected is None:
                _, sr_tmp = librosa.load(str(wav), sr=None, mono=True)
                sr_detected = int(sr_tmp)
            continue
        ok = _synth_chunk(entry, wav, args.narrator)
        status = "ok" if ok else "FAIL"
        sys.stderr.write(f"  [{i+1:03d}] {status}  {entry['text'][:40]!r}\n")
        if ok and sr_detected is None:
            _, sr_tmp = librosa.load(str(wav), sr=None, mono=True)
            sr_detected = int(sr_tmp)

    sr = sr_detected or 44100

    def do_merge() -> None:
        existing = [w for w in chunk_wavs if w.exists()]
        if not existing:
            print("[loop] no WAVs to merge")
            return
        merged = _merge_all(existing, chunks, sr)
        sf.write(str(merged_path), merged, sr)
        gaps = [chunks[i].get("gap_after_ms", 0) for i in range(len(existing) - 1)]
        gap_str = "/".join(str(g) for g in gaps) + "ms"
        print(f"[loop] merged → {merged_path}  ({len(merged)/sr:.2f}s  gaps={gap_str})")

    def do_play(n: int | None = None) -> None:
        path = chunk_wavs[n - 1] if n is not None else merged_path
        if not path.exists():
            print(f"  [play] not found: {path}")
            return
        subprocess.run(["afplay", str(path)])

    do_merge()
    do_play()

    _show_table(chunks)
    print("commands: play [N] | show | edit N key=val | caption N \"...\" | oracle N | gap N MS | save [path] | done")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        parts = line.split(None, 2)
        cmd = parts[0].lower()

        if cmd in ("done", "quit", "exit"):
            break

        elif cmd == "play":
            n = int(parts[1]) if len(parts) > 1 else None
            do_play(n)

        elif cmd == "show":
            _show_table(chunks)

        elif cmd == "gap" and len(parts) >= 3:
            n, ms = int(parts[1]) - 1, int(parts[2])
            chunks[n]["gap_after_ms"] = ms
            print(f"  chunk {n+1} gap_after_ms → {ms}ms")
            do_merge()
            do_play()

        elif cmd == "edit" and len(parts) >= 3:
            n = int(parts[1]) - 1
            try:
                patch = json.loads(parts[2])
            except json.JSONDecodeError:
                patch = {}
                for kv in parts[2].split():
                    k, _, v = kv.partition("=")
                    try:
                        patch[k] = json.loads(v)
                    except Exception:
                        patch[k] = v
            # Handle nested emotion: edit N emotion.lamenting=0.8
            flat_patch, nested_emotion = {}, {}
            for k, v in patch.items():
                if k.startswith("emotion."):
                    nested_emotion[k[8:]] = v
                else:
                    flat_patch[k] = v
            chunks[n].update(flat_patch)
            if nested_emotion:
                chunks[n].setdefault("emotion", {}).update(nested_emotion)
            print(f"  chunk {n+1} patched: {patch}")
            ok = _synth_chunk(chunks[n], chunk_wavs[n], args.narrator)
            if ok:
                do_merge()
                do_play()
            else:
                print(f"  [FAIL] synthesis failed")

        elif cmd == "caption" and len(parts) >= 3:
            n = int(parts[1]) - 1
            caption = parts[2].strip("\"'")
            chunks[n]["oracle_caption"] = caption
            print(f"  chunk {n+1} caption updated → {caption!r}")
            print(f"  [oracle] running for chunk {n+1}...")
            params = _oracle_single(
                chunks[n], oracle_dir / f"{n+1:03d}.wav",
                args.oracle_mode, args.num_steps, args.seed,
                args.emotion_floor, global_f0_midi,
            )
            if params:
                chunks[n]["emotion"] = params["emotion"]
                chunks[n]["pitch"] = params["pitch"]
                print(f"  chunk {n+1} → {params}")
                ok = _synth_chunk(chunks[n], chunk_wavs[n], args.narrator)
                if ok:
                    do_merge()
                    do_play()
            else:
                print("  [FAIL] oracle failed")

        elif cmd == "oracle" and len(parts) >= 2:
            n = int(parts[1]) - 1
            print(f"  [oracle] running for chunk {n+1}...")
            params = _oracle_single(
                chunks[n], oracle_dir / f"{n+1:03d}.wav",
                args.oracle_mode, args.num_steps, args.seed,
                args.emotion_floor, global_f0_midi,
            )
            if params:
                chunks[n]["emotion"] = params["emotion"]
                chunks[n]["pitch"] = params["pitch"]
                print(f"  chunk {n+1} → {params}")
                ok = _synth_chunk(chunks[n], chunk_wavs[n], args.narrator)
                if ok:
                    do_merge()
                    do_play()
            else:
                print("  [FAIL] oracle failed")

        elif cmd == "save":
            save_path = Path(parts[1]) if len(parts) > 1 else args.out_dir / "loop_chunks.json"
            save_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  saved → {save_path}")

        else:
            print("  ? play [N] | show | edit N key=val | caption N \"...\" | oracle N | gap N MS | save | done")

    # Auto-save on exit
    final = args.out_dir / "final_chunks.json"
    final.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[loop] saved → {final}")
    print(f"[loop] merged → {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
