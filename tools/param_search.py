#!/usr/bin/env python3
"""Random parameter search for chunk-based VOICEPEAK synthesis.

Generates N candidate variants by perturbing baseline chunk params,
synthesizes all, and presents a comparison table for selection.

Usage:
    # Generate and synthesize 5 candidates around baseline
    python tools/param_search.py --chunks baseline.json --out-dir search/ --n 5

    # Vary only specific chunks
    python tools/param_search.py --chunks baseline.json --out-dir search/ --n 5 --vary-chunks 4,5

    # Play candidate K
    python tools/param_search.py --out-dir search/ --play 3

    # Adopt candidate K as new baseline (writes baseline.json)
    python tools/param_search.py --out-dir search/ --adopt 3
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

_tools_dir = Path(__file__).parent
sys.path.insert(0, str(_tools_dir))
from synth_annotated import synthesize_one  # noqa: E402
from score_wav import score_expressiveness  # noqa: E402
from case_library import add_case, _DEFAULT_LIBRARY as _CASE_LIBRARY_DEFAULT  # noqa: E402

_VENV_EVAL_PYTHON = _tools_dir.parent / ".venv-eval" / "bin" / "python"

TOP_DB = 40
CROSSFADE_MS = 30

# Default perturbation ranges (±delta around baseline value)
DEFAULT_RANGES = {
    "speed":   0.07,   # ±0.07 (e.g. 1.05 → 0.98..1.12)
    "pitch":   0.25,   # ±0.25 annotation units
    "lam":     0.15,   # ±0.15 lamenting intensity
    "gap":     80,     # ±80ms gap_after_ms
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _perturb_chunk(entry: dict, rng: random.Random, ranges: dict) -> dict:
    c = dict(entry)
    c["speed"] = round(_clamp(
        c.get("speed", 1.0) + rng.uniform(-ranges["speed"], ranges["speed"]),
        0.5, 2.0,
    ), 3)
    c["pitch"] = round(_clamp(
        c.get("pitch", 0.0) + rng.uniform(-ranges["pitch"], ranges["pitch"]),
        -1.0, 1.0,
    ), 3)
    em = dict(c.get("emotion", {}))
    for k in em:
        em[k] = round(_clamp(
            em[k] + rng.uniform(-ranges["lam"], ranges["lam"]),
            0.0, 1.0,
        ), 3)
    c["emotion"] = em
    if "gap_after_ms" in c:
        c["gap_after_ms"] = max(50, int(
            c["gap_after_ms"] + rng.randint(-ranges["gap"], ranges["gap"])
        ))
    return c


def _trim_synth(entry: dict, wav_path: Path, narrator: str) -> bool:
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
    y_t, _ = librosa.effects.trim(y, top_db=TOP_DB)
    sf.write(str(wav_path), y_t, sr)
    raw.unlink(missing_ok=True)
    return True


def _merge(chunk_wavs: list[Path], chunks: list[dict], sr: int) -> np.ndarray:
    arrays = [librosa.load(str(p), sr=sr, mono=True)[0] for p in chunk_wavs]
    merged = arrays[0]
    for i, nxt in enumerate(arrays[1:]):
        gap_ms = chunks[i].get("gap_after_ms", 0)
        if gap_ms > 0:
            merged = np.concatenate([merged, np.zeros(int(gap_ms / 1000 * sr), np.float32)])
        fn = min(int(CROSSFADE_MS / 1000 * sr), len(merged), len(nxt))
        if fn > 0:
            r = np.linspace(0, 1, fn, dtype=np.float32)
            out = np.empty(len(merged) + len(nxt) - fn, np.float32)
            out[:len(merged)-fn] = merged[:len(merged)-fn]
            out[len(merged)-fn:len(merged)] = merged[len(merged)-fn:] * (1-r) + nxt[:fn] * r
            out[len(merged):] = nxt[fn:]
            merged = out
        else:
            merged = np.concatenate([merged, nxt])
    return merged


def _score_candidate(wav_path: Path) -> dict:
    """Score a merged WAV with expressiveness (always) and UTMOS (if venv-eval exists)."""
    scores: dict = {}
    try:
        expr = score_expressiveness(wav_path)
        scores["expr_score"] = expr["expr_score"]
        scores["f0_range_st"] = expr["f0_range_st"]
    except Exception as e:
        sys.stderr.write(f"  [score] expr failed: {e}\n")

    if _VENV_EVAL_PYTHON.exists():
        try:
            proc = subprocess.run(
                [str(_VENV_EVAL_PYTHON), str(_tools_dir / "score_wav.py"),
                 "--mos-only", "--json", str(wav_path)],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                # stdout may contain model-load messages before the JSON line
                last_line = proc.stdout.strip().splitlines()[-1]
                scores["mos"] = json.loads(last_line)["mos"]
        except Exception as e:
            sys.stderr.write(f"  [score] mos failed: {e}\n")

    return scores


def _show_table(candidates: list[dict], baseline: list[dict]) -> None:
    n_chunks = len(baseline)
    chunk_labels = "  ".join(f"C{i+1}" for i in range(n_chunks))
    has_mos = any("mos" in c.get("scores", {}) for c in candidates)
    score_hdr = "  MOS  expr" if has_mos else "  expr"

    print(f"\n  {'#':>4}  {'seed':>6}  {chunk_labels}{score_hdr}")
    print(f"  {'':>4}  {'':>6}  " + "  ".join(f"{'lam/spd/pit':11s}" for _ in range(n_chunks)))
    print("  " + "─" * (14 + 13 * n_chunks + (14 if has_mos else 7)))

    # Baseline row
    row = "  ".join(
        f"{c.get('emotion',{}).get('lamenting',0):.2f}/{c.get('speed',1):.2f}/{c.get('pitch',0):+.2f}"
        for c in baseline
    )
    print(f"  {'base':>4}  {'':>6}  {row}")
    print()

    # Candidate rows
    best_k = None
    best_mos = -1.0
    for k, cand in enumerate(candidates):
        if not cand.get("ok"):
            continue
        sc = cand.get("scores", {})
        mos = sc.get("mos")
        if mos is not None and mos > best_mos:
            best_mos, best_k = mos, k

    for k, cand in enumerate(candidates):
        chunks = cand["chunks"]
        seed = cand["seed"]
        row = "  ".join(
            f"{c.get('emotion',{}).get('lamenting',0):.2f}/{c.get('speed',1):.2f}/{c.get('pitch',0):+.2f}"
            for c in chunks
        )
        dur = cand.get("duration_s", 0)
        ok_s = "ok" if cand.get("ok") else "FAIL"
        sc = cand.get("scores", {})
        score_s = ""
        if has_mos:
            mos_s = f"{sc['mos']:.3f}" if "mos" in sc else "  — "
            score_s = f"  {mos_s}"
        if "expr_score" in sc:
            score_s += f"  {sc['expr_score']:.3f}"
        marker = " ★" if k == best_k else ""
        print(f"  {k+1:>4}  {seed:>6}  {row}  ({dur:.1f}s {ok_s}){score_s}{marker}")
    print()
    if best_k is not None:
        print(f"  Best by MOS: cand {best_k+1}  (MOS={best_mos:.3f})")
        print()


def cmd_generate(args: argparse.Namespace) -> None:
    baseline: list[dict] = json.loads(args.chunks.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vary_chunks: set[int] | None = None
    if args.vary_chunks:
        vary_chunks = {int(x) - 1 for x in args.vary_chunks.split(",")}

    ranges = dict(DEFAULT_RANGES)
    if args.speed_range is not None:
        ranges["speed"] = args.speed_range
    if args.pitch_range is not None:
        ranges["pitch"] = args.pitch_range
    if args.lam_range is not None:
        ranges["lam"] = args.lam_range

    rng_seed = args.seed if args.seed else random.randint(0, 99999)
    rng = random.Random(rng_seed)

    candidates = []
    for k in range(args.n):
        seed_k = rng.randint(0, 99999)
        rng_k = random.Random(seed_k)
        chunks_k = []
        for i, entry in enumerate(baseline):
            if vary_chunks is None or i in vary_chunks:
                chunks_k.append(_perturb_chunk(entry, rng_k, ranges))
            else:
                chunks_k.append(dict(entry))
        candidates.append({"seed": seed_k, "chunks": chunks_k, "ok": False})

    # Synthesize
    sr_detected: int | None = None
    for k, cand in enumerate(candidates):
        cand_dir = args.out_dir / f"cand_{k+1:02d}"
        cand_dir.mkdir(exist_ok=True)
        chunk_wavs = []
        all_ok = True
        for i, entry in enumerate(cand["chunks"]):
            wav = cand_dir / f"{i+1:03d}.wav"
            ok = _trim_synth(entry, wav, args.narrator)
            if not ok:
                sys.stderr.write(f"  [FAIL] cand {k+1} chunk {i+1}\n")
                all_ok = False
            else:
                if sr_detected is None:
                    _, sr_detected = librosa.load(str(wav), sr=None, mono=True)
                    sr_detected = int(sr_detected)
                chunk_wavs.append(wav)
            sys.stderr.write(f"  [{k+1:02d}/{i+1:03d}] {'ok' if ok else 'FAIL'}  {entry['text'][:30]!r}\n")

        if all_ok and len(chunk_wavs) == len(cand["chunks"]):
            sr = sr_detected or 44100
            merged = _merge(chunk_wavs, cand["chunks"], sr)
            merged_path = cand_dir / "merged.wav"
            sf.write(str(merged_path), merged, sr)
            cand["ok"] = True
            cand["merged_wav"] = str(merged_path)
            cand["duration_s"] = round(len(merged) / sr, 2)
            print(f"  cand {k+1:02d} → {merged_path}  ({cand['duration_s']}s)")

    # Score each successful candidate
    ok_cands = [c for c in candidates if c.get("ok")]
    if ok_cands:
        sys.stderr.write(f"\n  [score] scoring {len(ok_cands)} candidates...\n")
        for cand in ok_cands:
            wav = Path(cand["merged_wav"])
            cand["scores"] = _score_candidate(wav)
            sc = cand["scores"]
            mos_s = f"MOS={sc['mos']:.3f}" if "mos" in sc else ""
            expr_s = f"expr={sc.get('expr_score','?')}"
            sys.stderr.write(f"  [score] cand seed={cand['seed']}  {mos_s}  {expr_s}\n")

    # Save manifest
    manifest = {"baseline": str(args.chunks), "candidates": candidates, "ranges": ranges}
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _show_table(candidates, baseline)
    print(f"manifest → {manifest_path}")
    print("next: --adopt K")


def cmd_play(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    cand = manifest["candidates"][args.play - 1]
    wav = cand.get("merged_wav")
    if not wav or not Path(wav).exists():
        print(f"cand {args.play}: no WAV found")
        return
    print(f"playing cand {args.play}: {wav}")
    subprocess.run(["afplay", wav])


def cmd_adopt(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    cand = manifest["candidates"][args.adopt - 1]
    baseline_src = Path(manifest["baseline"])
    out_path = args.out_dir / "adopted.json"
    out_path.write_text(
        json.dumps(cand["chunks"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adopted cand {args.adopt} → {out_path}")
    print(f"use as next baseline: --chunks {out_path}")

    # Also show diff from baseline
    baseline = json.loads(baseline_src.read_text(encoding="utf-8"))
    _show_table([cand], baseline)

    mos = cand.get("scores", {}).get("mos")

    # Save to case library if --situation provided
    if getattr(args, "situation", None):
        case_id = add_case(
            chunks=cand["chunks"],
            situation=args.situation,
            narrator=getattr(args, "narrator", "Koharu Rikka"),
            scene_mode=getattr(args, "scene_mode", "") or "",
            notes=getattr(args, "notes", "") or "",
            mos=mos,
            source=f"param_search seed={cand['seed']}",
            library=getattr(args, "library", None),
        )
        print(f"saved to case library: {case_id}")
    else:
        mos_flag = f" --mos {mos:.3f}" if mos is not None else ""
        print(
            f"\nhint: save to case library:\n"
            f"  python tools/case_library.py add \\\n"
            f"    --chunks {out_path} \\\n"
            f"    --situation \"<シーン説明>\"{mos_flag} \\\n"
            f"    --notes \"<修正の経緯・指示>\""
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Random parameter search for chunk synthesis")
    parser.add_argument("--out-dir", type=Path, required=True)

    # Generate mode
    parser.add_argument("--chunks", type=Path, help="Baseline chunks JSON")
    parser.add_argument("--n", type=int, default=5, help="Number of candidates (default: 5)")
    parser.add_argument("--narrator", default="Koharu Rikka")
    parser.add_argument("--vary-chunks", help="Comma-separated chunk indices to vary (default: all)")
    parser.add_argument("--seed", type=int, help="Master RNG seed")
    parser.add_argument("--speed-range", type=float)
    parser.add_argument("--pitch-range", type=float)
    parser.add_argument("--lam-range", type=float)

    # Inspect / select modes
    parser.add_argument("--play", type=int, metavar="K", help="Play candidate K")
    parser.add_argument("--adopt", type=int, metavar="K", help="Adopt candidate K as new baseline")
    parser.add_argument("--show", action="store_true", help="Show table from existing manifest")

    # Case library options (used with --adopt)
    parser.add_argument("--situation", help="Save adopted case to library with this situation description")
    parser.add_argument("--scene-mode", help="scene_mode tag for case library (e.g. dramatic_fear)")
    parser.add_argument("--notes", help="Correction history / instructions for case library")
    parser.add_argument("--library", type=Path, default=_CASE_LIBRARY_DEFAULT,
                        help="Case library path (default: case_library/cases.jsonl)")

    args = parser.parse_args()

    if args.play:
        cmd_play(args)
    elif args.adopt:
        cmd_adopt(args)
    elif args.show:
        manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
        baseline = json.loads(Path(manifest["baseline"]).read_text(encoding="utf-8"))
        _show_table(manifest["candidates"], baseline)
    elif args.chunks:
        cmd_generate(args)
    else:
        parser.error("Provide --chunks (generate) or --play/--adopt/--show (inspect)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
