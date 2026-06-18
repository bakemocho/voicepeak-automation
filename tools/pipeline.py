#!/usr/bin/env python3
"""End-to-end TTS synthesis pipeline.

Orchestrates: annotate_script → boundary_search → param_search → [case_library]
Auto-picks best MOS at each stage. Falls back to expr_score when MOS unavailable.

Usage:
    # From raw script text
    python tools/pipeline.py \\
        --text "台詞テキスト" \\
        --out-dir /tmp/scene_01/ \\
        --scene-mode drama \\
        --narrator "Koharu Rikka"

    # From existing chunks.json (skip annotation)
    python tools/pipeline.py \\
        --chunks baseline.json \\
        --out-dir /tmp/scene_01/

    # Skip boundary search
    python tools/pipeline.py \\
        --chunks baseline.json --out-dir /tmp/out/ --no-boundary

    # Skip param search
    python tools/pipeline.py \\
        --chunks baseline.json --out-dir /tmp/out/ --no-params

    # Control param search candidates
    python tools/pipeline.py \\
        --chunks baseline.json --out-dir /tmp/out/ --param-n 8

    # Save result to case library
    python tools/pipeline.py \\
        --text "怖い..." --out-dir /tmp/out/ \\
        --situation "主人公が恐怖で動けないシーン" \\
        --scene-mode drama \\
        --notes "boundary_search+param_search自動選択"

Stages:
  1. annotate  — split text + LLM annotation → initial chunks.json
  2. boundary  — merge-adjacent variants → pick best MOS boundary
  3. params    — random perturbation candidates → pick best MOS params
  4. final     — write final.wav + final_chunks.json
  5. library   — save to case_library (if --situation given)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

_tools_dir = Path(__file__).parent
_python = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hdr(msg: str) -> None:
    sys.stderr.write(f"\n{'='*60}\n[pipeline] {msg}\n{'='*60}\n")


def _pick_best(items: list[dict]) -> int | None:
    """Return index of item with best MOS (fallback: expr_score). Skip failed items."""
    ok = [(k, v) for k, v in enumerate(items) if v.get("ok")]
    if not ok:
        return None
    # Prefer MOS
    with_mos = [(k, v["scores"]["mos"]) for k, v in ok if "mos" in v.get("scores", {})]
    if with_mos:
        return max(with_mos, key=lambda t: t[1])[0]
    # Fallback: expr_score
    with_expr = [(k, v["scores"].get("expr_score", 0)) for k, v in ok if v.get("scores")]
    if with_expr:
        return max(with_expr, key=lambda t: t[1])[0]
    return ok[0][0]


def _run(cmd: list[str | Path], step: str) -> bool:
    sys.stderr.write(f"[pipeline] running: {' '.join(str(c) for c in cmd)}\n")
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        sys.stderr.write(f"[pipeline] {step} failed (rc={proc.returncode})\n")
        return False
    return True


# ---------------------------------------------------------------------------
# Stage 1: Annotate
# ---------------------------------------------------------------------------

def stage_annotate(args: argparse.Namespace, out_dir: Path) -> Path | None:
    chunks_path = out_dir / "01_annotated.json"
    cmd = [_python, _tools_dir / "annotate_script.py",
           "--scene-mode", args.scene_mode,
           "--backend", args.backend]
    if args.model:
        cmd += ["--model", args.model]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.text:
        cmd += ["--text", args.text]
    elif args.file:
        cmd += ["--file", str(args.file)]

    proc = subprocess.run([str(c) for c in cmd], capture_output=False,
                          stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write("[pipeline] annotate failed\n")
        return None

    chunks_path.write_text(proc.stdout, encoding="utf-8")
    chunks = json.loads(proc.stdout)
    sys.stderr.write(f"[pipeline] annotated → {len(chunks)} chunks  ({chunks_path})\n")
    return chunks_path


# ---------------------------------------------------------------------------
# Stage 2: Boundary search
# ---------------------------------------------------------------------------

def stage_boundary(chunks_path: Path, out_dir: Path, narrator: str) -> Path | None:
    boundary_dir = out_dir / "02_boundary"
    ok = _run([_python, _tools_dir / "boundary_search.py",
               "--chunks", chunks_path,
               "--out-dir", boundary_dir,
               "--narrator", narrator], "boundary_search")
    if not ok:
        return None

    manifest = json.loads((boundary_dir / "manifest.json").read_text(encoding="utf-8"))
    variants = manifest["variants"]
    best_k = _pick_best(variants)
    if best_k is None:
        sys.stderr.write("[pipeline] boundary_search: no successful variant\n")
        return None

    best = variants[best_k]
    sc = best.get("scores", {})
    mos_s = f"MOS={sc['mos']:.3f}" if "mos" in sc else f"expr={sc.get('expr_score','?')}"
    from boundary_search import _config_label  # noqa: F401
    sys.stderr.write(
        f"[pipeline] best boundary: variant {best_k+1}  "
        f"{_config_label(best['config'])}  {mos_s}\n"
    )

    adopted_path = out_dir / "02_boundary_adopted.json"
    adopted_path.write_text(
        json.dumps(best["chunks"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return adopted_path


# ---------------------------------------------------------------------------
# Stage 3: Param search
# ---------------------------------------------------------------------------

def stage_params(chunks_path: Path, out_dir: Path, narrator: str, n: int) -> Path | None:
    param_dir = out_dir / "03_params"
    ok = _run([_python, _tools_dir / "param_search.py",
               "--chunks", chunks_path,
               "--out-dir", param_dir,
               "--narrator", narrator,
               "--n", str(n)], "param_search")
    if not ok:
        return None

    manifest = json.loads((param_dir / "manifest.json").read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    best_k = _pick_best(candidates)
    if best_k is None:
        sys.stderr.write("[pipeline] param_search: no successful candidate\n")
        return None

    best = candidates[best_k]
    sc = best.get("scores", {})
    mos_s = f"MOS={sc['mos']:.3f}" if "mos" in sc else f"expr={sc.get('expr_score','?')}"
    sys.stderr.write(
        f"[pipeline] best params: cand {best_k+1}  seed={best['seed']}  {mos_s}\n"
    )

    adopted_path = out_dir / "03_params_adopted.json"
    adopted_path.write_text(
        json.dumps(best["chunks"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy best merged WAV
    src_wav = Path(best["merged_wav"])
    if src_wav.exists():
        dst_wav = out_dir / "03_params_best.wav"
        shutil.copy2(src_wav, dst_wav)
        sys.stderr.write(f"[pipeline] best WAV → {dst_wav}\n")

    return adopted_path


# ---------------------------------------------------------------------------
# Stage 4: Final output
# ---------------------------------------------------------------------------

def stage_final(chunks_path: Path, out_dir: Path, src_wav: Path | None) -> Path:
    final_chunks = out_dir / "final_chunks.json"
    shutil.copy2(chunks_path, final_chunks)

    final_wav = out_dir / "final.wav"
    wav_src = src_wav or (out_dir / "03_params_best.wav")
    if wav_src and wav_src.exists():
        shutil.copy2(wav_src, final_wav)
    sys.stderr.write(f"[pipeline] final chunks → {final_chunks}\n")
    if final_wav.exists():
        y, sr = sf.read(str(final_wav))
        dur = len(y) / sr
        sys.stderr.write(f"[pipeline] final WAV   → {final_wav}  ({dur:.1f}s)\n")
    return final_chunks


# ---------------------------------------------------------------------------
# Stage 5: Case library
# ---------------------------------------------------------------------------

def stage_library(chunks_path: Path, args: argparse.Namespace,
                  wav_path: Path | None) -> None:
    sys.path.insert(0, str(_tools_dir))
    from case_library import add_case

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    mos: float | None = None
    if wav_path and wav_path.exists():
        try:
            from score_wav import score_expressiveness
            # Try to get MOS from param manifest
            param_manifest = args.out_dir / "03_params" / "manifest.json"
            if param_manifest.exists():
                m = json.loads(param_manifest.read_text(encoding="utf-8"))
                best_k = _pick_best(m["candidates"])
                if best_k is not None:
                    mos = m["candidates"][best_k].get("scores", {}).get("mos")
        except Exception:
            pass

    source = "pipeline auto"
    if args.scene_mode:
        source += f" scene={args.scene_mode}"

    case_id = add_case(
        chunks=chunks,
        situation=args.situation,
        narrator=args.narrator,
        scene_mode=getattr(args, "scene_mode", "") or "",
        notes=getattr(args, "notes", "") or "",
        mos=mos,
        source=source,
    )
    sys.stderr.write(f"[pipeline] saved to case library: {case_id}\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(out_dir: Path, stages_run: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"[pipeline] DONE  →  {out_dir}")
    print(f"  stages: {' → '.join(stages_run)}")
    final_wav = out_dir / "final.wav"
    final_chunks = out_dir / "final_chunks.json"
    if final_wav.exists():
        y, sr = sf.read(str(final_wav))
        print(f"  final.wav        ({len(y)/sr:.1f}s)  →  {final_wav}")
    if final_chunks.exists():
        chunks = json.loads(final_chunks.read_text(encoding="utf-8"))
        print(f"  final_chunks.json ({len(chunks)} chunks)  →  {final_chunks}")
    print(f"\nnext steps:")
    print(f"  afplay {final_wav}")
    print(f"  python tools/param_search.py --chunks {final_chunks} --out-dir <dir> --n 5")
    print(f"  python tools/boundary_search.py --chunks {final_chunks} --out-dir <dir>")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end TTS synthesis pipeline")
    parser.add_argument("--out-dir", type=Path, required=True)

    # Input
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Script text to annotate and synthesize")
    src.add_argument("--file", type=Path, help="Script text file")
    src.add_argument("--chunks", type=Path, help="Existing chunks.json (skip annotation)")

    # Annotation options
    parser.add_argument("--scene-mode", default="natural",
                        choices=["natural", "drama", "comedy"])
    parser.add_argument("--backend", default="ollama",
                        choices=["anthropic", "ollama"])
    parser.add_argument("--model", help="LLM model override")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use neutral annotations (no LLM call)")

    # Synthesis options
    parser.add_argument("--narrator", default="Koharu Rikka")
    parser.add_argument("--param-n", type=int, default=5,
                        help="Number of param_search candidates (default: 5)")

    # Stage control
    parser.add_argument("--no-boundary", action="store_true",
                        help="Skip boundary_search stage")
    parser.add_argument("--no-params", action="store_true",
                        help="Skip param_search stage")

    # Case library
    parser.add_argument("--situation",
                        help="Save final result to case library with this description")
    parser.add_argument("--notes", help="Correction notes for case library")

    args = parser.parse_args()

    if not args.text and not args.file and not args.chunks:
        parser.error("Provide --text, --file, or --chunks")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stages_run: list[str] = []
    current_chunks: Path

    # --- Stage 1: Annotate ---
    if args.chunks:
        current_chunks = args.chunks
        sys.stderr.write(f"[pipeline] using existing chunks: {current_chunks}\n")
    else:
        _hdr("Stage 1: Annotate")
        result = stage_annotate(args, args.out_dir)
        if result is None:
            sys.stderr.write("[pipeline] ABORT: annotation failed\n")
            return 1
        current_chunks = result
        stages_run.append("annotate")

    # --- Stage 2: Boundary search ---
    if not args.no_boundary:
        _hdr("Stage 2: Boundary search")
        result = stage_boundary(current_chunks, args.out_dir, args.narrator)
        if result is not None:
            current_chunks = result
            stages_run.append("boundary")
        else:
            sys.stderr.write("[pipeline] boundary stage failed — continuing with previous chunks\n")
    else:
        sys.stderr.write("[pipeline] skipping boundary search (--no-boundary)\n")

    # --- Stage 3: Param search ---
    best_wav: Path | None = None
    if not args.no_params:
        _hdr("Stage 3: Param search")
        result = stage_params(current_chunks, args.out_dir, args.narrator, args.param_n)
        if result is not None:
            current_chunks = result
            stages_run.append("params")
            best_wav = args.out_dir / "03_params_best.wav"
        else:
            sys.stderr.write("[pipeline] param stage failed — continuing with previous chunks\n")
    else:
        sys.stderr.write("[pipeline] skipping param search (--no-params)\n")

    # --- Stage 4: Final output ---
    _hdr("Stage 4: Finalise")
    stage_final(current_chunks, args.out_dir, best_wav)
    stages_run.append("final")

    # --- Stage 5: Case library ---
    if args.situation:
        _hdr("Stage 5: Case library")
        final_wav = args.out_dir / "final.wav"
        stage_library(current_chunks, args, final_wav if final_wav.exists() else None)
        stages_run.append("library")

    _print_summary(args.out_dir, stages_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
