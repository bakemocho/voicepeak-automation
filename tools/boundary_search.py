#!/usr/bin/env python3
"""Chunk boundary search for VOICEPEAK synthesis.

Generates candidate boundary configurations by merging adjacent chunks,
synthesizes each, scores with MOS + expressiveness, and presents a
comparison table so you can pick the best grouping.

Boundary configs tried:
  - Original (no merge)
  - Every adjacent pair merge  (N-1 configs)
  - Every adjacent triple merge (N-2 configs)
  - Full merge (all chunks as one)

Usage:
    # Generate all boundary variants from baseline chunks.json
    python tools/boundary_search.py --chunks baseline.json --out-dir bsearch/ --narrator "Koharu Rikka"

    # Play variant K
    python tools/boundary_search.py --out-dir bsearch/ --play 3

    # Adopt variant K as new baseline (writes adopted.json)
    python tools/boundary_search.py --out-dir bsearch/ --adopt 3

    # Show table from existing run
    python tools/boundary_search.py --out-dir bsearch/ --show
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
from score_wav import score_expressiveness  # noqa: E402
from case_library import add_case, _DEFAULT_LIBRARY as _CASE_LIBRARY_DEFAULT  # noqa: E402

_VENV_EVAL_PYTHON = _tools_dir.parent / ".venv-eval" / "bin" / "python"

TOP_DB = 40
CROSSFADE_MS = 30


# ---------------------------------------------------------------------------
# Boundary configuration generation
# ---------------------------------------------------------------------------

def _generate_configs(n: int) -> list[list[list[int]]]:
    """Return all boundary configs for n chunks.

    Each config is a list of groups; each group is a list of chunk indices.
    Example for n=4:
      [[0],[1],[2],[3]]          — original
      [[0,1],[2],[3]]            — merge 0+1
      [[0],[1,2],[3]]            — merge 1+2
      ...
      [[0,1,2],[3]]              — merge triple 0+1+2
      ...
      [[0,1,2,3]]                — merge all
    """
    configs: list[list[list[int]]] = []

    # Original
    configs.append([[i] for i in range(n)])

    # Adjacent pair merges
    for pivot in range(n - 1):
        config: list[list[int]] = []
        i = 0
        while i < n:
            if i == pivot:
                config.append([i, i + 1])
                i += 2
            else:
                config.append([i])
                i += 1
        configs.append(config)

    # Adjacent triple merges
    for pivot in range(n - 2):
        config = []
        i = 0
        while i < n:
            if i == pivot:
                config.append([i, i + 1, i + 2])
                i += 3
            else:
                config.append([i])
                i += 1
        configs.append(config)

    # Full merge (only when meaningful)
    if n > 3:
        configs.append([list(range(n))])

    return configs


def _config_label(config: list[list[int]]) -> str:
    """Human-readable label: [1][2+3][4] etc. (1-indexed)."""
    parts = []
    for group in config:
        parts.append("+".join(str(i + 1) for i in group))
    return "[" + "][".join(parts) + "]"


# ---------------------------------------------------------------------------
# Parameter merging for a group of original chunks
# ---------------------------------------------------------------------------

def _merge_group(chunks: list[dict], group: list[int]) -> dict:
    """Merge a group of chunk entries into one synthesizable chunk."""
    selected = [chunks[i] for i in group]
    text = "".join(c["text"] for c in selected)

    # Weighted average speed (by text length)
    lengths = [max(len(c["text"]), 1) for c in selected]
    total = sum(lengths)
    speed = sum(c.get("speed", 1.0) * l for c, l in zip(selected, lengths)) / total

    # Average pitch
    pitch = sum(c.get("pitch", 0.0) for c in selected) / len(selected)

    # Max emotion per key (preserve expressiveness)
    emotion: dict[str, float] = {}
    for c in selected:
        for k, v in c.get("emotion", {}).items():
            emotion[k] = max(emotion.get(k, 0.0), v)

    result: dict = {
        "text": text,
        "speed": round(speed, 3),
        "pitch": round(pitch, 3),
    }
    if emotion:
        result["emotion"] = {k: round(v, 3) for k, v in emotion.items()}

    # Inherit last chunk's gap
    last = selected[-1]
    if "gap_after_ms" in last:
        result["gap_after_ms"] = last["gap_after_ms"]

    return result


def _config_to_chunks(original: list[dict], config: list[list[int]]) -> list[dict]:
    return [_merge_group(original, group) for group in config]


# ---------------------------------------------------------------------------
# Synthesis helpers (mirrors param_search.py)
# ---------------------------------------------------------------------------

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


def _merge_wavs(chunk_wavs: list[Path], chunks: list[dict], sr: int) -> np.ndarray:
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
            out[: len(merged) - fn] = merged[: len(merged) - fn]
            out[len(merged) - fn : len(merged)] = merged[len(merged) - fn :] * (1 - r) + nxt[:fn] * r
            out[len(merged) :] = nxt[fn:]
            merged = out
        else:
            merged = np.concatenate([merged, nxt])
    return merged


def _score(wav_path: Path) -> dict:
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
                last_line = proc.stdout.strip().splitlines()[-1]
                scores["mos"] = json.loads(last_line)["mos"]
        except Exception as e:
            sys.stderr.write(f"  [score] mos failed: {e}\n")
    return scores


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _show_table(variants: list[dict]) -> None:
    has_mos = any("mos" in v.get("scores", {}) for v in variants)
    score_hdr = "  MOS    expr" if has_mos else "  expr"

    print(f"\n  {'#':>4}  {'chunks':>6}  {'boundary':<36}{score_hdr}")
    print("  " + "─" * (54 + (14 if has_mos else 7)))

    best_k = max(
        (k for k, v in enumerate(variants) if v.get("ok") and "mos" in v.get("scores", {})),
        key=lambda k: variants[k]["scores"]["mos"],
        default=None,
    )

    for k, v in enumerate(variants):
        sc = v.get("scores", {})
        dur = v.get("duration_s", 0)
        ok_s = "ok" if v.get("ok") else "FAIL"
        label = _config_label(v["config"])
        n_groups = len(v["config"])

        score_s = ""
        if has_mos:
            mos_s = f"{sc['mos']:.3f}" if "mos" in sc else "  —  "
            score_s = f"  {mos_s}"
        if "expr_score" in sc:
            score_s += f"  {sc['expr_score']:.3f}"

        marker = " ★" if k == best_k else ""
        print(
            f"  {k+1:>4}  {n_groups:>6}  {label:<36}"
            f"  ({dur:.1f}s {ok_s}){score_s}{marker}"
        )
    print()
    if best_k is not None:
        sc = variants[best_k]["scores"]
        print(f"  Best by MOS: variant {best_k+1}  (MOS={sc['mos']:.3f})")
        print(f"  Boundary: {_config_label(variants[best_k]['config'])}")
        print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    original: list[dict] = json.loads(args.chunks.read_text(encoding="utf-8"))
    n = len(original)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = _generate_configs(n)
    sys.stderr.write(f"[boundary] {n} chunks → {len(configs)} configs to try\n")

    variants: list[dict] = []
    sr_detected: int | None = None

    for k, config in enumerate(configs):
        merged_chunks = _config_to_chunks(original, config)
        label = _config_label(config)
        var_dir = args.out_dir / f"var_{k+1:02d}"
        var_dir.mkdir(exist_ok=True)

        sys.stderr.write(f"\n  [var {k+1:02d}] {label}\n")
        chunk_wavs: list[Path] = []
        all_ok = True
        for i, entry in enumerate(merged_chunks):
            wav = var_dir / f"{i+1:03d}.wav"
            ok = _trim_synth(entry, wav, args.narrator)
            if not ok:
                sys.stderr.write(f"    [FAIL] chunk {i+1}\n")
                all_ok = False
            else:
                if sr_detected is None:
                    _, sr_detected = librosa.load(str(wav), sr=None, mono=True)
                    sr_detected = int(sr_detected)
                chunk_wavs.append(wav)
            sys.stderr.write(f"    [{i+1:03d}] {'ok' if ok else 'FAIL'}  {entry['text'][:40]!r}\n")

        var: dict = {"config": config, "chunks": merged_chunks, "ok": False}
        if all_ok and len(chunk_wavs) == len(merged_chunks):
            sr = sr_detected or 44100
            audio = _merge_wavs(chunk_wavs, merged_chunks, sr)
            merged_path = var_dir / "merged.wav"
            sf.write(str(merged_path), audio, sr)
            var["ok"] = True
            var["merged_wav"] = str(merged_path)
            var["duration_s"] = round(len(audio) / sr, 2)
            print(f"  var {k+1:02d} {label} → {merged_path}  ({var['duration_s']}s)")

        variants.append(var)

    # Score successful variants
    ok_vars = [v for v in variants if v.get("ok")]
    if ok_vars:
        sys.stderr.write(f"\n  [score] scoring {len(ok_vars)} variants...\n")
        for v in ok_vars:
            wav = Path(v["merged_wav"])
            v["scores"] = _score(wav)
            sc = v["scores"]
            mos_s = f"MOS={sc['mos']:.3f}" if "mos" in sc else ""
            sys.stderr.write(
                f"  [score] {_config_label(v['config'])}  {mos_s}"
                f"  expr={sc.get('expr_score', '?')}\n"
            )

    manifest = {"original_chunks": str(args.chunks), "variants": variants}
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _show_table(variants)
    print(f"manifest → {manifest_path}")
    print("next: --adopt K")


def cmd_play(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    v = manifest["variants"][args.play - 1]
    wav = v.get("merged_wav")
    if not wav or not Path(wav).exists():
        print(f"variant {args.play}: no WAV found")
        return
    print(f"playing variant {args.play}: {_config_label(v['config'])}  {wav}")
    subprocess.run(["afplay", wav])


def cmd_adopt(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    v = manifest["variants"][args.adopt - 1]
    out_path = args.out_dir / "adopted.json"
    out_path.write_text(
        json.dumps(v["chunks"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adopted variant {args.adopt} → {out_path}")
    print(f"boundary: {_config_label(v['config'])}")
    print(f"use as next baseline: --chunks {out_path}")

    mos = v.get("scores", {}).get("mos")
    if mos is not None:
        print(f"MOS: {mos:.3f}")

    # Save to case library if --situation provided
    if getattr(args, "situation", None):
        case_id = add_case(
            chunks=v["chunks"],
            situation=args.situation,
            narrator=getattr(args, "narrator", "Koharu Rikka"),
            scene_mode=getattr(args, "scene_mode", "") or "",
            notes=getattr(args, "notes", "") or "",
            mos=mos,
            source=f"boundary_search {_config_label(v['config'])}",
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
    parser = argparse.ArgumentParser(description="Chunk boundary search for chunk synthesis")
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--chunks", type=Path, help="Baseline chunks JSON")
    parser.add_argument("--narrator", default="Koharu Rikka")

    parser.add_argument("--play", type=int, metavar="K", help="Play variant K")
    parser.add_argument("--adopt", type=int, metavar="K", help="Adopt variant K as new baseline")
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
        _show_table(manifest["variants"])
    elif args.chunks:
        cmd_generate(args)
    else:
        parser.error("Provide --chunks (generate) or --play/--adopt/--show (inspect)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
