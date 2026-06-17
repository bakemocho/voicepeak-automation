#!/usr/bin/env python3
"""Irodori-TTS batch runner — executed inside Irodori-TTS/.venv Python.

Called by oracle_synth.py via subprocess. Loads model once and synthesizes
all chunks. Takes task JSON from stdin, writes per-chunk WAVs, outputs result
JSON to stdout.

Task JSON (stdin):
    {
      "checkpoint": "Aratako/Irodori-TTS-500M-v3",
      "codec_repo": "Aratako/Semantic-DACVAE-Japanese-32dim",
      "ref_wav": null,              # or path string
      "caption": null,              # or style caption string
      "no_ref": true,
      "ref_normalize_db": -16.0,
      "num_steps": 40,
      "cfg_scale_text": 3.0,
      "cfg_scale_caption": 3.2,
      "cfg_scale_speaker": 5.0,
      "seed": 20260618,
      "device": "mps",
      "chunks": [
        {"text": "怖いです", "out_wav": "/tmp/oracle/001.wav"},
        {"text": "ね。",     "out_wav": "/tmp/oracle/002.wav"}
      ]
    }

Result JSON (stdout):
    [
      {"text": "怖いです", "out_wav": "/tmp/oracle/001.wav", "ok": true, "duration_sec": 0.84},
      ...
    ]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add Irodori-TTS repo to path (this script runs inside the Irodori .venv)
_irodori_root = Path(__file__).parent.parent.parent / "Irodori-TTS"
if not _irodori_root.exists():
    # Allow override via environment variable
    _irodori_root = Path(os.environ.get("IRODORI_ROOT", str(_irodori_root)))
sys.path.insert(0, str(_irodori_root))

from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    resolve_cfg_scales,
    save_wav,
)


def main() -> int:
    task = json.loads(sys.stdin.read())

    # Irodori prints diagnostic messages to stdout; redirect to stderr so our
    # JSON result can be cleanly parsed by the parent process.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr

    checkpoint_repo = task.get("checkpoint", "Aratako/Irodori-TTS-500M-v3")
    codec_repo = task.get("codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim")
    device = task.get("device", default_runtime_device())
    num_steps = int(task.get("num_steps", 40))
    seed = int(task.get("seed", 20260618))
    ref_wav = task.get("ref_wav")
    caption = task.get("caption")
    no_ref = bool(task.get("no_ref", ref_wav is None))
    ref_normalize_db = float(task.get("ref_normalize_db", -16.0))
    cfg_scale_text = float(task.get("cfg_scale_text", 3.0))
    cfg_scale_caption = float(task.get("cfg_scale_caption", 3.2))
    cfg_scale_speaker = float(task.get("cfg_scale_speaker", 5.0))
    chunks = task["chunks"]

    checkpoint_path = hf_hub_download(repo_id=checkpoint_repo, filename="model.safetensors")
    print(f"[runner] checkpoint: {checkpoint_path}", file=sys.stderr)

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=device,
            codec_repo=codec_repo,
            model_precision="fp32",
            codec_device=device,
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    print(f"[runner] model loaded", file=sys.stderr)

    use_speaker = bool(runtime.model_cfg.use_speaker_condition_resolved and not no_ref)
    _, cfg_scale_caption_resolved, cfg_scale_speaker_resolved, _ = resolve_cfg_scales(
        cfg_guidance_mode="standard",
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_scale=None,
        use_caption_condition=bool(
            runtime.model_cfg.use_caption_condition
            and caption is not None
            and str(caption).strip() != ""
        ),
        use_speaker_condition=use_speaker,
    )

    results = []
    for chunk in chunks:
        text = chunk["text"]
        out_wav = chunk["out_wav"]
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

        try:
            result = runtime.synthesize(
                SamplingRequest(
                    text=text,
                    caption=caption,
                    ref_wav=ref_wav,
                    ref_latent=None,
                    ref_embed=None,
                    no_ref=no_ref,
                    ref_normalize_db=ref_normalize_db,
                    ref_ensure_max=False,
                    num_candidates=1,
                    num_steps=num_steps,
                    duration_scale=1.0,
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_caption=cfg_scale_caption_resolved,
                    cfg_scale_speaker=cfg_scale_speaker_resolved,
                    seed=seed,
                )
            )
            save_wav(out_wav, result.audio, result.sample_rate)
            duration_sec = result.audio.shape[-1] / result.sample_rate
            print(f"[runner] ok  {Path(out_wav).name}  {duration_sec:.2f}s", file=sys.stderr)
            results.append({
                "text": text,
                "out_wav": out_wav,
                "ok": True,
                "duration_sec": round(duration_sec, 3),
                "sample_rate": result.sample_rate,
            })
        except Exception as exc:
            print(f"[runner] FAIL {text!r}: {exc}", file=sys.stderr)
            results.append({"text": text, "out_wav": out_wav, "ok": False, "error": str(exc)})

    sys.stdout = _real_stdout
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
