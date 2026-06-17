#!/usr/bin/env python3
"""WAV quality scorer: UTMOS (naturalness MOS) + SenseVoice (emotion/ASR) + audeering VAD.

Requires .venv-eval (Python 3.12):
    source .venv-eval/bin/activate
    pip install git+https://github.com/sarulab-speech/UTMOSv2.git funasr transformers librosa huggingface_hub

Usage:
    python tools/score_wav.py /path/to/audio.wav
    python tools/score_wav.py /path/to/audio.wav --json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Patch cuda autocast before any torch import that might touch it
import torch
torch.cuda.amp.autocast = contextlib.nullcontext  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# UTMOS (naturalness MOS, 1–5)
# --------------------------------------------------------------------------- #

_utmos_model = None


def _load_utmos():
    global _utmos_model
    if _utmos_model is None:
        import utmosv2
        _utmos_model = utmosv2.create_model(pretrained=True, device="cpu")
    return _utmos_model


def score_mos(wav_path: str | Path) -> float:
    """Return UTMOS MOS prediction (1.0–5.0). Higher = more natural."""
    model = _load_utmos()
    return float(model.predict(input_path=str(wav_path), device="cpu", verbose=False))


# --------------------------------------------------------------------------- #
# SenseVoice (emotion label + ASR transcript)
# --------------------------------------------------------------------------- #

_sense_model = None
_EMOTION_RE = re.compile(r"<\|(EMO_UNKNOWN|HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>")
_TEXT_TAG_RE = re.compile(r"<\|[^|]+\|>")


def _load_sense():
    global _sense_model
    if _sense_model is None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("FunAudioLLM/SenseVoiceSmall")
        _sense_model = AutoModel(model=model_dir, device="cpu", disable_update=True)
    return _sense_model


def score_emotion(wav_path: str | Path) -> dict[str, str]:
    """Return {'emotion': label, 'transcript': text} from SenseVoice."""
    model = _load_sense()
    results = model.generate(input=str(wav_path), language="ja", use_itn=False)
    if not results:
        return {"emotion": "EMO_UNKNOWN", "transcript": ""}
    raw_text = results[0].get("text", "")
    m = _EMOTION_RE.search(raw_text)
    emotion = m.group(1) if m else "EMO_UNKNOWN"
    transcript = _TEXT_TAG_RE.sub("", raw_text).strip()
    return {"emotion": emotion, "transcript": transcript}


# --------------------------------------------------------------------------- #
# audeering VAD (arousal / dominance / valence, each 0–1)
# --------------------------------------------------------------------------- #

_vad_model = None
_vad_proc = None
_VAD_LABELS = ["arousal", "dominance", "valence"]


def _load_vad():
    global _vad_model, _vad_proc
    if _vad_model is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        _vad_proc = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _vad_model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        _vad_model.eval()
    return _vad_proc, _vad_model


def score_vad(wav_path: str | Path) -> dict[str, float]:
    """Return arousal/dominance/valence scores (0–1). arousal ≈ excitement level."""
    import librosa
    proc, model = _load_vad()
    wav, sr = librosa.load(str(wav_path), sr=16000)
    inputs = proc(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    scores = torch.sigmoid(out.logits).squeeze().tolist()
    return dict(zip(_VAD_LABELS, scores))


# --------------------------------------------------------------------------- #
# Combined scorer
# --------------------------------------------------------------------------- #

def score_wav(wav_path: str | Path) -> dict[str, Any]:
    """
    Run all three scorers on wav_path and return combined result.

    Returns:
        {
            "mos": float,          # 1.0–5.0 (UTMOS naturalness)
            "emotion": str,        # NEUTRAL/HAPPY/SAD/ANGRY/EMO_UNKNOWN (SenseVoice)
            "transcript": str,     # ASR text (SenseVoice)
            "arousal": float,      # 0–1 (audeering, excitement proxy)
            "dominance": float,
            "valence": float,
        }
    """
    mos = score_mos(wav_path)
    emo = score_emotion(wav_path)
    vad = score_vad(wav_path)
    return {
        "mos": round(mos, 3),
        "emotion": emo["emotion"],
        "transcript": emo["transcript"],
        **{k: round(v, 3) for k, v in vad.items()},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Score a WAV file for TTS quality and emotion")
    parser.add_argument("wav", help="path to WAV file")
    parser.add_argument("--json", action="store_true", help="output JSON")
    parser.add_argument("--mos-only", action="store_true", help="only run UTMOS")
    args = parser.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        print(f"[error] not found: {wav}")
        return 1

    if args.mos_only:
        result = {"mos": score_mos(wav)}
    else:
        result = score_wav(wav)

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"MOS (naturalness): {result['mos']:.3f}  [1–5, higher=better]")
        if "emotion" in result:
            print(f"Emotion:           {result['emotion']}")
            print(f"Transcript:        {result.get('transcript', '')}")
            print(f"Arousal:           {result['arousal']:.3f}  [0–1, higher=more excited]")
            print(f"Valence:           {result['valence']:.3f}  [0–1, higher=more positive]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
