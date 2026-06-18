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
# Expressiveness (acoustic features — no model needed)
# --------------------------------------------------------------------------- #

def score_expressiveness(wav_path: str | Path) -> dict[str, float]:
    """
    Measure acoustic expressiveness from raw audio features.
    Higher values = more dynamic / expressive delivery.

    Returns:
        f0_range_st    : F0 range in semitones (voiced frames only)
        f0_cv          : F0 coefficient of variation (std/mean, voiced only)
        f0_voiced_ratio: fraction of frames that are voiced
        f0_octave_jumps: count of consecutive-voiced-frame F0 jumps > 6 semitones
                         (high values indicate pyin octave errors or synthesis artifacts)
        energy_cv      : RMS energy coefficient of variation (per 20ms frame)
        speaking_rate  : estimated syllables/second (energy-peak based)
        expr_score     : composite 0–1 (normalised weighted sum)
    """
    import librosa
    import numpy as np

    wav, sr = librosa.load(str(wav_path), sr=22050)
    duration = len(wav) / sr

    # F0 via pyin (voiced frames only)
    f0, voiced_flag, _ = librosa.pyin(
        wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=2048,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    f0_voiced_ratio = float(np.sum(voiced_flag)) / len(voiced_flag) if len(voiced_flag) > 0 else 0.0
    if len(voiced_f0) > 2:
        f0_hz_mean = float(np.mean(voiced_f0))
        f0_hz_std  = float(np.std(voiced_f0))
        f0_cv      = f0_hz_std / f0_hz_mean if f0_hz_mean > 0 else 0.0
        # Convert range to semitones: 12 * log2(max/min)
        f0_range_st = float(12 * np.log2(np.max(voiced_f0) / np.min(voiced_f0))) if np.min(voiced_f0) > 0 else 0.0
        # Octave jump detection: consecutive voiced-frame F0 diff > 6 semitones
        # Catches pyin octave errors and synthesis pitch instability
        f0_diffs_st = np.abs(12 * np.log2(voiced_f0[1:] / voiced_f0[:-1] + 1e-9))
        f0_octave_jumps = int(np.sum(f0_diffs_st > 6.0))
    else:
        f0_cv, f0_range_st, f0_octave_jumps = 0.0, 0.0, 0

    # RMS energy per frame (~20ms)
    hop = 512
    rms = librosa.feature.rms(y=wav, hop_length=hop)[0]
    rms_mean = float(np.mean(rms))
    energy_cv = float(np.std(rms) / rms_mean) if rms_mean > 0 else 0.0

    # Speaking rate: count energy peaks above 30% of max RMS (syllable proxy)
    threshold = float(np.max(rms)) * 0.3
    peaks = np.where((rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]) & (rms[1:-1] > threshold))[0]
    # Merge peaks within 80ms
    min_gap = int(0.08 * sr / hop)
    merged = []
    for p in peaks:
        if not merged or p - merged[-1] >= min_gap:
            merged.append(p)
    speaking_rate = len(merged) / duration if duration > 0 else 0.0

    # Composite expressiveness score (0–1, rough normalisation)
    # f0_range_st: 0–24 semitones typical range → /24
    # f0_cv: 0–0.3 typical → /0.3
    # energy_cv: 0–1.5 typical → /1.5
    expr_score = min(1.0, (
        0.4 * min(1.0, f0_range_st / 24.0) +
        0.3 * min(1.0, f0_cv / 0.3) +
        0.3 * min(1.0, energy_cv / 1.5)
    ))

    return {
        "f0_range_st":     round(f0_range_st, 2),
        "f0_cv":           round(f0_cv, 3),
        "f0_voiced_ratio": round(f0_voiced_ratio, 3),
        "f0_octave_jumps": f0_octave_jumps,
        "energy_cv":       round(energy_cv, 3),
        "speaking_rate":   round(speaking_rate, 2),
        "expr_score":      round(expr_score, 3),
    }


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
    expr = score_expressiveness(wav_path)
    return {
        "mos": round(mos, 3),
        "emotion": emo["emotion"],
        "transcript": emo["transcript"],
        **{k: round(v, 3) for k, v in vad.items()},
        **expr,
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
        if "expr_score" in result:
            print(f"Expressiveness:    {result['expr_score']:.3f}  [0–1, composite]")
            print(f"  F0 range:        {result['f0_range_st']:.1f} st  CV={result['f0_cv']:.3f}")
            print(f"  Voiced ratio:    {result['f0_voiced_ratio']:.3f}")
            octave_warn = "  ← pyin artifact?" if result['f0_octave_jumps'] > 2 else ""
            print(f"  Octave jumps:    {result['f0_octave_jumps']}{octave_warn}")
            print(f"  Energy CV:       {result['energy_cv']:.3f}")
            print(f"  Speaking rate:   {result['speaking_rate']:.1f} syl/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
