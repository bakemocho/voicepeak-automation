"""VOICEPEAK .vpp project file generator.

Generates a JSON .vpp file (with trailing \\x00) from plain Japanese text,
using MeCab for tokenization and the kana→phoneme table for G2P.

Accent assignment strategy:
  1. Look up the word in dic.json by surface form → use accentType.
  2. Fall back to accent_type=0 (平板型, all-high from mora 2).

Pause duration:
  Punctuation tokens (、。！？) get a pause phoneme with configurable
  duration multiplier (d field, 1.0 = default).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicepeak_automation.dic import DEFAULT_DIC_PATH, DicEntry, load_dic
from voicepeak_automation.kana import is_vowel_phoneme, kana_to_phonemes
from voicepeak_automation.mecab_adapter import MeCabToken, split_sentences, tokenize

# Accent encoding constants
_A_LOW = 8192   # 0x2000
_A_HIGH = 8193  # 0x2001
_A_PAUSE = 4096  # 0x1000

VPP_VERSION = "1.2.9"

PUNCT_CHARS = frozenset("、。！？…・")


@dataclass
class VppParams:
    narrator: str = "Koharu Rikka"
    language: str = "japanese"
    speed: float = 1.0
    pitch: float = 0.0
    pause_scale: float = 1.0
    volume: float = 1.0
    emotions: dict[str, float] = field(default_factory=dict)
    # Pause duration multiplier at 、 and 。 respectively
    comma_pause_d: float = 1.0
    period_pause_d: float = 1.5
    # Pause multiplier for ！？
    exclaim_pause_d: float = 1.0


def _accent_pattern(n_moras: int, accent_type: int) -> list[int]:
    """Return list of _A_LOW/_A_HIGH per mora given accent type."""
    result: list[int] = []
    for i in range(n_moras):
        if accent_type == 0:
            # 平板型: first mora LOW, rest HIGH
            result.append(_A_LOW if i == 0 else _A_HIGH)
        elif accent_type == 1:
            # 頭高型: first mora HIGH, rest LOW
            result.append(_A_HIGH if i == 0 else _A_LOW)
        else:
            # N型: first mora LOW, moras 2..N HIGH, rest LOW
            if i == 0:
                result.append(_A_LOW)
            elif i < accent_type:
                result.append(_A_HIGH)
            else:
                result.append(_A_LOW)
    return result


def _build_token(
    tok: MeCabToken,
    char_offset: int,
    byte_offset: int,
    dic_index: dict[str, DicEntry],
    params: VppParams,
) -> tuple[dict[str, Any], int, int]:
    """Build a single .vpp token dict.

    Returns (token_dict, new_char_offset, new_byte_offset).
    """
    surface = tok.surface
    sur_bytes = surface.encode("utf-8")
    r8 = [byte_offset, byte_offset + len(sur_bytes)]
    r32 = [char_offset, char_offset + len(surface)]

    if surface in PUNCT_CHARS:
        # Pause token
        if surface in ("！", "？"):
            d = params.exclaim_pause_d
        elif surface == "、":
            d = params.comma_pause_d
        else:  # 。 …
            d = params.period_pause_d

        syl = {
            "s": "",
            "ig": False,
            "a": _A_PAUSE,
            "i": 0.0,
            "u": False,
            "p": [{"s": "pau", "d": d, "n": False}],
        }
        token_dict = {
            "s": surface,
            "pos": tok.vpp_pos,
            "lang": 0,
            "pe": False,
            "syl": [syl],
            "r8": r8,
            "r32": r32,
        }
        return token_dict, r32[1], r8[1]

    # Normal token — use pronunciation (pron) for phoneme breakdown
    reading = tok.pron or tok.yomi

    # Accent lookup: dic.json first, then default 0型
    dic_entry = dic_index.get(surface)
    accent_type = dic_entry.accentType if dic_entry else 0

    mora_phonemes = kana_to_phonemes(reading)
    n_moras = len(mora_phonemes)

    # Handle ッ (geminate): it shares the accent of the following mora
    # Simple strategy: treat ッ as LOW pitch always
    accent_vals = _accent_pattern(n_moras, accent_type)

    syls: list[dict[str, Any]] = []
    for mora_idx, phoneme_seq in enumerate(mora_phonemes):
        a_val = accent_vals[mora_idx] if mora_idx < len(accent_vals) else _A_HIGH
        phonemes = [
            {"s": p, "d": 1.0, "n": is_vowel_phoneme(p)}
            for p in phoneme_seq
        ]
        # ッ (closure) has no mora count contribution — mark a as LOW
        if phoneme_seq == ("cl",):
            a_val = _A_LOW

        syls.append({
            "s": reading[mora_idx] if mora_idx < len(reading) else "",
            "ig": True,
            "a": a_val,
            "i": 0.0,
            "u": False,
            "p": phonemes,
        })

    token_dict = {
        "s": surface,
        "pos": tok.vpp_pos,
        "lang": 0,
        "pe": False,
        "syl": syls,
        "r8": r8,
        "r32": r32,
    }
    return token_dict, r32[1], r8[1]


def _build_sentence(
    sentence_text: str,
    dic_index: dict[str, DicEntry],
    params: VppParams,
) -> dict[str, Any]:
    tokens_raw = tokenize(sentence_text)
    token_dicts: list[dict[str, Any]] = []
    char_off = 0
    byte_off = 0
    for tok in tokens_raw:
        td, char_off, byte_off = _build_token(tok, char_off, byte_off, dic_index, params)
        token_dicts.append(td)
    return {"text": sentence_text, "tokens": token_dicts}


def generate_vpp(
    text: str,
    params: VppParams | None = None,
    dic_path: Path = DEFAULT_DIC_PATH,
) -> dict[str, Any]:
    """Generate a .vpp project dict from plain Japanese text."""
    if params is None:
        params = VppParams()

    dic_entries = load_dic(dic_path)
    dic_index = {e.sur: e for e in dic_entries}

    sentences = split_sentences(text)
    sentence_list = [_build_sentence(s, dic_index, params) for s in sentences]

    block: dict[str, Any] = {
        "narrator": {
            "key": params.narrator,
            "language": params.language,
            "narrator-version": -1,
        },
        "time-offset-mode": 2,
        "time-offset": 0.0,
        "params": {
            "speed": params.speed,
            "pitch": params.pitch,
            "pause": params.pause_scale,
            "volume": params.volume,
        },
        "emotions": params.emotions,
        "sentence-list": sentence_list,
    }

    return {
        "version": VPP_VERSION,
        "project": {
            "params": {
                "speed": params.speed,
                "pitch": params.pitch,
                "pause": params.pause_scale,
                "volume": params.volume,
            },
            "emotions": params.emotions,
            "blocks": [block],
        },
    }


def write_vpp(data: dict[str, Any], path: Path) -> None:
    """Write a .vpp project dict to file (single-line JSON + \\x00)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ": "))
    with path.open("wb") as fh:
        fh.write(json_str.encode("utf-8"))
        fh.write(b"\x00")
