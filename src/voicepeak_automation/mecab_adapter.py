"""MeCab subprocess adapter for Japanese text tokenization.

Uses the system mecab binary (no Python binding required).
Output format: IPAdic fields
  surface \\t pos1,pos2,pos3,pos4,conj_type,conj_form,base,yomi,pron
"""
from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MECAB_BIN = "/opt/homebrew/bin/mecab"

# .vpp pos value mapping from MeCab IPAdic pos1
_POS_MAP: dict[str, int] = {
    "名詞":   0x1000,  # 4096 - noun
    "代名詞": 0x1001,  # 4097 - pronoun
    "動詞":   0x1005,  # 4101 - verb
    "助動詞": 0x1007,  # 4103 - auxiliary verb
    "助詞":   0x1008,  # 4104 - particle
    "形容詞": 0x1000,  # noun-like
    "副詞":   0x100E,  # 4110 - adverb
    "接続詞": 0x1008,  # treat like particle
    "感動詞": 0x1000,
    "記号":   0x100B,  # 4107 - symbol/punctuation
    "接頭詞": 0x1000,
    "接尾辞": 0x1000,
    "フィラー": 0x1000,
    "その他": 0x1000,
}

PUNCT_POS = 0x100B  # 4107


@dataclass
class MeCabToken:
    surface: str
    pos1: str
    pos2: str
    yomi: str   # katakana reading
    pron: str   # pronunciation (may differ from yomi for は→ワ)
    vpp_pos: int


def _to_katakana(s: str) -> str:
    """Convert hiragana to katakana."""
    return "".join(
        chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s
    )


def _is_punct(surface: str) -> bool:
    return surface in {"、", "。", "！", "？", "…", "・", "「", "」", "『", "』",
                       "（", "）", "【", "【", "】", "\n"}


def _number_to_reading(surface: str) -> str:
    """Very basic digit-string reading (handles simple cases only)."""
    digit_map = {"0": "ゼロ", "1": "イチ", "2": "ニ", "3": "サン", "4": "ヨン",
                 "5": "ゴ", "6": "ロク", "7": "ナナ", "8": "ハチ", "9": "キュウ"}
    return "".join(digit_map.get(c, c) for c in surface)


def tokenize(text: str, mecab_bin: str = MECAB_BIN) -> list[MeCabToken]:
    """Run MeCab on text and return parsed tokens."""
    try:
        proc = subprocess.run(
            [mecab_bin],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        output = proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"MeCab failed: {exc}") from exc

    tokens: list[MeCabToken] = []
    for line in output.splitlines():
        if line in ("EOS", ""):
            continue
        if "\t" not in line:
            continue
        surface, fields_str = line.split("\t", 1)
        fields = fields_str.split(",")
        # IPAdic: pos1,pos2,pos3,pos4,conj_type,conj_form,base,yomi,pron
        pos1 = fields[0] if len(fields) > 0 else ""
        pos2 = fields[1] if len(fields) > 1 else ""
        yomi_raw = fields[7] if len(fields) > 7 else ""
        pron_raw = fields[8] if len(fields) > 8 else ""

        # Fallback reading for numbers and unknown entries
        if not yomi_raw or yomi_raw == "*":
            if re.fullmatch(r"[0-9]+", surface):
                yomi_raw = _number_to_reading(surface)
            else:
                # Romanize or use surface as-is (pass to VOICEPEAK plain)
                yomi_raw = _to_katakana(surface)

        yomi = _to_katakana(yomi_raw)
        pron = _to_katakana(pron_raw) if pron_raw and pron_raw != "*" else yomi

        vpp_pos = PUNCT_POS if _is_punct(surface) else _POS_MAP.get(pos1, 0x1000)

        tokens.append(MeCabToken(
            surface=surface,
            pos1=pos1,
            pos2=pos2,
            yomi=yomi,
            pron=pron,
            vpp_pos=vpp_pos,
        ))

    return tokens


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at 。！？ boundaries."""
    parts = re.split(r"(?<=[。！？])", text)
    return [p for p in parts if p.strip()]
