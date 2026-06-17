"""Japanese text normalizer for TTS preprocessing.

Converts digits, symbols, and abbreviations to spoken Japanese so that
VOICEPEAK reads them correctly without relying on its internal G2P.

Supported transforms:
  - Arabic numerals → Japanese reading (1500 → せんごひゃく)
  - Kanji numerals mixed with Arabic (2024年 → にせんにじゅうよねん)
  - Phone numbers → digit-by-digit reading (090-1234-5678 → ぜろきゅうぜろ…)
  - Fractions/decimals → X点Y reading (3.14 → さんてんいちよん)
  - Common abbreviations (km → キロメートル, % → パーセント)
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Integer reading
# --------------------------------------------------------------------------- #

_DIGIT_READ = {
    "0": "ぜろ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
    "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう",
}

_PLACE = ["", "じゅう", "ひゃく", "せん"]
_LARGE = ["", "まん", "おく", "ちょう"]


def _read_group(n: int) -> str:
    """Read a 1–4 digit group (0–9999) as Japanese, omitting leading いち for large places."""
    if n == 0:
        return ""
    if n < 10:
        return _DIGIT_READ[str(n)]
    parts = []
    digits = [int(d) for d in str(n)]
    length = len(digits)
    for i, d in enumerate(digits):
        place = length - i - 1
        if d == 0:
            continue
        read = "" if d == 1 and place > 0 else _DIGIT_READ[str(d)]
        parts.append(read + _PLACE[place])
    return "".join(parts)


def int_to_japanese(n: int) -> str:
    """Convert non-negative integer to Japanese reading string."""
    if n < 0:
        return "まいなす" + int_to_japanese(-n)
    if n == 0:
        return "ぜろ"
    groups = []
    i = 0
    while n > 0:
        g = n % 10000
        if g:
            groups.append(_read_group(g) + _LARGE[i])
        n //= 10000
        i += 1
    return "".join(reversed(groups))


# --------------------------------------------------------------------------- #
# Regex-based transforms
# --------------------------------------------------------------------------- #

# Phone numbers: 090-1234-5678, (03)1234-5678, 03-1234-5678
_PHONE_RE = re.compile(
    r"(?<!\d)(\(?\d{2,5}\)?[-－])(\d{2,4}[-－]\d{4})(?!\d)"
)

# Decimal numbers: 3.14, 12.5
_DECIMAL_RE = re.compile(r"(\d+)[\.。](\d+)")

# Plain integers embedded in text (not already consumed by other patterns)
_INT_RE = re.compile(r"\d+")

# Common unit/abbreviation substitutions (order matters: longer first)
_ABBREV = [
    (r"km/h", "キロメートルパーアワー"),
    (r"km", "キロメートル"),
    (r"m/s", "メートルパーセカンド"),
    (r"(?<!\d)m(?!\d)", "メートル"),
    (r"cm", "センチメートル"),
    (r"mm", "ミリメートル"),
    (r"kg", "キログラム"),
    (r"(?<!\d)g(?!\d)", "グラム"),
    (r"ml|mL", "ミリリットル"),
    (r"(?<!\d)L(?!\d)", "リットル"),
    (r"%", "パーセント"),
    (r"℃", "ど"),
    (r"°C", "ど"),
    (r"・", "、"),  # interpunct → pause
]
_ABBREV_RE = [(re.compile(pat), repl) for pat, repl in _ABBREV]


def _phone_replace(m: re.Match) -> str:
    raw = m.group(0).replace("(", "").replace(")", "").replace("-", "").replace("－", "")
    return "".join(_DIGIT_READ[d] for d in raw if d.isdigit())


def _decimal_replace(m: re.Match) -> str:
    int_part = int_to_japanese(int(m.group(1)))
    frac_part = "".join(_DIGIT_READ[d] for d in m.group(2))
    return int_part + "てん" + frac_part


def _int_replace(m: re.Match) -> str:
    return int_to_japanese(int(m.group(0)))


def normalize(text: str) -> str:
    """Apply all normalization transforms to text."""
    # 1. Abbreviations/units (before digit conversion)
    for pattern, repl in _ABBREV_RE:
        text = pattern.sub(repl, text)

    # 2. Phone numbers (digit-by-digit)
    text = _PHONE_RE.sub(_phone_replace, text)

    # 3. Decimals (X点Y)
    text = _DECIMAL_RE.sub(_decimal_replace, text)

    # 4. Remaining integers
    text = _INT_RE.sub(_int_replace, text)

    return text
