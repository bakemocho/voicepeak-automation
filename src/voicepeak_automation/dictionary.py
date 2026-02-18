from __future__ import annotations

import re
from pathlib import Path

try:
    import romkan  # type: ignore
except ImportError:  # pragma: no cover - optional fallback
    romkan = None


_SPLIT_PATTERN = re.compile(r"([{}()\[\]_'`\":.,\s]|(?=[^a-zA-Z])|(?<=[^a-zA-Z]))")


def load_dictionaries(dictionary_dir: Path | None) -> dict[str, str]:
    if dictionary_dir is None:
        return {}
    if not dictionary_dir.exists() or not dictionary_dir.is_dir():
        raise FileNotFoundError(f"dictionary directory not found: {dictionary_dir}")

    mapping: dict[str, str] = {}
    for dic_file in sorted(dictionary_dir.glob("*.dic")):
        with dic_file.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                mapping[parts[0].upper()] = parts[1]

    return mapping


def split_camel_case(word: str) -> list[str]:
    if not word:
        return []
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
    parts = [p for p in re.split(r"\s+", value) if p]
    return parts or [word]


def _latin_to_katakana(word: str) -> str:
    if not word:
        return ""
    if romkan is None:
        return word
    return str(romkan.to_katakana(word))


def convert_alphabet_to_katakana(text: str, dictionaries: dict[str, str]) -> str:
    if not text:
        return ""

    split_words = _SPLIT_PATTERN.split(text)
    katakana_words: list[str] = []

    for word in split_words:
        if word is None or word == "":
            continue

        # Keep separators, with optional symbol dictionary substitution.
        if re.fullmatch(r"[{}()\[\]_'`\":.,\s]", word):
            katakana_words.append(dictionaries.get(word, word))
            continue

        base_word = word.upper()
        mapped = dictionaries.get(base_word)
        if mapped:
            katakana_words.append(mapped)
            continue

        converted_parts = []
        for part in split_camel_case(word):
            part_key = part.upper()
            converted_parts.append(dictionaries.get(part_key, _latin_to_katakana(part)))

        katakana_words.append("".join(converted_parts))

    return "".join(katakana_words)
