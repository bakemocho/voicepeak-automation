from __future__ import annotations

import re

from voicepeak_automation.dictionary import convert_alphabet_to_katakana

_FORMULA_PATTERN = re.compile(r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\[[^\]]*\])")


def apply_formula_mode(text: str, mode: str) -> str:
    if mode == "keep":
        return text
    if mode == "strip":
        return _FORMULA_PATTERN.sub("", text)
    if mode == "placeholder":
        return _FORMULA_PATTERN.sub("数式", text)
    raise ValueError(f"unknown formula mode: {mode}")


def strip_markdown_decorations(text: str) -> str:
    value = re.sub(r"\*\*|\*|__|_", "", text)
    value = re.sub(r"(\d+)\.", r"\1。", value)
    return value


def split_text(text: str, max_length: int) -> list[str]:
    text = strip_markdown_decorations(text)

    def split_chunk(chunk: str) -> list[str]:
        sub_chunks: list[str] = []
        current = ""

        for part in re.split(r"(?<=。|．|！|？|\n)", chunk):
            if not part:
                continue
            if len(current) + len(part) <= max_length:
                current += part
            else:
                if current:
                    sub_chunks.append(current)
                if len(part) <= max_length:
                    current = part
                else:
                    # Hard split when there is no punctuation boundary.
                    if current:
                        sub_chunks.append(current)
                        current = ""
                    for i in range(0, len(part), max_length):
                        sub = part[i : i + max_length]
                        if len(sub) == max_length:
                            sub_chunks.append(sub)
                        else:
                            current = sub

        if current:
            sub_chunks.append(current)
        return sub_chunks

    initial_chunks = re.split(r"(?<=#)", text)
    evenly_split_chunks: list[str] = []
    for chunk in initial_chunks:
        if not chunk:
            continue
        if len(chunk) <= max_length:
            evenly_split_chunks.append(chunk)
        else:
            evenly_split_chunks.extend(split_chunk(chunk))

    combined_chunks: list[str] = []
    current_chunk = ""
    for chunk in evenly_split_chunks:
        if len(current_chunk) + len(chunk) <= max_length:
            current_chunk += chunk
        else:
            if current_chunk:
                combined_chunks.append(current_chunk)
            current_chunk = chunk

    if current_chunk:
        combined_chunks.append(current_chunk)

    # Trim/compact blanks.
    return [c.strip() for c in combined_chunks if c and c.strip()]


def prepare_chunks(
    text: str,
    dictionaries: dict[str, str],
    formula_mode: str,
    max_chunk_chars: int,
) -> list[str]:
    processed = apply_formula_mode(text, formula_mode)
    converted = convert_alphabet_to_katakana(processed, dictionaries)
    return split_text(converted, max_chunk_chars)
