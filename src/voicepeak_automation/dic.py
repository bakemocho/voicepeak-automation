from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DIC_PATH = (
    Path.home() / "Library/Application Support/Dreamtonics/Voicepeak/settings/dic.json"
)

KNOWN_POS = frozenset(
    {
        "Japanese_Futsuu_meishi",
        "Japanese_Koyuumeishi_ippan",
        "Japanese_Koyuumeishi_chiiki",
        "Japanese_Koyuumeishi_jinmei",
        "Japanese_Koyuumeishi_sei",
    }
)

DEFAULT_POS = "Japanese_Koyuumeishi_ippan"
DEFAULT_PRIORITY = 5
DEFAULT_LANG = "ja"


@dataclass
class DicEntry:
    sur: str
    pron: str
    pos: str
    priority: int
    accentType: int
    lang: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sur": self.sur,
            "pron": self.pron,
            "pos": self.pos,
            "priority": self.priority,
            "accentType": self.accentType,
            "lang": self.lang,
        }


class DicError(ValueError):
    pass


def load_dic(path: Path = DEFAULT_DIC_PATH) -> list[DicEntry]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise DicError(f"dic.json must be a JSON array: {path}")
    entries: list[DicEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DicError(f"dic.json entry {i} must be an object")
        entries.append(
            DicEntry(
                sur=str(item.get("sur", "")),
                pron=str(item.get("pron", "")),
                pos=str(item.get("pos", DEFAULT_POS)),
                priority=int(item.get("priority", DEFAULT_PRIORITY)),
                accentType=int(item.get("accentType", 0)),
                lang=str(item.get("lang", DEFAULT_LANG)),
            )
        )
    return entries


def save_dic(entries: list[DicEntry], path: Path = DEFAULT_DIC_PATH) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(".json.bak"))
    tmp = path.with_suffix(".json.tmp")
    data = [e.to_dict() for e in entries]
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp.replace(path)


def validate_entry(entry: DicEntry) -> list[str]:
    errors: list[str] = []
    if not entry.sur.strip():
        errors.append("sur must not be empty")
    if not entry.pron.strip():
        errors.append("pron must not be empty")
    if entry.pos not in KNOWN_POS:
        errors.append(f"unknown pos: {entry.pos!r} (known: {sorted(KNOWN_POS)})")
    if not 1 <= entry.priority <= 9:
        errors.append(f"priority must be 1-9, got {entry.priority}")
    if entry.accentType < 0:
        errors.append(f"accentType must be >= 0, got {entry.accentType}")
    if entry.lang != "ja":
        errors.append(f"unexpected lang: {entry.lang!r}")
    return errors


def add_entry(
    entries: list[DicEntry], entry: DicEntry
) -> tuple[list[DicEntry], bool]:
    """Add or replace entry matching sur. Returns (new_list, was_replaced)."""
    result: list[DicEntry] = []
    replaced = False
    for existing in entries:
        if existing.sur == entry.sur:
            result.append(entry)
            replaced = True
        else:
            result.append(existing)
    if not replaced:
        result.append(entry)
    return result, replaced


def remove_entry(
    entries: list[DicEntry], sur: str
) -> tuple[list[DicEntry], bool]:
    """Remove all entries matching sur. Returns (new_list, was_found)."""
    result = [e for e in entries if e.sur != sur]
    return result, len(result) < len(entries)
