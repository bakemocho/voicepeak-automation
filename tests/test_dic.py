from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicepeak_automation.dic import (
    DicEntry,
    DicError,
    add_entry,
    load_dic,
    remove_entry,
    save_dic,
    validate_entry,
)


def _entry(**kwargs) -> DicEntry:
    defaults = dict(sur="田中", pron="タナカ", pos="Japanese_Koyuumeishi_ippan", priority=5, accentType=1, lang="ja")
    defaults.update(kwargs)
    return DicEntry(**defaults)


# --- validate_entry ---


def test_validate_entry_valid():
    assert validate_entry(_entry()) == []


def test_validate_entry_empty_sur():
    errors = validate_entry(_entry(sur=""))
    assert any("sur" in e for e in errors)


def test_validate_entry_empty_pron():
    errors = validate_entry(_entry(pron=""))
    assert any("pron" in e for e in errors)


def test_validate_entry_unknown_pos():
    errors = validate_entry(_entry(pos="Unknown_pos"))
    assert any("pos" in e for e in errors)


def test_validate_entry_priority_out_of_range():
    assert validate_entry(_entry(priority=0)) != []
    assert validate_entry(_entry(priority=10)) != []
    assert validate_entry(_entry(priority=1)) == []
    assert validate_entry(_entry(priority=9)) == []


def test_validate_entry_negative_accent():
    errors = validate_entry(_entry(accentType=-1))
    assert any("accentType" in e for e in errors)


def test_validate_entry_zero_accent_ok():
    assert validate_entry(_entry(accentType=0)) == []


# --- load_dic / save_dic ---


def test_load_dic_missing_file(tmp_path: Path):
    assert load_dic(tmp_path / "nonexistent.json") == []


def test_load_dic_empty_array(tmp_path: Path):
    p = tmp_path / "dic.json"
    p.write_text("[]", encoding="utf-8")
    assert load_dic(p) == []


def test_load_dic_not_array(tmp_path: Path):
    p = tmp_path / "dic.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(DicError):
        load_dic(p)


def test_roundtrip(tmp_path: Path):
    p = tmp_path / "dic.json"
    original = [_entry(sur="東京", pron="トウキョウ", accentType=0), _entry(sur="大阪", pron="オオサカ", accentType=2)]
    save_dic(original, p)
    loaded = load_dic(p)
    assert loaded == original


def test_save_creates_backup(tmp_path: Path):
    p = tmp_path / "dic.json"
    p.write_text(json.dumps([_entry().to_dict()], ensure_ascii=False), encoding="utf-8")
    save_dic([_entry(sur="新規", pron="シンキ", accentType=1)], p)
    assert p.with_suffix(".json.bak").exists()


def test_save_atomic_tmp_removed(tmp_path: Path):
    p = tmp_path / "dic.json"
    save_dic([_entry()], p)
    assert not p.with_suffix(".json.tmp").exists()


# --- add_entry ---


def test_add_entry_new():
    entries = [_entry(sur="東京", pron="トウキョウ", accentType=0)]
    new = _entry(sur="大阪", pron="オオサカ", accentType=2)
    result, replaced = add_entry(entries, new)
    assert not replaced
    assert len(result) == 2
    assert result[-1].sur == "大阪"


def test_add_entry_replace():
    entries = [_entry(sur="田中", pron="タナカ", accentType=1)]
    updated = _entry(sur="田中", pron="タナカ", accentType=0)
    result, replaced = add_entry(entries, updated)
    assert replaced
    assert len(result) == 1
    assert result[0].accentType == 0


def test_add_entry_order_preserved():
    entries = [_entry(sur="A", pron="エー", accentType=1), _entry(sur="B", pron="ビー", accentType=1)]
    result, _ = add_entry(entries, _entry(sur="A", pron="エー", accentType=0))
    assert result[0].sur == "A"
    assert result[1].sur == "B"


# --- remove_entry ---


def test_remove_entry_found():
    entries = [_entry(sur="田中", pron="タナカ", accentType=1), _entry(sur="佐藤", pron="サトウ", accentType=2)]
    result, found = remove_entry(entries, "田中")
    assert found
    assert len(result) == 1
    assert result[0].sur == "佐藤"


def test_remove_entry_not_found():
    entries = [_entry(sur="田中", pron="タナカ", accentType=1)]
    result, found = remove_entry(entries, "鈴木")
    assert not found
    assert len(result) == 1


def test_remove_entry_empty():
    result, found = remove_entry([], "田中")
    assert not found
    assert result == []
