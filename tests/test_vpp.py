from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from voicepeak_automation.vpp import VppParams, _accent_pattern, generate_vpp, write_vpp

_A_LOW = 8192
_A_HIGH = 8193


# --- accent pattern ---

def test_accent_flat():
    # 0型 (平板型): L H H H
    assert _accent_pattern(4, 0) == [_A_LOW, _A_HIGH, _A_HIGH, _A_HIGH]


def test_accent_atamadaka():
    # 1型 (頭高型): H L L L
    assert _accent_pattern(4, 1) == [_A_HIGH, _A_LOW, _A_LOW, _A_LOW]


def test_accent_nakadaka_2():
    # 2型: L H L L
    assert _accent_pattern(4, 2) == [_A_LOW, _A_HIGH, _A_LOW, _A_LOW]


def test_accent_nakadaka_3():
    # 3型: L H H L
    assert _accent_pattern(4, 3) == [_A_LOW, _A_HIGH, _A_HIGH, _A_LOW]


def test_accent_single_mora():
    assert _accent_pattern(1, 0) == [_A_LOW]
    assert _accent_pattern(1, 1) == [_A_HIGH]


# --- generate_vpp ---

def test_generate_vpp_basic(tmp_path: Path):
    data = generate_vpp("田中さんは東京に行きました。")
    assert data["version"] == "1.2.9"
    project = data["project"]
    assert "blocks" in project
    assert len(project["blocks"]) == 1
    block = project["blocks"][0]
    assert "sentence-list" in block
    assert len(block["sentence-list"]) >= 1


def test_generate_vpp_multi_sentence(tmp_path: Path):
    data = generate_vpp("テストです。これは二文目です。")
    block = data["project"]["blocks"][0]
    # Should have 2 sentences
    assert len(block["sentence-list"]) == 2


def test_generate_vpp_narrator(tmp_path: Path):
    params = VppParams(narrator="Frimomen")
    data = generate_vpp("テスト。", params=params)
    block = data["project"]["blocks"][0]
    assert block["narrator"]["key"] == "Frimomen"


def test_generate_vpp_pause_params(tmp_path: Path):
    params = VppParams(comma_pause_d=2.0, period_pause_d=3.0)
    data = generate_vpp("テスト、です。", params=params)
    block = data["project"]["blocks"][0]
    sent = block["sentence-list"][0]
    pause_tokens = [t for t in sent["tokens"] if t["s"] in ("、", "。")]
    # Should have pause tokens with configured d values
    assert len(pause_tokens) >= 1
    for tok in pause_tokens:
        d = tok["syl"][0]["p"][0]["d"]
        if tok["s"] == "、":
            assert d == pytest.approx(2.0)
        elif tok["s"] == "。":
            assert d == pytest.approx(3.0)


def test_write_vpp_null_terminator(tmp_path: Path):
    data = generate_vpp("テスト。")
    out = tmp_path / "test.vpp"
    write_vpp(data, out)
    raw = out.read_bytes()
    assert raw[-1:] == b"\x00"


def test_write_vpp_valid_json(tmp_path: Path):
    data = generate_vpp("テスト。")
    out = tmp_path / "test.vpp"
    write_vpp(data, out)
    raw = out.read_bytes().rstrip(b"\x00")
    parsed = json.loads(raw)
    assert parsed["version"] == "1.2.9"


def test_r8_r32_coverage(tmp_path: Path):
    text = "田中さん。"
    data = generate_vpp(text)
    block = data["project"]["blocks"][0]
    sent = block["sentence-list"][0]
    # r32 of last token should end at len(text)
    last_tok = sent["tokens"][-1]
    assert last_tok["r32"][1] == len(text)
    # r8 of last token should end at len(text.encode('utf-8'))
    assert last_tok["r8"][1] == len(text.encode("utf-8"))
