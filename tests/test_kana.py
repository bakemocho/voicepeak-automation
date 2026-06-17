from __future__ import annotations

from voicepeak_automation.kana import kana_to_phonemes


def test_basic_vowels():
    assert kana_to_phonemes("アイウエオ") == [("a",), ("i",), ("u",), ("e",), ("o",)]


def test_basic_consonants():
    assert kana_to_phonemes("カキクケコ") == [("k","a"), ("k","i"), ("k","u"), ("k","e"), ("k","o")]


def test_special_N():
    assert kana_to_phonemes("ン") == [("N",)]


def test_geminate():
    assert kana_to_phonemes("ッ") == [("cl",)]


def test_compound_kana():
    assert kana_to_phonemes("シュ") == [("sh", "u")]
    assert kana_to_phonemes("リョ") == [("ry", "o")]
    assert kana_to_phonemes("ヒョ") == [("hy", "o")]


def test_long_vowel():
    result = kana_to_phonemes("コーヒー")
    assert result == [("k","o"), ("o",), ("h","i"), ("i",)]


def test_real_word_tanaka():
    result = kana_to_phonemes("タナカ")
    assert result == [("t","a"), ("n","a"), ("k","a")]


def test_real_word_tokyo():
    result = kana_to_phonemes("トウキョウ")
    assert result == [("t","o"), ("u",), ("ky","o"), ("u",)]


def test_compound_then_basic():
    result = kana_to_phonemes("シュウ")
    assert result == [("sh","u"), ("u",)]
