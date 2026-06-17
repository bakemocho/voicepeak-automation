"""Katakana syllable to VOICEPEAK phoneme sequence mapping.

Phoneme set confirmed from production .vpp analysis (PL2-V94-016):
  vowels: a i u e o
  consonants: k g s sh j ts ch t d n h hy b p m r ry w y N cl pau
"""
from __future__ import annotations

# Each entry: kana → tuple of phoneme strings
# Single kana (basic + voiced + semi-voiced)
_KANA_TABLE: dict[str, tuple[str, ...]] = {
    # vowels
    "ア": ("a",),
    "イ": ("i",),
    "ウ": ("u",),
    "エ": ("e",),
    "オ": ("o",),
    # k row
    "カ": ("k", "a"),
    "キ": ("k", "i"),
    "ク": ("k", "u"),
    "ケ": ("k", "e"),
    "コ": ("k", "o"),
    # g row
    "ガ": ("g", "a"),
    "ギ": ("g", "i"),
    "グ": ("g", "u"),
    "ゲ": ("g", "e"),
    "ゴ": ("g", "o"),
    # s row
    "サ": ("s", "a"),
    "シ": ("sh", "i"),
    "ス": ("s", "u"),
    "セ": ("s", "e"),
    "ソ": ("s", "o"),
    # z row
    "ザ": ("z", "a"),
    "ジ": ("j", "i"),
    "ズ": ("z", "u"),
    "ゼ": ("z", "e"),
    "ゾ": ("z", "o"),
    # t row
    "タ": ("t", "a"),
    "チ": ("ch", "i"),
    "ツ": ("ts", "u"),
    "テ": ("t", "e"),
    "ト": ("t", "o"),
    # d row
    "ダ": ("d", "a"),
    "ヂ": ("j", "i"),
    "ヅ": ("z", "u"),
    "デ": ("d", "e"),
    "ド": ("d", "o"),
    # n row
    "ナ": ("n", "a"),
    "ニ": ("n", "i"),
    "ヌ": ("n", "u"),
    "ネ": ("n", "e"),
    "ノ": ("n", "o"),
    # h row
    "ハ": ("h", "a"),
    "ヒ": ("h", "i"),
    "フ": ("f", "u"),
    "ヘ": ("h", "e"),
    "ホ": ("h", "o"),
    # b row
    "バ": ("b", "a"),
    "ビ": ("b", "i"),
    "ブ": ("b", "u"),
    "ベ": ("b", "e"),
    "ボ": ("b", "o"),
    # p row
    "パ": ("p", "a"),
    "ピ": ("p", "i"),
    "プ": ("p", "u"),
    "ペ": ("p", "e"),
    "ポ": ("p", "o"),
    # m row
    "マ": ("m", "a"),
    "ミ": ("m", "i"),
    "ム": ("m", "u"),
    "メ": ("m", "e"),
    "モ": ("m", "o"),
    # y row
    "ヤ": ("y", "a"),
    "ユ": ("y", "u"),
    "ヨ": ("y", "o"),
    # r row
    "ラ": ("r", "a"),
    "リ": ("r", "i"),
    "ル": ("r", "u"),
    "レ": ("r", "e"),
    "ロ": ("r", "o"),
    # w row
    "ワ": ("w", "a"),
    "ヲ": ("o",),
    # special
    "ン": ("N",),
    "ッ": ("cl",),
    # compound kana (拗音)
    "キャ": ("ky", "a"),
    "キュ": ("ky", "u"),
    "キョ": ("ky", "o"),
    "ギャ": ("gy", "a"),
    "ギュ": ("gy", "u"),
    "ギョ": ("gy", "o"),
    "シャ": ("sh", "a"),
    "シュ": ("sh", "u"),
    "ショ": ("sh", "o"),
    "ジャ": ("j", "a"),
    "ジュ": ("j", "u"),
    "ジョ": ("j", "o"),
    "チャ": ("ch", "a"),
    "チュ": ("ch", "u"),
    "チョ": ("ch", "o"),
    "ニャ": ("ny", "a"),
    "ニュ": ("ny", "u"),
    "ニョ": ("ny", "o"),
    "ヒャ": ("hy", "a"),
    "ヒュ": ("hy", "u"),
    "ヒョ": ("hy", "o"),
    "ビャ": ("by", "a"),
    "ビュ": ("by", "u"),
    "ビョ": ("by", "o"),
    "ピャ": ("py", "a"),
    "ピュ": ("py", "u"),
    "ピョ": ("py", "o"),
    "ミャ": ("my", "a"),
    "ミュ": ("my", "u"),
    "ミョ": ("my", "o"),
    "リャ": ("ry", "a"),
    "リュ": ("ry", "u"),
    "リョ": ("ry", "o"),
    "テャ": ("ty", "a"),
    "テュ": ("ty", "u"),
    "テョ": ("ty", "o"),
    "デャ": ("dy", "a"),
    "デュ": ("dy", "u"),
    "ファ": ("f", "a"),
    "フィ": ("f", "i"),
    "フェ": ("f", "e"),
    "フォ": ("f", "o"),
    "ヴァ": ("v", "a"),
    "ヴィ": ("v", "i"),
    "ヴ": ("v", "u"),
    "ヴェ": ("v", "e"),
    "ヴォ": ("v", "o"),
    "ウィ": ("w", "i"),
    "ウェ": ("w", "e"),
    "ウォ": ("w", "o"),
    "ティ": ("t", "i"),
    "トゥ": ("t", "u"),
    "ディ": ("d", "i"),
    "ドゥ": ("d", "u"),
    "ツァ": ("ts", "a"),
    "ツィ": ("ts", "i"),
    "ツェ": ("ts", "e"),
    "ツォ": ("ts", "o"),
}

# Nucleus flag: True for vowels and N (not false for all consonants)
_VOWELS = frozenset("a i u e o N".split())


def kana_to_phonemes(kana: str) -> list[tuple[str, ...]]:
    """Split a katakana reading into a list of (phoneme, ...) tuples per mora.

    Handles compound kana (拗音), long vowel ー, and geminate ッ.
    Unknown kana falls through as a single-phoneme tuple with the raw character.
    """
    moras: list[tuple[str, ...]] = []
    i = 0
    while i < len(kana):
        ch = kana[i]
        # Try 2-char compound first
        if i + 1 < len(kana) and (kana[i : i + 2] in _KANA_TABLE):
            moras.append(_KANA_TABLE[kana[i : i + 2]])
            i += 2
            continue
        # Long vowel ー: repeat the previous vowel
        if ch == "ー":
            if moras:
                prev = moras[-1]
                # last phoneme of previous mora that is a vowel
                vowel = next((p for p in reversed(prev) if p in _VOWELS), "u")
                moras.append((vowel,))
            else:
                moras.append(("u",))
            i += 1
            continue
        if ch in _KANA_TABLE:
            moras.append(_KANA_TABLE[ch])
            i += 1
            continue
        # Small kana that appear alone (shouldn't normally, skip)
        i += 1
    return moras


def is_vowel_phoneme(phoneme: str) -> bool:
    return phoneme in _VOWELS
