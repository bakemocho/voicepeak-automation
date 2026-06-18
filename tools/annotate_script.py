#!/usr/bin/env python3
"""LLM-driven per-sentence script annotation for VOICEPEAK synthesis.

Annotates each sentence with emotion weights, speed/pitch adjustments,
and accent corrections so the vpp generator can produce expressive output.

Usage:
    python tools/annotate_script.py --text "文章一。文章二。"
    python tools/annotate_script.py --file script.txt
    echo "台詞一。台詞二。" | python tools/annotate_script.py
    python tools/annotate_script.py --dry-run   # dummy output, no API call
    python tools/annotate_script.py --scene-mode comedy --text "なんと！そんなバカな！"

Scene modes:
    natural  (default) Naturalness first. Emotion cap 0.4. Optimises for MOS.
    drama              Heightened expression. Emotion cap 0.6. Lamenting/hightension peaks OK.
    comedy             No cap (up to 1.0). Exaggeration for comic effect; mismatch is fine.

Output JSON (stdout):
    [
      {
        "text": "文章一。",
        "scene_mode": "natural",
        "emotion": {"happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 0.0},
        "speed": 1.0,
        "pitch": 0.0,
        "accent_corrections": [
          {"word": "東京", "reading": "とうきょう", "accent_type": 0}
        ]
      },
      ...
    ]

Requires: ANTHROPIC_API_KEY env var (or set in ~/.anthropic/config).
Model: claude-haiku-4-5-20251001 (fast, cheap for annotation tasks).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voicepeak_automation.text_normalizer import normalize as _normalize_text

# Emotion labels supported by VOICEPEAK Koharu Rikka.
# Each value is a float 0.0–1.0.
VOICEPEAK_EMOTIONS = ["happy", "sad", "angry", "calm"]

SCENE_MODES = ("natural", "drama", "comedy")

_SYSTEM_PROMPT_BASE = """\
あなたはVOICEPEAK向けの音声台本アノテーターです。
ユーザーから日本語の台本テキストを受け取り、チャンク分割とアノテーションをJSON配列で返します。

## 出力フォーマット

```json
[
  {{
    "text": "チャンクのテキスト（統合した場合は連結テキスト）",
    "source_sentences": [1, 2],
    "emotion": {{
      "happy": 0.0,
      "sad": 0.0,
      "angry": 0.0,
      "calm": 0.0
    }},
    "speed": 1.0,
    "pitch": 0.0,
    "accent_corrections": []
  }}
]
```

## チャンクの統合ルール（重要）

入力テキストはすでに文単位に分割されています。
隣接する文を **1つのチャンクにまとめてよい** 条件：

- 統合すべきケース：
  - 感情が連続していて合計文字数が30字以内（例:「怖くて。」+「足が動かない。」）
  - 途切れた発話の断片（「……」始まり + 直後の短文）
  - 3字以下の超短文（「うん。」「え。」等）が後続文と一体感がある

- 統合してはいけないケース：
  - 感情・速度が明確に変わる境界
  - 1文だけで20字以上ある
  - 間（ポーズ）を強調する独立文

`source_sentences`: 元の文番号（1始まり）のリスト。統合しない場合は単一要素 `[N]`。
`text`: 統合した場合は各文を連結したテキスト（入力テキストをそのまま使う、変更しない）。

## シーンモード: {mode_name}

{mode_rules}

## 共通ルール

- `speed`: 1.0 が標準。早口なら 1.2、ゆっくりなら 0.8 程度。
- `pitch`: 0.0 が標準。高めなら +0.1〜+0.3、低めなら -0.1〜-0.3（annotation単位）。
- `accent_corrections`: アクセント辞書で誤読されやすい固有名詞・複合語を指定。
  - `word`: 表層形（MeCabが分割する前の文字列）
  - `reading`: 読みのひらがな
  - `accent_type`: 0=平板、1=頭高、N=第N拍以降が低下
- JSON 配列のみ出力。説明文は不要。
"""

_MODE_RULES = {
    "natural": (
        "natural（自然）",
        """\
自然な読み上げを優先します。感情は抑えめに。

- `emotion` の各値は **0.0〜0.4**。**0.4 を超えてはならない**。
  高強度はVOICEPEAKのF0範囲を潰してMOSと表現力を両方下げる。
- テキストのニュアンスに合わせて emotion をわずかに動かす程度でよい。""",
    ),
    "drama": (
        "drama（ドラマ）",
        """\
感情的な緊張感を出します。山場や感情的シーンに使います。

- `emotion` の各値は **0.0〜0.6**。**0.6 を超えてはならない**。
- lamenting（悲嘆）や hightension（緊迫）のピークを適切に使う。
- speed は 0.8〜1.2 の範囲で演技的にコントロールしてよい。""",
    ),
    "comedy": (
        "comedy（コメディ）",
        """\
誇張・不自然さを意図的に使います。バラエティ・ギャグ・パロディシーンに使います。

- `emotion` の各値は **0.0〜1.0**（キャップなし）。
- テキストの感情と emotion が一致しなくても構わない（ズレが笑いになる）。
- speed は 0.6〜1.5 の広い範囲で使える。極端な値も可。
- pitch も大きく動かしてよい（±20〜30 まで）。
- 「棒読み感」「ロボット感」が求められる場合は calm=1.0 + speed=1.0 が有効。""",
    ),
}


def _build_system_prompt(scene_mode: str) -> str:
    mode_name, mode_rules = _MODE_RULES[scene_mode]
    return _SYSTEM_PROMPT_BASE.format(mode_name=mode_name, mode_rules=mode_rules)


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation or newlines."""
    sentences = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[。！？])", line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def _extract_json_array(text: str) -> list:
    """Extract first JSON array from text (handles ```json ... ``` wrappers)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array found in LLM response:\n{text[:300]}")
    return json.loads(m.group(0))


def _build_prompt(sentences: list[str]) -> str:
    user_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return (
        f"以下の台本（{len(sentences)}文）をチャンク分割＋アノテーションしてください。\n"
        f"必要なら隣接する文を1チャンクに統合し、source_sentencesに元の番号を示してください。\n\n"
        f"{user_text}"
    )


def annotate_with_anthropic(
    sentences: list[str],
    model: str = "claude-haiku-4-5-20251001",
    scene_mode: str = "natural",
) -> list[dict]:
    """Call Anthropic API. Requires ANTHROPIC_API_KEY with available credits."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("[error] anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("[error] ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_build_system_prompt(scene_mode),
        messages=[{"role": "user", "content": _build_prompt(sentences)}],
    )
    return _extract_json_array(message.content[0].text.strip())


def annotate_with_ollama(
    sentences: list[str],
    model: str = "qwen3.6:27b",
    base_url: str = "http://localhost:11434",
    scene_mode: str = "natural",
) -> list[dict]:
    """Call local Ollama API. No credentials required."""
    import urllib.request

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_system_prompt(scene_mode)},
            {"role": "user", "content": _build_prompt(sentences)},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    raw = result["message"]["content"].strip()
    return _extract_json_array(raw)


def annotate_with_llm(
    sentences: list[str],
    backend: str = "ollama",
    model: str | None = None,
    scene_mode: str = "natural",
) -> list[dict]:
    """Dispatch to selected LLM backend."""
    if backend == "anthropic":
        return annotate_with_anthropic(sentences, model=model or "claude-haiku-4-5-20251001", scene_mode=scene_mode)
    elif backend == "ollama":
        return annotate_with_ollama(sentences, model=model or "qwen3.6:27b", scene_mode=scene_mode)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'anthropic' or 'ollama'")


def annotate_dry_run(sentences: list[str], scene_mode: str = "natural") -> list[dict]:
    """Return neutral annotations without API call (for testing pipeline)."""
    return [
        {
            "text": s,
            "source_sentences": [i + 1],
            "scene_mode": scene_mode,
            "emotion": {"happy": 0.0, "sad": 0.0, "angry": 0.0, "calm": 0.0},
            "speed": 1.0,
            "pitch": 0.0,
            "accent_corrections": [],
        }
        for i, s in enumerate(sentences)
    ]


def _inject_scene_mode(result: list[dict], scene_mode: str) -> list[dict]:
    """Add scene_mode field to each annotation entry."""
    for entry in result:
        entry.setdefault("scene_mode", scene_mode)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM annotation for VOICEPEAK scripts")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Text to annotate")
    src.add_argument("--file", type=Path, help="Text file to annotate")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM; return neutral annotations")
    parser.add_argument("--backend", default="ollama", choices=["anthropic", "ollama"], help="LLM backend")
    parser.add_argument("--model", default=None, help="Model ID (default: qwen3.6:27b for ollama, claude-haiku-4-5-20251001 for anthropic)")
    parser.add_argument(
        "--scene-mode", default="natural", choices=SCENE_MODES,
        help="Scene mode: natural (cap 0.4), drama (cap 0.6), comedy (no cap). Default: natural",
    )
    args = parser.parse_args()

    if args.text:
        raw_text = args.text
    elif args.file:
        raw_text = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        raw_text = sys.stdin.read()
    else:
        parser.error("Provide --text, --file, or pipe text via stdin")
        return 1

    sentences = _split_sentences(raw_text)
    if not sentences:
        print("[]")
        return 0

    # Pre-normalize numbers/units so LLM sees readable Japanese
    sentences = [_normalize_text(s) for s in sentences]

    scene_mode = args.scene_mode
    if args.dry_run:
        result = annotate_dry_run(sentences, scene_mode=scene_mode)
    else:
        result = annotate_with_llm(sentences, backend=args.backend, model=args.model, scene_mode=scene_mode)
        result = _inject_scene_mode(result, scene_mode)

    # Validate coverage via source_sentences
    covered: set[int] = set()
    for entry in result:
        for n in entry.get("source_sentences", []):
            covered.add(n)
    missing = [i + 1 for i in range(len(sentences)) if (i + 1) not in covered]
    if covered and missing:
        sys.stderr.write(f"[warn] sentences not covered by any chunk: {missing}\n")
    elif not covered and len(result) != len(sentences):
        sys.stderr.write(
            f"[warn] LLM returned {len(result)} chunks for {len(sentences)} sentences "
            f"(no source_sentences field — old-format response)\n"
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
