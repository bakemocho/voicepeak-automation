from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VOICEPEAK_PATH = "/Applications/voicepeak.app/Contents/MacOS/voicepeak"
DEFAULT_SPEAKER = "Koharu Rikka"
DEFAULT_EMOTION = "hightension=50,livid=20,lamenting=0,despising=0,narration=100"
DEFAULT_SPEED = 200
DEFAULT_MAX_CHUNK_CHARS = 140
DEFAULT_FORMULA_MODE = "strip"
ALLOWED_FORMULA_MODES = {"strip", "keep", "placeholder"}


@dataclass(frozen=True)
class TaskSettings:
    voicepeak_path: str
    output_dir: Path
    speaker: str
    emotion: str
    speed: int
    play: bool
    dictionary_dir: Path | None
    max_chunk_chars: int
    formula_mode: str


@dataclass(frozen=True)
class TaskItem:
    item_id: str
    text: str
    speaker: str | None
    emotion: str | None
    speed: int | None


@dataclass(frozen=True)
class Task:
    project: str
    source_path: Path
    settings: TaskSettings
    items: list[TaskItem]


class TaskValidationError(ValueError):
    pass


def load_task_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"task file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TaskValidationError("task file must be a JSON object")
    return data


def _as_non_empty_str(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TaskValidationError(f"{name} must be a non-empty string")
    return text


def _as_positive_int(value: Any, name: str) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskValidationError(f"{name} must be an integer") from exc
    if num <= 0:
        raise TaskValidationError(f"{name} must be > 0")
    return num


def parse_task(path: Path) -> Task:
    raw = load_task_json(path)

    project = _as_non_empty_str(raw.get("project"), "project")

    settings_raw = raw.get("settings") or {}
    if not isinstance(settings_raw, dict):
        raise TaskValidationError("settings must be an object")

    output_dir_raw = settings_raw.get("output_dir") or "output"
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()

    dictionary_dir: Path | None = None
    dictionary_dir_raw = settings_raw.get("dictionary_dir")
    if dictionary_dir_raw:
        dictionary_dir = Path(dictionary_dir_raw)
        if not dictionary_dir.is_absolute():
            dictionary_dir = (path.parent / dictionary_dir).resolve()

    formula_mode = str(settings_raw.get("formula_mode") or DEFAULT_FORMULA_MODE).strip().lower()
    if formula_mode not in ALLOWED_FORMULA_MODES:
        raise TaskValidationError(
            f"formula_mode must be one of {sorted(ALLOWED_FORMULA_MODES)}"
        )

    settings = TaskSettings(
        voicepeak_path=str(settings_raw.get("voicepeak_path") or DEFAULT_VOICEPEAK_PATH),
        output_dir=output_dir,
        speaker=_as_non_empty_str(settings_raw.get("speaker") or DEFAULT_SPEAKER, "settings.speaker"),
        emotion=_as_non_empty_str(settings_raw.get("emotion") or DEFAULT_EMOTION, "settings.emotion"),
        speed=_as_positive_int(settings_raw.get("speed") or DEFAULT_SPEED, "settings.speed"),
        play=bool(settings_raw.get("play", False)),
        dictionary_dir=dictionary_dir,
        max_chunk_chars=_as_positive_int(
            settings_raw.get("max_chunk_chars") or DEFAULT_MAX_CHUNK_CHARS,
            "settings.max_chunk_chars",
        ),
        formula_mode=formula_mode,
    )

    items_raw = raw.get("items")
    if not isinstance(items_raw, list) or len(items_raw) == 0:
        raise TaskValidationError("items must be a non-empty list")

    items: list[TaskItem] = []
    for idx, value in enumerate(items_raw):
        if not isinstance(value, dict):
            raise TaskValidationError(f"items[{idx}] must be an object")
        item_id = _as_non_empty_str(value.get("id") or f"item-{idx + 1:03}", f"items[{idx}].id")
        text = _as_non_empty_str(value.get("text"), f"items[{idx}].text")

        speaker = value.get("speaker")
        emotion = value.get("emotion")
        speed = value.get("speed")

        parsed_speaker = str(speaker).strip() if isinstance(speaker, str) and speaker.strip() else None
        parsed_emotion = str(emotion).strip() if isinstance(emotion, str) and emotion.strip() else None
        parsed_speed = _as_positive_int(speed, f"items[{idx}].speed") if speed is not None else None

        items.append(
            TaskItem(
                item_id=item_id,
                text=text,
                speaker=parsed_speaker,
                emotion=parsed_emotion,
                speed=parsed_speed,
            )
        )

    return Task(
        project=project,
        source_path=path.resolve(),
        settings=settings,
        items=items,
    )
