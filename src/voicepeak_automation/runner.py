from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from voicepeak_automation.dictionary import load_dictionaries
from voicepeak_automation.task import Task, TaskItem
from voicepeak_automation.text import prepare_chunks


@dataclass(frozen=True)
class ChunkResult:
    item_id: str
    chunk_index: int
    text: str
    output_wav: Path


@dataclass(frozen=True)
class RunResult:
    task_project: str
    dry_run: bool
    chunk_results: list[ChunkResult]


class RunnerError(RuntimeError):
    pass


def _safe_item_id(item_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in item_id)
    return safe.strip("-") or "item"


def _resolve_item_options(task: Task, item: TaskItem) -> tuple[str, str, int]:
    speaker = item.speaker or task.settings.speaker
    emotion = item.emotion or task.settings.emotion
    speed = item.speed or task.settings.speed
    return speaker, emotion, speed


def _build_voicepeak_command(
    voicepeak_path: str,
    text: str,
    output_wav: Path,
    speaker: str,
    emotion: str,
    speed: int,
) -> list[str]:
    return [
        voicepeak_path,
        "-s",
        text,
        "-o",
        str(output_wav),
        "-n",
        speaker,
        "-e",
        emotion,
        "--speed",
        str(speed),
    ]


def _run_command(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RunnerError(f"command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        message = f"command failed ({exc.returncode}): {' '.join(args)}"
        if stderr:
            message = f"{message}\n{stderr}"
        raise RunnerError(message) from exc


def run_task(task: Task, dry_run: bool = False) -> RunResult:
    task.settings.output_dir.mkdir(parents=True, exist_ok=True)
    dictionaries = load_dictionaries(task.settings.dictionary_dir)

    results: list[ChunkResult] = []

    for item in task.items:
        speaker, emotion, speed = _resolve_item_options(task, item)
        chunks = prepare_chunks(
            text=item.text,
            dictionaries=dictionaries,
            formula_mode=task.settings.formula_mode,
            max_chunk_chars=task.settings.max_chunk_chars,
        )

        safe_id = _safe_item_id(item.item_id)

        for chunk_offset, chunk in enumerate(chunks, start=1):
            wav_path = task.settings.output_dir / f"{safe_id}_{chunk_offset:03d}.wav"
            results.append(
                ChunkResult(
                    item_id=item.item_id,
                    chunk_index=chunk_offset,
                    text=chunk,
                    output_wav=wav_path,
                )
            )

            if dry_run:
                continue

            command = _build_voicepeak_command(
                voicepeak_path=task.settings.voicepeak_path,
                text=chunk,
                output_wav=wav_path,
                speaker=speaker,
                emotion=emotion,
                speed=speed,
            )
            _run_command(command)

            if task.settings.play:
                _run_command(["afplay", str(wav_path)])

    return RunResult(task_project=task.project, dry_run=dry_run, chunk_results=results)
