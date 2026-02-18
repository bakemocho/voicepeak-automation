from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from voicepeak_automation.dictionary import load_dictionaries
from voicepeak_automation.task import Task, TaskItem
from voicepeak_automation.text import prepare_chunks

SPEED_MIN = 50
SPEED_MAX = 200
EMOTION_MIN = 0
EMOTION_MAX = 100
DEFAULT_FALLBACK_SPEAKER = "Koharu Rikka"
SYNTH_TIMEOUT_SEC = 30
LIST_TIMEOUT_SEC = 10
PLAY_TIMEOUT_SEC = 30


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
    warnings: list[str]
    errors: list[str]


class RunnerError(RuntimeError):
    pass


def _safe_item_id(item_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in item_id)
    return safe.strip("-") or "item"


def _non_noise_lines(text: str) -> list[str]:
    result: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "iconv_open is not supported" in line:
            continue
        if line.startswith("[debug]"):
            continue
        result.append(line)
    return result


def _build_voicepeak_command(
    voicepeak_path: str,
    text: str,
    output_wav: Path,
    speaker: str,
    emotion: str | None,
    speed: int,
) -> list[str]:
    command = [
        voicepeak_path,
        "-s",
        text,
        "-o",
        str(output_wav),
        "-n",
        speaker,
        "--speed",
        str(speed),
    ]
    if emotion:
        command.extend(["-e", emotion])
    return command


def _run_command(
    args: list[str],
    *,
    timeout_sec: int = SYNTH_TIMEOUT_SEC,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise RunnerError(f"command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        filtered = _non_noise_lines(stderr)
        message = f"command timed out after {timeout_sec}s: {' '.join(args)}"
        if filtered:
            message = f"{message}\n{'; '.join(filtered[-3:])}"
        raise RunnerError(message) from exc

    if check and completed.returncode != 0:
        filtered = _non_noise_lines(completed.stderr or "")
        message = f"command failed ({completed.returncode}): {' '.join(args)}"
        if filtered:
            message = f"{message}\n{'; '.join(filtered[-3:])}"
        raise RunnerError(message)

    return completed


def _list_narrators(voicepeak_path: str) -> list[str]:
    completed = _run_command(
        [voicepeak_path, "--list-narrator"],
        timeout_sec=LIST_TIMEOUT_SEC,
        check=True,
    )
    narrators = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return narrators


def _list_emotion_keys(voicepeak_path: str, narrator: str) -> list[str]:
    completed = _run_command(
        [voicepeak_path, "--list-emotion", narrator],
        timeout_sec=LIST_TIMEOUT_SEC,
        check=True,
    )
    keys = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return keys


def _pick_fallback_speaker(available: list[str]) -> str:
    if DEFAULT_FALLBACK_SPEAKER in available:
        return DEFAULT_FALLBACK_SPEAKER
    return available[0]


def _resolve_speaker(
    requested: str,
    available: list[str],
    warnings: list[str],
    context: str,
) -> str:
    if requested in available:
        return requested
    fallback = _pick_fallback_speaker(available)
    warnings.append(
        f"{context}: narrator '{requested}' is invalid; fallback to '{fallback}'"
    )
    return fallback


def _clamp_speed(speed: int, warnings: list[str], context: str) -> int:
    clamped = max(SPEED_MIN, min(SPEED_MAX, speed))
    if clamped != speed:
        warnings.append(
            f"{context}: speed {speed} out of range; clamped to {clamped}"
        )
    return clamped


def _parse_emotion_pairs(expr: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for part in expr.split(","):
        token = part.strip()
        if not token or "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        pairs.append((key, value))
    return pairs


def _sanitize_emotion(
    expr: str,
    allowed_keys: list[str],
    warnings: list[str],
    context: str,
) -> str | None:
    if not allowed_keys:
        warnings.append(f"{context}: no emotion keys available; omit -e")
        return None

    allowed = set(allowed_keys)
    sanitized: list[tuple[str, int]] = []

    for key, value in _parse_emotion_pairs(expr):
        if key not in allowed:
            warnings.append(f"{context}: unknown emotion key '{key}'; dropped")
            continue
        clamped = max(EMOTION_MIN, min(EMOTION_MAX, value))
        if clamped != value:
            warnings.append(
                f"{context}: emotion '{key}'={value} out of range; clamped to {clamped}"
            )
        sanitized.append((key, clamped))

    if not sanitized:
        if "narration" in allowed:
            return "narration=100"
        return f"{allowed_keys[0]}=100"

    return ",".join(f"{key}={value}" for key, value in sanitized)


def run_task(task: Task, dry_run: bool = False) -> RunResult:
    task.settings.output_dir.mkdir(parents=True, exist_ok=True)
    dictionaries = load_dictionaries(task.settings.dictionary_dir)

    results: list[ChunkResult] = []
    warnings: list[str] = []
    errors: list[str] = []

    available_narrators: list[str] = []
    if not dry_run:
        available_narrators = _list_narrators(task.settings.voicepeak_path)
        if not available_narrators:
            raise RunnerError("no narrator found from voicepeak --list-narrator")

    emotion_cache: dict[str, list[str]] = {}

    def emotion_keys_for(speaker: str) -> list[str]:
        if speaker in emotion_cache:
            return emotion_cache[speaker]
        if dry_run:
            # In dry-run mode, avoid external probing and trust raw expression.
            emotion_cache[speaker] = []
            return emotion_cache[speaker]
        try:
            keys = _list_emotion_keys(task.settings.voicepeak_path, speaker)
        except RunnerError as exc:
            warnings.append(f"speaker '{speaker}': failed to list emotions ({exc}); omit -e")
            keys = []
        emotion_cache[speaker] = keys
        return keys

    for item in task.items:
        requested_speaker = item.speaker or task.settings.speaker
        if dry_run:
            speaker = requested_speaker
        else:
            speaker = _resolve_speaker(
                requested_speaker,
                available_narrators,
                warnings,
                f"item {item.item_id}",
            )

        raw_emotion = item.emotion or task.settings.emotion
        if dry_run:
            emotion: str | None = raw_emotion
        else:
            emotion = _sanitize_emotion(
                raw_emotion,
                emotion_keys_for(speaker),
                warnings,
                f"item {item.item_id}",
            )

        raw_speed = item.speed or task.settings.speed
        speed = _clamp_speed(raw_speed, warnings, f"item {item.item_id}")

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
            try:
                _run_command(command, timeout_sec=SYNTH_TIMEOUT_SEC, check=True)
            except RunnerError as exc:
                errors.append(
                    f"item {item.item_id} chunk {chunk_offset}: synthesis failed ({exc})"
                )
                continue

            if task.settings.play:
                try:
                    _run_command(
                        ["afplay", str(wav_path)],
                        timeout_sec=PLAY_TIMEOUT_SEC,
                        check=True,
                    )
                except RunnerError as exc:
                    warnings.append(
                        f"item {item.item_id} chunk {chunk_offset}: playback failed ({exc})"
                    )

    return RunResult(
        task_project=task.project,
        dry_run=dry_run,
        chunk_results=results,
        warnings=warnings,
        errors=errors,
    )
