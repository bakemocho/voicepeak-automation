"""VOICEPEAK call logger — records every synthesize_one attempt.

Two log files (JSONL, one record per line):
  logs/vp_calls.jsonl   — all calls (for pattern analysis)
  logs/vp_errors.jsonl  — failures only (for investigation)

Each record:
  {
    "ts":           "2026-06-18T11:30:00.123456",
    "call_n":       32,
    "ok":           false,
    "rc":           1,           # -11 = SIGSEGV, -6 = SIGABRT, 1 = generic error
    "duration_ms":  1423,
    "text":         "怖くて。",
    "narrator":     "Koharu Rikka",
    "speed":        89,
    "pitch":        -2,
    "emotion":      "lamenting=51",
    "out_path":     "/tmp/...",
    "stdout":       "",
    "stderr":       "...",
    "crash_report": "/Users/.../DiagnosticReports/voicepeak-....ips"  # if found
  }
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_call_counter = 0

_LOG_DIR = Path(__file__).parent.parent / "logs"
_CALLS_LOG = _LOG_DIR / "vp_calls.jsonl"
_ERRORS_LOG = _LOG_DIR / "vp_errors.jsonl"

_CRASH_DIR = Path.home() / "Library" / "Logs" / "DiagnosticReports"
_CRASH_WINDOW_S = 8  # seconds after call end to look for a new crash report


def _find_crash_report(call_end_epoch: float) -> str | None:
    """Return path to the most recent voicepeak crash report created around call_end_epoch."""
    if not _CRASH_DIR.exists():
        return None
    # Give macOS up to _CRASH_WINDOW_S to write the report
    deadline = time.monotonic() + _CRASH_WINDOW_S
    while time.monotonic() < deadline:
        candidates = sorted(
            _CRASH_DIR.glob("voicepeak-*.ips"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in candidates:
            mtime = p.stat().st_mtime
            # Report must have been created within a 30-second window ending now
            if call_end_epoch - 5 <= mtime <= call_end_epoch + _CRASH_WINDOW_S:
                return str(p)
        time.sleep(0.5)
    return None


def _extract_crash_summary(report_path: str) -> dict:
    """Parse key fields from an .ips crash report."""
    try:
        lines = Path(report_path).read_text(encoding="utf-8", errors="replace").splitlines()
        body = json.loads("\n".join(lines[1:]))
        exc = body.get("exception", {})
        summary: dict = {
            "exception_type": exc.get("type", ""),
            "exception_signal": exc.get("signal", ""),
            "exception_codes": exc.get("codes", ""),
        }
        for t in body.get("threads", []):
            if t.get("triggered"):
                frames = []
                for f in t.get("frames", [])[:10]:
                    sym = f.get("symbol", "")
                    loc = f.get("symbolLocation", f.get("imageOffset", ""))
                    frames.append(f"{sym or '???'} +{loc}")
                summary["crashed_frames"] = frames
                break
        return summary
    except Exception:
        return {}


def record(
    *,
    ok: bool,
    rc: int,
    duration_ms: int,
    call_end_epoch: float,
    text: str,
    narrator: str,
    speed_int: int,
    pitch_int: int,
    emotion_arg: str | None,
    out_path: str,
    stdout: str,
    stderr: str,
) -> int:
    """Append one call record. Returns the call_n assigned to this call."""
    global _call_counter
    with _lock:
        _call_counter += 1
        call_n = _call_counter

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    row: dict = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "call_n": call_n,
        "ok": ok,
        "rc": rc,
        "duration_ms": duration_ms,
        "text": text,
        "narrator": narrator,
        "speed": speed_int,
        "pitch": pitch_int,
        "emotion": emotion_arg or "",
        "out_path": out_path,
        "stdout": stdout,
        "stderr": stderr,
    }

    # On failure, look for a crash report (SIGSEGV rc=-11, SIGABRT rc=-6, etc.)
    if not ok and rc < 0:
        crash_path = _find_crash_report(call_end_epoch)
        if crash_path:
            row["crash_report"] = crash_path
            row["crash_summary"] = _extract_crash_summary(crash_path)

    line = json.dumps(row, ensure_ascii=False) + "\n"

    with _lock:
        with _CALLS_LOG.open("a", encoding="utf-8") as f:
            f.write(line)
        if not ok:
            with _ERRORS_LOG.open("a", encoding="utf-8") as f:
                f.write(line)

    return call_n
