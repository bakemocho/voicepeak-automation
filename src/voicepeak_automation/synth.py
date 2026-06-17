"""VOICEPEAK GUI synthesis automation via AppleScript.

Opens a .vpp project file in VOICEPEAK and triggers Export All,
capturing the output WAV file path.

Note: VOICEPEAK CLI cannot load .vpp files. This module bridges that gap
by automating the GUI export path on macOS.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

VOICEPEAK_APP = "Voicepeak"
_DEFAULT_TIMEOUT = 60  # seconds to wait for export dialog


def _run_applescript(script: str, timeout: int = 30) -> str:
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"AppleScript error: {proc.stderr.strip()}")
    return proc.stdout.strip()


def open_vpp(vpp_path: Path) -> None:
    """Open a .vpp file in VOICEPEAK via AppleScript."""
    abs_path = str(vpp_path.resolve())
    script = f"""
tell application "{VOICEPEAK_APP}"
    activate
    open POSIX file "{abs_path}"
end tell
"""
    _run_applescript(script)
    time.sleep(2)  # wait for file to load


def export_all(output_dir: Path, timeout: int = _DEFAULT_TIMEOUT) -> None:
    """Trigger File > Export All in VOICEPEAK and set output directory.

    This uses the keyboard shortcut or menu item for Export All.
    The output directory dialog is handled via System Events.
    """
    abs_dir = str(output_dir.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)

    script = f"""
tell application "{VOICEPEAK_APP}"
    activate
end tell
delay 0.5
tell application "System Events"
    tell process "{VOICEPEAK_APP}"
        -- File menu → Export All (or use keyboard shortcut if known)
        click menu item "Export All" of menu "File" of menu bar 1
        delay 2
        -- Handle the save panel: set directory and confirm
        keystroke "G" using {{command down, shift down}}
        delay 1
        keystroke "{abs_dir}"
        delay 0.5
        key code 36
        delay 1
        key code 36
    end tell
end tell
"""
    _run_applescript(script, timeout=timeout)


def synthesize_vpp(
    vpp_path: Path,
    output_dir: Path,
    timeout: int = _DEFAULT_TIMEOUT,
) -> None:
    """Open .vpp in VOICEPEAK GUI and export audio to output_dir.

    Requires:
      - macOS with Accessibility access granted to Terminal/Claude Code
      - VOICEPEAK installed at /Applications/voicepeak.app
    """
    open_vpp(vpp_path)
    export_all(output_dir, timeout=timeout)
