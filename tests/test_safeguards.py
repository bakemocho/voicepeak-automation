from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicepeak_automation.runner import RunnerError, run_task
from voicepeak_automation.task import parse_task


class SafeguardsTest(unittest.TestCase):
    def test_invalid_options_are_sanitized_before_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "task.json"
            output_dir = Path(tmp_dir) / "wav"
            payload = {
                "project": "guard-test",
                "settings": {
                    "output_dir": str(output_dir),
                    "speaker": "NoSuchSpeaker",
                    "emotion": "nosuch=999,livid=-5",
                    "speed": 300,
                },
                "items": [{"id": "line-001", "text": "テスト"}],
            }
            task_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            task = parse_task(task_path)

            calls: list[list[str]] = []

            def fake_run_command(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                calls.append(args)
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            with (
                patch("voicepeak_automation.runner._list_narrators", return_value=["Koharu Rikka"]),
                patch(
                    "voicepeak_automation.runner._list_emotion_keys",
                    return_value=["hightension", "livid", "narration"],
                ),
                patch("voicepeak_automation.runner._run_command", side_effect=fake_run_command),
            ):
                result = run_task(task, dry_run=False)

            self.assertEqual(len(calls), 1)
            synthesis_cmd = calls[0]
            self.assertIn("-n", synthesis_cmd)
            self.assertIn("Koharu Rikka", synthesis_cmd)
            self.assertIn("--speed", synthesis_cmd)
            self.assertIn("200", synthesis_cmd)
            self.assertIn("-e", synthesis_cmd)
            self.assertIn("livid=0", synthesis_cmd)
            self.assertEqual(result.errors, [])
            self.assertGreaterEqual(len(result.warnings), 3)

    def test_synthesis_error_is_collected_and_next_chunk_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "task.json"
            output_dir = Path(tmp_dir) / "wav"
            payload = {
                "project": "error-continue-test",
                "settings": {
                    "output_dir": str(output_dir),
                    "speaker": "Koharu Rikka",
                    "emotion": "narration=100",
                },
                "items": [
                    {"id": "line-001", "text": "テスト1"},
                    {"id": "line-002", "text": "テスト2"},
                ],
            }
            task_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            task = parse_task(task_path)

            call_count = {"n": 0}

            def flaky_run_command(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RunnerError("mock crash")
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            with (
                patch("voicepeak_automation.runner._list_narrators", return_value=["Koharu Rikka"]),
                patch("voicepeak_automation.runner._list_emotion_keys", return_value=["narration"]),
                patch("voicepeak_automation.runner._run_command", side_effect=flaky_run_command),
            ):
                result = run_task(task, dry_run=False)

            self.assertEqual(len(result.chunk_results), 2)
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(call_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
