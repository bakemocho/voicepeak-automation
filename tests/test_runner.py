from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voicepeak_automation.runner import run_task
from voicepeak_automation.task import parse_task


class RunnerTest(unittest.TestCase):
    def test_run_task_dry_run_generates_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task_path = tmp_path / "task.json"
            output_dir = tmp_path / "wav"

            task_payload = {
                "project": "dry-run-test",
                "settings": {
                    "output_dir": str(output_dir),
                    "max_chunk_chars": 40,
                    "formula_mode": "strip",
                },
                "items": [
                    {
                        "id": "line-001",
                        "text": "This is a long sentence. " * 5,
                    }
                ],
            }
            task_path.write_text(json.dumps(task_payload, ensure_ascii=False), encoding="utf-8")

            task = parse_task(task_path)
            result = run_task(task=task, dry_run=True)

            self.assertEqual(result.task_project, "dry-run-test")
            self.assertTrue(result.dry_run)
            self.assertGreater(len(result.chunk_results), 1)
            self.assertTrue(output_dir.exists())
            self.assertFalse(any(output_dir.glob("*.wav")))


if __name__ == "__main__":
    unittest.main()
