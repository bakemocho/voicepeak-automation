from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voicepeak_automation.task import TaskValidationError, parse_task


class TaskParsingTest(unittest.TestCase):
    def _write_json(self, path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_parse_task_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "task.json"
            self._write_json(
                task_path,
                {
                    "project": "demo",
                    "items": [{"id": "line-1", "text": "Hello world"}],
                },
            )

            task = parse_task(task_path)

            self.assertEqual(task.project, "demo")
            self.assertEqual(len(task.items), 1)
            self.assertEqual(task.settings.formula_mode, "strip")
            self.assertEqual(task.settings.max_chunk_chars, 140)

    def test_parse_task_rejects_unknown_formula_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = Path(tmp_dir) / "task.json"
            self._write_json(
                task_path,
                {
                    "project": "demo",
                    "settings": {"formula_mode": "ask-openai"},
                    "items": [{"id": "line-1", "text": "Hello world"}],
                },
            )

            with self.assertRaises(TaskValidationError):
                parse_task(task_path)


if __name__ == "__main__":
    unittest.main()
