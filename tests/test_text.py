from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voicepeak_automation.dictionary import convert_alphabet_to_katakana, load_dictionaries
from voicepeak_automation.text import apply_formula_mode, prepare_chunks


class TextProcessingTest(unittest.TestCase):
    def test_formula_modes(self) -> None:
        text = "A [x+y] B"

        self.assertEqual(apply_formula_mode(text, "keep"), "A [x+y] B")
        self.assertEqual(apply_formula_mode(text, "strip"), "A  B")
        self.assertEqual(apply_formula_mode(text, "placeholder"), "A 数式 B")

    def test_dictionary_loader_and_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dic_dir = Path(tmp_dir) / "dic"
            dic_dir.mkdir()
            (dic_dir / "base.dic").write_text("HELLO ハロー\n, テン\n", encoding="utf-8")

            mapping = load_dictionaries(dic_dir)
            self.assertEqual(mapping["HELLO"], "ハロー")

            converted = convert_alphabet_to_katakana("Hello, world", mapping)
            self.assertIn("ハロー", converted)

    def test_prepare_chunks_respects_max_chars(self) -> None:
        dictionaries = {"HELLO": "ハロー"}
        text = "Hello. " * 40

        chunks = prepare_chunks(
            text=text,
            dictionaries=dictionaries,
            formula_mode="strip",
            max_chunk_chars=60,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 60 for c in chunks))


if __name__ == "__main__":
    unittest.main()
