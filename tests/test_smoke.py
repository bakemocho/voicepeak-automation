import unittest

from voicepeak_automation.cli import build_parser


class ParserSmokeTest(unittest.TestCase):
    def test_parser_builds(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.prog, "voicepeak-automation")


if __name__ == "__main__":
    unittest.main()
