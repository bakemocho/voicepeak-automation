from voicepeak_automation.cli import build_parser


def test_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "voicepeak-automation"
