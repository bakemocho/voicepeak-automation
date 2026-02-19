# voicepeak-automation

Local-first toolkit to automate Voicepeak operation from reproducible scripts.

## Why

Current one-off scripts are brittle and hard to maintain. This repo rebuilds the workflow with:

- clear command interface
- typed task inputs
- deterministic execution logs
- testable, modular adapters

## Operating Model

- User-triggered automation only.
- No autonomous browsing or background account actions.
- Designed for local execution on your own machine.

## Current Capabilities

- Task schema validation (`validate`)
- JSON task runner (`run` / `--dry-run`)
- Dictionary-based alphabet-to-katakana conversion (`*.dic`)
- Formula handling modes (`strip`, `keep`, `placeholder`)
- Chunk splitting for long text
- Voicepeak safety guards (narrator/emotion preflight, clamping, timeout, fail-and-skip)

## Dictionary License Policy

- This project supports optional user-provided `*.dic` files.
- This repository does not bundle or auto-download third-party pronunciation dictionaries.
- If you use external dictionary data (for example, BEP-derived files), you are responsible for acquisition and license compliance.
- Before redistribution, commercial use, or bundled delivery, confirm that your selected dictionary license is compatible with your distribution model.

## Repository Layout

- `src/voicepeak_automation/`: core package and CLI
- `examples/`: sample task files
- `docs/`: architecture and design notes
- `tests/`: automated tests

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

voicepeak-automation validate examples/task.sample.json
voicepeak-automation run --task examples/task.sample.json --dry-run
```

Run without `--dry-run` when `voicepeak` is installed at your configured path.

## Legal

- License: MIT (`LICENSE`)
- Legal notice: `LEGAL.md`

## Behavior Notes

- Voicepeak probing notes: `docs/VOICEPEAK_BEHAVIOR.md`
