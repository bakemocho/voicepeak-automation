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

## Planned Capabilities

- Task-runner: execute synthesis steps from JSON task files.
- Watch mode: monitor task queue and process sequentially.
- Adapter layer: separate app-control backend from workflow logic.
- Export bridge: produce artifacts usable as LLM context.

## Repository Layout

- `src/voicepeak_automation/`: core package and CLI
- `examples/`: sample task files
- `docs/`: architecture and design notes
- `tests/`: basic tests

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

voicepeak-automation validate examples/task.sample.json
voicepeak-automation run --task examples/task.sample.json
```

## Legal

- License: MIT (`LICENSE`)
- Legal notice: `LEGAL.md`
