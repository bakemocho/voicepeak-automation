# Architecture (Draft)

## Layers

1. CLI layer (`cli.py`)
2. Task schema layer (`task.py`)
3. Text processing layer (`dictionary.py`, `text.py`)
4. Runner layer (`runner.py`)
5. Artifact layer (`output/*.wav`, logs)
6. Safety layer (parameter preflight, clamping, timeout, fail-and-skip)

## Design Goal

Keep backend-specific fragility isolated from workflow logic so backend changes do not break task schema.

## Current Flow

1. Parse and validate JSON task.
2. Load optional `.dic` dictionary set.
3. Apply formula mode.
4. Convert latin tokens with dictionary fallback.
5. Split into chunk-sized utterances.
6. Build and execute `voicepeak` commands (or print plan in dry-run).

## Runtime Safety

- Validate narrator against `--list-narrator`.
- Validate emotion keys against `--list-emotion <narrator>`.
- Clamp speed to safe range.
- Wrap subprocess calls with timeout.
- Record synthesis failure per chunk and continue with next chunk.
