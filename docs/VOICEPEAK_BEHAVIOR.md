# Voicepeak CLI Behavior Notes

Last updated: 2026-02-18

## Scope

Environment used for probing:

- macOS
- binary: `/Applications/voicepeak.app/Contents/MacOS/voicepeak`
- app version: `1.2.20` (`CFBundleShortVersionString`)

All findings below are from direct CLI invocations in this environment.

## Confirmed Behaviors

### 1) Discovery commands

- `--list-narrator` returns narrator names on `stdout`.
- `--list-emotion <Narrator>` returns emotion keys for that narrator on `stdout`.
- Both commands also emit noisy `stderr` lines (`[debug]...`, `iconv_open is not supported`).

Observed emotion keys:

- `Koharu Rikka`: `hightension`, `livid`, `lamenting`, `despising`, `narration`
- `Frimomen`: `happy`, `angry`, `sad`, `ochoushimono`

### 2) Character length limit

- A single `-s` / `-t` input over 140 characters fails with return code `1` and message:
  - `In this version, the character limit for a single run is 140 characters...`
- Exactly 140 characters is accepted.

### 3) Invalid parameter handling is unsafe

Observed failure modes:

- Invalid narrator (`-n NoSuchNarrator`) can terminate with signal (`rc=-6`, abort).
- Invalid emotion key (`-e nosuch=50`) can terminate with signal (`rc=-6`, abort).
- Extreme emotion values (e.g. `hightension=150`) can terminate with signal (`rc=-11`, segfault).

These crashes are consistent with the popup you reported (“Voicepeak unexpectedly quit”).

Crash reports in `~/Library/Logs/DiagnosticReports` matched this:

- `SIGABRT` (`EXC_CRASH`) for invalid narrator/emotion-key style failures
- `SIGSEGV` (`EXC_BAD_ACCESS`) for invalid emotion value style failures
- faulting thread was the `JUCE Message Thread` (main thread) in sampled reports

### 4) Numeric range behavior

- `--speed` out of documented range is accepted and appears clamped internally:
  - `49` and `0` matched `50` output characteristics.
  - `201` and `300` matched `200` output characteristics.
- `--pitch` out of documented range is accepted and appears clamped internally:
  - `301` matched `300` output characteristics.
  - `-301` matched `-300` output characteristics.

### 5) I/O behavior

- Missing output directory fails (`rc=1`) with file open error.
- Existing output path is overwritten.
- `-s` and `-t` can be passed together; observed behavior matched `-s` input output.
- Narrator name matching is case-sensitive (`Koharu Rikka` works, uppercase/lowercase variants crashed).

## Safeguards Added in This Repo

Implemented in `src/voicepeak_automation/runner.py`:

- Preflight narrator validation via `--list-narrator`.
- Preflight emotion key validation via `--list-emotion <narrator>`.
- Fallback narrator selection when configured narrator is invalid.
- Emotion sanitization:
  - drop unknown keys
  - clamp value range to `0..100`
  - fallback to a safe expression if all pairs are invalid
  - omit `-e` when no emotion key is available
- Speed clamping to `50..200` before invocation.
- Timeout-protected subprocess execution.
- Synthesis failure is recorded and skipped per chunk (no retry loop, continue next chunk).

## Ongoing Investigation

Open points still worth probing:

- Whether there is any stable `pitch` policy we should enforce in task schema.
- Whether `-t` has encoding constraints beyond UTF-8 and line-break nuances.
- Whether some crash signatures are narrator-specific or reproducible across all narrators.
- Whether we should expose a strict mode that fails fast on any warning (instead of sanitize-and-continue).
