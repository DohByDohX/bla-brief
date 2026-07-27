# AGENTS.md — Meeting Recorder

Dual-source (mic + WASAPI system loopback) synchronized audio recorder for **Windows**,
Python 3.12. See [README.md](README.md) for install, usage, and the full project layout —
do not duplicate that content here.

> ## ⚠️ OFFLINE-FIRST — non-negotiable (company PC)
> This runs on a corporate machine. **Transcription must make ZERO network calls during a
> recording.** The faster-whisper model is read from the local cache only, with
> `HF_HUB_OFFLINE=1` enforced in code ([transcription.py](src/meeting_recorder/transcription.py)).
> - The **only** sanctioned online action is `--download-model` (one-time cache fetch), and even
>   that validates TLS against the **OS certificate store** via `truststore` — it never disables
>   or bypasses certificate verification.
> - Never add code that fetches models/weights/data at record time, hard-codes credentials, or
>   weakens TLS (`verify=False`, `CERT_NONE`, custom/unpinned CAs). Keep everything auditable and
>   local. Models come only from official sources (HuggingFace) through the download step.

## Setup & commands

Run everything **from this project root** (`meeting_recorder/`), not the parent workspace —
pytest/ruff/mypy only discover their config in [pyproject.toml](pyproject.toml) from here.

```powershell
python -m pip install -e ".[dev]"   # editable install (required: src layout)
pytest                              # unit tier — fast, hardware-free (default)
pytest -m integration               # end-to-end — needs a real mic + loopback (~2 min)
check.bat                           # ruff + ruff format --check + mypy + pytest, one shot
```

Always run `check.bat` (or its four commands) **before considering a change done**.

## Architecture

Single package `src/meeting_recorder/`, one responsibility per module:

| Module | Responsibility |
|--------|----------------|
| [config.py](src/meeting_recorder/config.py) | Tunable constants (rates, chunk sizes, normalization) — no logic |
| [devices.py](src/meeting_recorder/devices.py) | Mic/loopback/output discovery + interactive picker |
| [recorder.py](src/meeting_recorder/recorder.py) | `StreamingDualRecorder` — captures both streams to disk |
| [mixing.py](src/meeting_recorder/mixing.py) | Pure, streaming, sample-rate-aware mixdown |
| [transcription.py](src/meeting_recorder/transcription.py) | Offline-first local STT (faster-whisper); model download + atomic transcript write |
| [cli.py](src/meeting_recorder/cli.py) | Arg parsing, prompts, run orchestration |

## Engineering methodology — follow for EVERY change

1. **Plan before code.** For anything spanning >2 files or ~30 lines, state what you will and
   won't change, and the key design decisions, before editing. Surface tradeoffs; don't guess.
2. **Backward-compatible, opt-in.** New behavior goes behind a flag/parameter that defaults to
   the current behavior. Never break existing CLI contracts, watch-folder automation, or tests.
   (Example: interactive mode is gated behind `-i`; non-interactive runs are unchanged.)
3. **Separation of concerns / testable seams.** Keep pure logic (parsing, path building, DSP)
   separate from I/O (audio streams, `input()`, disk). Pure functions get unit tests.
4. **Robust boundary handling.** Validate and default at input boundaries only — prompts fall
   back on empty/invalid/EOF/Ctrl+C; file ops use safe-unlink. Don't add defensive code for
   states that cannot occur internally.
5. **Types, lint, format, docstrings.** Full type hints on new code; `ruff` + `mypy` clean;
   docstrings explain *why*, not *what*.
6. **Automated tests.** Add fast, hardware-free unit tests for new logic. New hardware-dependent
   behavior goes in `tests/integration/` behind the `integration` marker. Do not merge a feature
   whose logic has no test.
7. **Scope discipline.** Only touch what the task needs. No unrequested refactors, dependencies,
   abstractions, or reformatting of untouched code.
8. **Version control.** Commit logical units with clear messages once `check.bat` passes; don't
   leave the tree broken or the work uncommitted. Bump the version ([pyproject.toml](pyproject.toml)
   + [__init__.py](src/meeting_recorder/__init__.py)) for user-facing changes (SemVer: MINOR for
   features, PATCH for fixes).

## Project-specific conventions & gotchas

- **`src` layout** — the package is under `src/`; it is only importable after `pip install -e .`.
- **Test helpers live in `tests/support.py`**, imported as `from support import ...`. Do NOT put
  importable helpers in a `conftest.py` — multiple conftests in the tree collide on a bare import.
- **Integration determinism** — [tests/integration/conftest.py](tests/integration/conftest.py)
  plays a tone on the default output so the loopback always has audio. Keep it; without it the
  recording tests fail on a silent machine.
- **Logging vs UI** — library modules (`devices`, `recorder`, `mixing`) use `logging`, never
  `print`. `print`/`input` are only for the interactive console UI in `cli.py`.
- **mypy** checks `src/meeting_recorder` only; PyAudioWPatch is untyped (ignored via override).
- **Windows-only** — WASAPI loopback via PyAudioWPatch. Do not add a cross-platform abstraction.

## Invariants that must not break

- **Watch-folder safety:** raw mic/system tracks are written OUTSIDE the output folder; the mixed
  file is built as a hidden `.<name>.wav.part` and **atomically renamed** to its final `.wav`.
  A `*.wav` watcher must only ever see one finished file.
- **Track sync:** both streams start together and the mixer end-anchors them via the measured/auto
  offset. Changes to capture or mixing must preserve mic/system alignment.
