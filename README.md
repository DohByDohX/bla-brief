# Meeting Recorder

<!-- Replace OWNER with your GitHub username/org once the repo is pushed. -->
[![CI](https://github.com/OWNER/meeting-recorder/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/meeting-recorder/actions/workflows/ci.yml)

Dual-source audio capture for Windows: records your **microphone** and the
**system audio** (other meeting participants) simultaneously into separate WAV
tracks, then produces a normalized **mixed** file suitable for transcription.

- Single PyAudioWPatch backend for both sources → synchronized start (no
  multi-second offset).
- Streams straight to disk → constant memory even for multi-hour meetings.
- Automatically follows the Windows default output device, so connecting a
  Bluetooth/USB headset routes capture (and the headset mic) accordingly.
- Watch-folder safe: raw tracks live outside the output folder and the mixed
  file is published via an atomic rename, so a `*.wav` watcher only ever sees a
  single finished file.

## Requirements

- Windows (uses WASAPI loopback via PyAudioWPatch)
- Python 3.12+

## Install

```powershell
python -m pip install -e .          # runtime
python -m pip install -e ".[dev]"   # + test/lint/type tooling
```

## Usage

```powershell
python -m meeting_recorder                     # auto-named recording
python -m meeting_recorder -i                  # interactive: pick devices, then name + outputs
python -m meeting_recorder -n "weekly-sync"    # named recording
python -m meeting_recorder -l                  # list audio devices
python -m meeting_recorder --mic 15            # force a specific mic
python -m meeting_recorder --system 17         # force a specific system/loopback device
python -m meeting_recorder --discard-tracks    # keep only the mixed file
python -m meeting_recorder --debug             # verbose logging
```

Press **ENTER** (or **Ctrl+C**) to stop. Output defaults to
`%LOCALAPPDATA%\audacity\Recordings`. Or double-click `Record Meeting.bat`
(which runs in `-i` interactive mode: it prompts for the mic/system devices
before recording, then for a meeting name and which outputs to keep afterward).

## Project layout

```
src/meeting_recorder/
  config.py         tunable constants (rates, chunk sizes, normalization)
  logging_setup.py  logging configuration (--debug)
  devices.py        mic / loopback / output device discovery
  recorder.py       StreamingDualRecorder (capture to disk)
  mixing.py         normalized, sample-rate-aware mixdown
  cli.py            argument parsing + run orchestration
tests/
  test_mixing.py    unit: alignment, resampling, normalization (no hardware)
  test_devices.py   unit: device-selection logic (fake PyAudio)
  test_cli.py       unit: filename sanitization, path layout
  integration/      end-to-end tests that need real audio hardware
```

## Development

```powershell
pytest                    # fast unit tier (default; hardware-free)
pytest -m integration     # full end-to-end suite (needs mic + loopback)
check.bat                 # ruff + mypy + pytest in one shot
```

`.pre-commit-config.yaml` runs ruff + mypy on every commit once the hooks are
installed:

```powershell
pip install pre-commit
pre-commit install
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`, on a
`windows-latest` runner (PyAudioWPatch is Windows-only). It performs the same
checks as `check.bat`: ruff lint, ruff format check, mypy, and the hardware-free
unit tests. Integration tests are excluded in CI because no runner has a real
microphone or WASAPI loopback device.
