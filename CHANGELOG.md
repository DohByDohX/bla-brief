# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.3.0] - 2026-07-09

### Added
- Interactive mode (`-i` / `--interactive`): before recording, choose the mic and
  system (loopback) devices from a numbered list (Enter accepts the auto-detected
  default); after recording, type a meeting name and select which outputs to keep
  (mixed / your voice / system).
- `--system <id>` flag to select the system (loopback) device non-interactively,
  mirroring `--mic`.
- `Record Meeting.bat` now launches in interactive mode.

### Changed
- `find_devices()` accepts a `loopback_index` override, and `StreamingDualRecorder`
  accepts a matching `loopback_index` argument.
- Post-recording finalize refactored into composable helpers (`_produce_outputs`,
  `_rename_recording`); mixed-file production honors the interactive keep-set.

### Notes
- All interactive prompts are opt-in behind `-i` and fall back to safe defaults on
  EOF/Ctrl+C, so non-interactive and watch-folder use is unchanged.

## [4.2.0] - 2026-07-07

### Added
- Initial packaged release: `src/` layout with the `meeting_recorder` package,
  pinned dependencies, ruff/mypy/pytest tooling, and a unit + integration
  (hardware) test suite. Refactored from the original single-file script.
