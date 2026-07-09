# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.3.1] - 2026-07-09

### Fixed
- Interactive mode: picking a system (loopback) device other than the current
  Windows default output produced an empty system track. The keepalive tone was
  played on the default output, leaving the chosen loopback endpoint idle so
  WASAPI delivered no frames. `find_devices()` now routes the keepalive to the
  output that feeds the chosen loopback (`_find_output_for_loopback`), falling
  back to the default output when no match exists.
- Post-recording summary showed the in-progress `.part` temp filename for the
  mixed file instead of the published name. `create_mixed_file()` accepts an
  optional `report_path` so the summary reports the final published file.

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
