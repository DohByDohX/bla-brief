# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.4.0] - 2026-07-18

### Added
- Framed console UI (built on `rich`): a minimalistic banner, device-selection
  panels with the auto-detected default marked, a live recording panel (timer +
  mic/system size meters), and an output-summary panel. Single Tesla-red
  (`#E82127`) accent, no emoji.
- New `meeting_recorder.ui` module centralizes all presentation; the rest of the
  package stays logic-only.

### Changed
- Device listing/selection and the post-recording summary now render through the
  UI module. Interactive prompts use a styled caret.

### Notes
- The styled UI is gated on an interactive TTY: when stdout is piped/redirected
  (watch-folder, automation), everything falls back to plain text — the
  non-interactive output shape is unchanged.
- The live recording panel is a fixed-width inline panel so it stays correct
  across terminal resizes (no stacked/duplicated panels).
- Adds a runtime dependency on `rich~=13.7`.

## [4.3.3] - 2026-07-18

### Changed
- Interactive picker (`-i`) and device listing (`-l`) now show only WASAPI mic
  devices. Windows exposes each physical mic once per host API (MME, DirectSound,
  WASAPI), which cluttered the list with 2-3 duplicates of every device; since
  the recorder only ever captures via WASAPI, the extras are noise. Falls back to
  all inputs when WASAPI is unavailable or has no inputs. `--mic <id>` still
  accepts any device index for power users.

## [4.3.2] - 2026-07-09

### Fixed
- Post-recording summary listed files that had just been pruned. When keeping
  only the mixed output, the "Output files" block still showed the mic-only and
  system-only tracks even though they were deleted. The summary now reports only
  the files actually left on disk, with their real sizes.
- "Cannot mix" fallback now truly keeps the one usable track. Previously, if a
  mix was requested but one track was empty, the log claimed the surviving track
  was kept while the prune step deleted it (leaving nothing). The usable track is
  now preserved and reported.

### Changed
- Output reporting moved from `create_mixed_file()` (which now just confirms the
  mix) into the CLI, which owns knowledge of the interactive keep-set.

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
