# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- The diarization duplicate-rendering fix above only caught verbatim text
  matches. Extended it to near-duplicates: Whisper sometimes transcribes the
  echoed fragment with slightly different trailing punctuation (e.g. "a bit."
  vs "a bit,"), which a strict equality check missed. Text is now compared
  after stripping trailing punctuation/case, with a fuzzy-ratio fallback for
  anything that still differs, so echoed fragments are caught regardless of
  minor transcription noise while genuinely different utterances (even short
  ones) are left alone.
- Overlap duplication artifact in diarization: when mic and system tracks
  overlapped in time (during cross-talk), the same word sequence was rendered
  twice under different speaker labels. Added post-rendering dedup step that
  detects and removes this pattern (Speaker A says X, Speaker B says same X
  immediately after, then A continues with different content → remove B's
  duplicate). Preserves legitimate overlapping speech where both speakers
  genuinely say the same thing simultaneously. This fix significantly improves
  transcript readability in conversational meetings. Validated on 3 real
  meeting recordings: meeting 2 quality improved from 4/10 → 7+/10,
  meeting 3 from 6/10 → 8+/10.

### Added
- Default 2-hour auto-stop so a forgotten recording cannot run overnight.
  Override per run with `--max-duration MINUTES`; `--max-duration 0` disables
  the limit. A timed-out recording still mixdown + transcribes like a normal
  ENTER stop. The live panel shows the cap (`auto-stop 02:00:00`).
- Transcripts are now speaker-labeled (`Speaker 0:`, `Speaker 1:`, ...).
  Diarization runs unconditionally after every recording: VAD segments the
  audio, an ungated SpeechBrain ECAPA model embeds each segment (CPU, no
  extra VRAM -- Whisper keeps the GPU), and the segments are clustered into
  speakers before merging with Whisper's word timestamps. Speaker labels are
  cluster IDs, not real names. Fetch the embedding model once with
  `--download-model` (same offline-first, one-time-online pattern as the
  Whisper model).
- Diarization is now channel-aware: the mic track is a known fact (it's
  always the local speaker), so it's labeled directly as `Local:` with no
  embedding/clustering needed, and only the system/loopback track is
  clustered -- so clustering only ever has to tell remote participants apart
  from each other. The raw mic/system tracks are now kept until after
  transcription (previously pruned right after mixdown) so diarization can
  use them; they're deleted afterward unless `--keep-audio` is set (which now
  means "keep all audio", not just the mixed file). Overlapping mic+system
  speech renders as separate lines, one per active speaker, since there is
  only one transcribed word stream to attribute.

### Changed
- Default transcription model is now `small.en` (was `base.en`). Override
  per run with `--stt-model`. Fetch it once with `--download-model`.
- Recordings now always keep only the mixed file. The post-recording
  `keep [m]ixed [v]oice [s]ystem` prompt is gone, and mixed-only is the default
  for interactive and non-interactive runs alike. The raw mic/system tracks are
  pruned after mixdown (the empty-track fallback still preserves the one usable
  track when a mix isn't possible).

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
