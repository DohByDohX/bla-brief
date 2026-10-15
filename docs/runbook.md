# Deployment Runbook: Local Transcription + Diarization

Operational notes for rolling out STT + diarization (v4.9.0+; diarization v2 since v4.10.0) to a new
machine or troubleshooting an existing one. See [README.md](../README.md) for
user-facing usage; this doc is for setup/ops issues.

## First-run setup on a new machine

1. `python -m pip install -e ".[transcribe]"` — installs faster-whisper,
   speechbrain, scikit-learn, the CUDA DLL wheels, and truststore.
2. On an approved/trusted network (one that can reach huggingface.co),
   run `python -m meeting_recorder --download-model` once. This is the
   **only** step that goes online; it fetches both the Whisper model and the
   diarization embedder into the local cache (`~/.cache/huggingface`,
   `~/.cache/meeting_recorder/speaker_embedder`).
3. After that, every recording transcribes **fully offline**
   (`HF_HUB_OFFLINE=1` is enforced in code) — no further network access is
   attempted, by design.

## Expected latency (measured, v4.9.0)

Post-recording wait (mixdown + diarization + transcription) for a 3-minute
meeting, same audio on both backends:

| Stage | GPU (`cuda`) | CPU |
|---|---|---|
| Diarization (VAD + ECAPA embed + cluster) | ~5.5s | ~5.1s |
| Whisper model load | ~4.3s | ~2.8s |
| Whisper transcribe | ~10.3s | ~37.1s |
| **Total** | **~20s** | **~45s** |

**v4.10.0 (diarization v2):** diarization costs about 2.4x the v4.9.0 figure.
Measured on a 38-min meeting (CPU), it takes about 10.4 s per audio-minute
vs 4.3 s before; most of the extra is re-segmentation windows. So expect
roughly 12 s instead of about 5 s for the 3-min case above, and about
10 min for a one-hour meeting. Transcription times are unchanged.

Diarization always runs on CPU by design, so its cost doesn't depend on the
device. Transcription scales with recording length, so the CPU/GPU
gap widens on longer meetings. CPU-only machines are still well within
acceptable wait times for a typical meeting.

## Troubleshooting

**"faster-whisper is not installed" error.** The `transcribe` extra isn't
installed; run step 1 above, or pass `--no-transcribe` to record without it.

**Transcription fails with an `SSLError`/`CERTIFICATE_VERIFY_FAILED` trying
to reach huggingface.co.** Something caused an online call despite the
offline-first design — usually a script bypassing `transcribe_with_speakers`/
`transcribe_file` (which both call `_enable_offline()` first) and calling
`WhisperModel(...)` directly. Normal recordings never hit this path.

**"No transcription backend could be loaded."** The model hasn't been
downloaded yet (run `--download-model`) or the CUDA wheels are missing/broken
for `cuda`/`auto`. Use `--stt-device cpu` to confirm the model itself is
cached correctly, independent of the GPU path.

**Mic track is silent / `Speaker None` dominates the transcript.** Check the
physical mic isn't muted and is actually the selected device (`-l` lists
devices; `--mic <ID>` to override). A silent mic track produces an empty or
near-empty transcript, not a crash.

**Custom `--output-dir` that doesn't exist yet.** Fixed in v4.9.0+ (the
directory is created before mixdown starts, not only after). If you see a
`FileNotFoundError` writing the `.wav.part` file on an older build, update.

**Process exits immediately with code 1, no recording happens, and stderr
shows a `UnicodeEncodeError` from Rich.** Fixed in v4.9.0+. Happened when
stdout was piped/redirected (Task Scheduler, log redirection, a wrapper
script) without `PYTHONIOENCODING=utf-8` set — the styled banner used a
box-drawing character the legacy Windows console renderer couldn't encode.
Update if you see this on an older build.

## Flags relevant to deployment

- `--no-transcribe` — record only, skip STT/diarization entirely.
- `--stt-device {auto,cuda,cpu}` — force a backend; `auto` tries CUDA then
  falls back to CPU silently. An explicit `cuda`/`cpu` fails loudly instead of
  silently falling back, which is useful when diagnosing GPU setup issues.
- `--stt-model` — override the default `small.en`; must be fetched via
  `--download-model <model>` first if not already cached.
- `--keep-audio` — keep the mixed file and raw mic/system tracks after
  transcription (normally deleted). Useful for debugging a bad transcript.
- `--no-automation` — skip firing the downstream catch-up script after
  transcription.

## Canary rollout (week of 2026-10-07)

Transcription is enabled by default for all recordings during canary. Goal:
confirm real day-to-day usage holds up before GA on 2026-10-21.

**What to watch, per recording:**
- Did a transcript get written at all (check for the `.md` file; a silent
  failure logs a warning but doesn't crash)?
- Rough diarization quality (v4.10.0+): the number of `Speaker N` labels
  roughly matches the attendees (a 1:1 shows 2; nobody who spoke a lot is
  missing). Watch for phantom speakers holding only a line or two, and for
  one person split across two labels. `Speaker None` should stay rare.
  Spot-check, don't need to grade every meeting.
- Wall-clock wait after stopping a recording: should track the latency
  notes above (v4.10.0 adds about 10 s of diarization per audio-minute on
  CPU); flag anything wildly outside that range.
- Any crash/traceback in the console output, especially around mixdown,
  model loading, or the catch-up automation firing.

**Where to log issues:** add a dated entry to the
[canary log](#canary-log) at the end of this doc for quality issues and
crashes alike. Fixes land as regular commits, same as the fixes found during
Phase 2 validation (mixdown directory creation, banner crash).

**Go/no-go bar for GA (2026-10-21):** no crashes on normal use, diarization
quality holding up on the spot-checks above (speaker counts matching
attendees, no phantom or split speakers), and no new latency or
offline-network regressions. A known limitation (acoustic bleed when a remote participant
uses speakers instead of a headset) is accepted as-is, not a blocker.

## Canary log

One entry per issue, newest first. Format:

```
### YYYY-MM-DD - short title
- Meeting: length, number of attendees, setup (headset/speakers, call type)
- Finding: what went wrong (wrong speaker count, phantom/split speaker,
  crash, latency); quote a line or two if it helps
- Fix: commit, or "open"
```

_No entries yet._
