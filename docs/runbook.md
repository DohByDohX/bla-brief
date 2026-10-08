# Deployment Runbook: Local Transcription + Diarization

Operational notes for rolling out STT + diarization (v4.9.0+) to a new
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

Diarization cost is roughly fixed regardless of device (it always runs on
CPU by design). Transcription scales with recording length, so the CPU/GPU
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
