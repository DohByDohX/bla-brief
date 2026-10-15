"""Central configuration constants for the meeting recorder.

Keeping tunables here (rather than scattered as literals through the logic)
means behavior changes never require hunting through the audio code.
"""

from pathlib import Path

# Default location for finished, mixed recordings (the "watch folder").
OUTPUT_DIR: Path = Path.home() / "AppData" / "Local" / "audacity" / "Recordings"

# Audio engine settings.
SAMPLE_RATE: int = 48000
CHUNK_SIZE: int = 1024  # frames per buffer (~21 ms at 48 kHz)

# Post-processing / mixing.
NORM_TARGET: float = 0.95  # peak the mixed file is normalized to (headroom below clipping)
ALIGN_THRESHOLD_S: float = 0.05  # ignore sub-50ms start offsets when aligning tracks
MIX_CHUNK_FRAMES: int = SAMPLE_RATE * 10  # streaming mix window (bounds memory)

# Forgotten-to-stop safety net. Auto-stops the recording after this many
# minutes so a laptop left running cannot fill the disk overnight. 0 = unlimited.
MAX_DURATION_MIN: int = 120

# -- Local transcription (faster-whisper) ------------------------------------
# Where finished transcripts (.md) are written. This is also the folder the
# downstream meeting catch-up automation watches.
TRANSCRIPT_DIR: Path = (
    Path.home() / "OneDrive - Tesla" / "Tesla.pruthviraj" / "Work" / "Ops" / "Meeting Notes" / "Raw"
)
STT_MODEL: str = "small.en"  # Whisper model size (small.en = accuracy/speed balance, English-only)
STT_DEVICE: str = "auto"  # "auto" (GPU-first, CPU fallback), "cuda", or "cpu"
STT_LANGUAGE: str = "en"  # source language hint passed to the model
# PowerShell wrapper fired (detached) after a successful transcription. It is
# self-gating and locked, so firing it unconditionally is safe.
POST_TRANSCRIBE_SCRIPT: Path = (
    Path.home()
    / "OneDrive - Tesla"
    / "Tesla.pruthviraj"
    / "Work"
    / "Ops"
    / "Meeting Notes"
    / "_automation"
    / "process-meetings.ps1"
)

# -- Speaker diarization (SpeechBrain ECAPA) ----------------------------------
# Ungated voice-embedding model used to tell speakers apart (clustered into
# "Speaker N" labels, not real names). Cached under the user's profile so it
# resolves the same way regardless of the recorder's working directory.
DIARIZE_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"
DIARIZE_CACHE_DIR: Path = Path.home() / ".cache" / "meeting_recorder" / "speaker_embedder"
# Diarization v2 (4.10.0). Tuned on 7 hand-labelled meetings and checked on 14
# unlabelled ones; evidence and rationale in spike/port_plan.md (spike branch).
#
# VAD: a pause this long splits speech into segments. 300 ms (was 500) splits
# more turns that two speakers share without a long gap.
DIARIZE_VAD_MIN_SILENCE_MS: int = 300
DIARIZE_MIN_SEGMENT_S: float = 1.0  # drop VAD slivers too short to embed meaningfully
# Agglomerative clustering distance threshold (euclidean, on L2-normalized
# embeddings). Each meeting picks the threshold in DIARIZE_THRESHOLD_SWEEP
# (min, max, step) whose clustering has the best cosine silhouette; this fixed
# value is only the fallback when no candidate gives 2..n-1 clusters.
DIARIZE_CLUSTER_THRESHOLD: float = 1.2
DIARIZE_THRESHOLD_SWEEP: tuple[float, float, float] = (1.00, 1.50, 0.05)
# Two-stage clustering: only segments at least this long build the clusters;
# shorter ones join the nearest cluster afterwards and can't start a new one
# (their embeddings are noisy and produced phantom speakers).
DIARIZE_TWO_STAGE_MIN_S: float = 2.0
# Re-segmentation: segments at least DIARIZE_RESEG_MIN_S long are re-checked in
# windows of DIARIZE_RESEG_WIN_S every DIARIZE_RESEG_HOP_S. A segment is split
# where another existing cluster wins for at least DIARIZE_RESEG_RUN_S by a
# cosine margin of at least DIARIZE_RESEG_MARGIN (two speakers in one segment).
DIARIZE_RESEG_MIN_S: float = 3.0
DIARIZE_RESEG_WIN_S: float = 1.5
DIARIZE_RESEG_HOP_S: float = 0.75
DIARIZE_RESEG_RUN_S: float = 1.0
DIARIZE_RESEG_MARGIN: float = 0.05
