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
# Agglomerative clustering distance threshold (euclidean, on L2-normalized
# embeddings). Lower = more speakers (over-splits); higher = fewer speakers
# (over-merges). Calibrated against a hand-counted recording; validate further
# before trusting it across very different meeting sizes/styles.
DIARIZE_CLUSTER_THRESHOLD: float = 1.3
DIARIZE_MIN_SEGMENT_S: float = 0.3  # drop VAD slivers too short to embed meaningfully
# Mic-channel segments are a known fact (it's always this one person), so they
# are labeled directly -- no embedding/clustering needed for that channel.
DIARIZE_LOCAL_SPEAKER_LABEL: str = "Local"
