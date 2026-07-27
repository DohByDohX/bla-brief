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

# -- Local transcription (faster-whisper) ------------------------------------
# Where finished transcripts (.md) are written. This is also the folder the
# downstream meeting catch-up automation watches.
TRANSCRIPT_DIR: Path = (
    Path.home() / "OneDrive - Tesla" / "Tesla.pruthviraj" / "Work" / "Ops" / "Meeting Notes" / "Raw"
)
STT_MODEL: str = "base.en"  # Whisper model size (base.en = fast, English-only)
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
