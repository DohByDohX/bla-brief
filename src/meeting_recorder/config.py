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
