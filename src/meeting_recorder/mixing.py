"""Post-recording mixdown: combine mic + system tracks into a normalized file.

The mix is streamed in fixed-size chunks (two passes over the on-disk tracks)
so memory stays bounded even for multi-hour recordings, and differing sample
rates are handled by resampling the mic track to the system rate.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from meeting_recorder.config import ALIGN_THRESHOLD_S, MIX_CHUNK_FRAMES, NORM_TARGET

log = logging.getLogger(__name__)


def _read_mono_float(wav: wave.Wave_read, nframes: int) -> np.ndarray:
    """Read up to ``nframes`` frames from a mono int16 WAV as float32 in [-1, 1)."""
    raw = wav.readframes(nframes)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _mic_resampled_chunk(
    mic_wav: wave.Wave_read,
    mic_total: int,
    mic_rate: int,
    out_rate: int,
    j0: int,
    length: int,
) -> np.ndarray:
    """Return ``length`` output samples (at out_rate) of the mic track.

    Starts at output index ``j0`` and resamples from ``mic_rate`` via linear
    interpolation. Absolute input/output coordinates are used so successive
    chunks join seamlessly (no per-chunk boundary discontinuity). Returns
    fewer samples when the mic track ends.
    """
    ratio = mic_rate / out_rate
    pos = np.arange(j0, j0 + length, dtype=np.float64) * ratio
    i0 = int(np.floor(pos[0]))
    if i0 >= mic_total:
        return np.empty(0, dtype=np.float32)
    # Read one extra input frame so the last output sample has both neighbors.
    i1 = min(int(np.floor(pos[-1])) + 2, mic_total)
    mic_wav.setpos(i0)
    src = _read_mono_float(mic_wav, i1 - i0)
    if src.size == 0:
        return np.empty(0, dtype=np.float32)
    xp = np.arange(i0, i0 + src.size, dtype=np.float64)
    # Only emit outputs whose interpolation neighbors actually exist in src,
    # so we never fabricate audio past the end of the mic track.
    valid = pos <= xp[-1]
    return np.interp(pos[valid], xp, src).astype(np.float32)


def create_mixed_file(
    mic_path: Path,
    sys_path: Path,
    mixed_path: Path,
    mic_gain: float = 1.0,
    sys_gain: float = 1.0,
    sync_offset: str | float = "auto",
    report_path: Path | None = None,
) -> bool:
    """Create a normalized mixed WAV from the mic + system tracks.

    Streams the mix in chunks (two passes over the on-disk tracks) so memory
    stays bounded even for multi-hour recordings. Handles differing sample
    rates by resampling the mic track to the system rate (e.g., AirPods 16kHz).

    Alignment (``sync_offset``): the two streams start and stop together, but
    one (usually the system loopback) can begin delivering audio a few seconds
    late, leaving the tracks different lengths. Aligning naively from frame 0
    then drifts them apart. ``sync_offset`` is the number of seconds the SYSTEM
    track started after the MIC track (positive = system late). The later-
    starting track is padded with leading silence so both END together.
      - ``"auto"``: infer the offset from the length difference (assumes the
        two streams stopped together, which the recorder guarantees).
      - float: use a measured offset (e.g., from callback timestamps).
      - ``0.0``: legacy frame-0 alignment (no correction).

    ``report_path`` is the name to show in the "Output files" summary when the
    mix is written to a temp file and renamed by the caller (so the log shows
    the published name, not the ``.part`` temp).

    Returns:
        True on success, False if the tracks are missing/empty/unreadable.
    """
    try:
        mic_wav = wave.open(str(mic_path), "rb")
        sys_wav = wave.open(str(sys_path), "rb")
    except Exception as e:
        log.error("Cannot open track files: %s", e)
        return False

    try:
        mic_rate = mic_wav.getframerate()
        sys_rate = sys_wav.getframerate()
        mic_total = mic_wav.getnframes()
        sys_total = sys_wav.getnframes()

        if mic_rate <= 0 or sys_rate <= 0:
            log.error("Invalid sample rate (mic=%d, sys=%d).", mic_rate, sys_rate)
            return False
        if mic_total == 0 or sys_total == 0:
            log.warning("One or both tracks are empty!")
            return False

        out_rate = sys_rate  # output at system rate
        mic_dur = mic_total / mic_rate
        sys_dur = sys_total / sys_rate
        log.info("Mic: %.1fs @ %dHz | Sys: %.1fs @ %dHz", mic_dur, mic_rate, sys_dur, sys_rate)

        need_resample = mic_rate != sys_rate
        if need_resample:
            log.info("Resampling mic %dHz -> %dHz...", mic_rate, sys_rate)

        # Length of each track expressed in output-rate frames.
        mic_out_len = (
            mic_total if not need_resample else int(round(mic_total * out_rate / mic_rate))
        )
        sys_out_len = sys_total

        # Resolve alignment offset (seconds system started after mic).
        if sync_offset == "auto":
            offset_sec = (mic_out_len - sys_out_len) / out_rate
        else:
            offset_sec = float(sync_offset)

        lead = int(round(offset_sec * out_rate))
        if lead >= 0:
            sys_lead, mic_lead = lead, 0  # system late -> pad its front
        else:
            sys_lead, mic_lead = 0, -lead  # mic late -> pad its front

        total_out = max(mic_lead + mic_out_len, sys_lead + sys_out_len)
        if abs(offset_sec) > ALIGN_THRESHOLD_S:
            late = "system" if lead >= 0 else "mic"
            log.info(
                "Aligning: %s track started ~%.2fs late -> padding its front (end-anchored).",
                late,
                abs(offset_sec),
            )

        def mic_out_frames(start: int, count: int) -> np.ndarray:
            """`count` mic output-rate samples starting at output index `start`."""
            if start >= mic_out_len or count <= 0:
                return np.empty(0, dtype=np.float32)
            if need_resample:
                return _mic_resampled_chunk(mic_wav, mic_total, mic_rate, out_rate, start, count)
            mic_wav.setpos(start)
            return _read_mono_float(mic_wav, min(count, mic_total - start))

        def sys_out_frames(start: int, count: int) -> np.ndarray:
            """`count` system output-rate samples starting at output index `start`."""
            if start >= sys_out_len or count <= 0:
                return np.empty(0, dtype=np.float32)
            sys_wav.setpos(start)
            return _read_mono_float(sys_wav, min(count, sys_total - start))

        def iter_mixed():
            """Yield successive mixed float chunks over the full aligned timeline."""
            for k in range(0, total_out, MIX_CHUNK_FRAMES):
                length = min(MIX_CHUNK_FRAMES, total_out - k)
                out = np.zeros(length, dtype=np.float32)
                # Mic contributes to output indices [mic_lead, mic_lead+mic_out_len).
                lo = max(k, mic_lead)
                hi = min(k + length, mic_lead + mic_out_len)
                if hi > lo:
                    seg = mic_out_frames(lo - mic_lead, hi - lo)
                    if seg.size:
                        out[lo - k : lo - k + seg.size] += seg * mic_gain
                # System contributes to [sys_lead, sys_lead+sys_out_len).
                lo = max(k, sys_lead)
                hi = min(k + length, sys_lead + sys_out_len)
                if hi > lo:
                    seg = sys_out_frames(lo - sys_lead, hi - lo)
                    if seg.size:
                        out[lo - k : lo - k + seg.size] += seg * sys_gain
                yield out

        # Pass 1: find the peak for normalization.
        peak = 0.0
        total_frames = 0
        for chunk in iter_mixed():
            total_frames += chunk.size
            if chunk.size:
                peak = max(peak, float(np.max(np.abs(chunk))))

        if total_frames == 0:
            log.warning("No overlapping audio to mix!")
            return False

        scale = (NORM_TARGET / peak) if peak > 0 else 1.0

        # Pass 2: write the normalized mix.
        log.info("Writing mixed file...")
        out_wav = wave.open(str(mixed_path), "wb")
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)
        out_wav.setframerate(out_rate)
        try:
            for chunk in iter_mixed():
                pcm = (chunk * scale * 32767).clip(-32768, 32767).astype(np.int16)
                out_wav.writeframes(pcm.tobytes())
        finally:
            out_wav.close()

        duration = total_frames / out_rate
    finally:
        mic_wav.close()
        sys_wav.close()

    size_mb = mixed_path.stat().st_size / (1024 * 1024)
    mic_mb = mic_path.stat().st_size / (1024 * 1024)
    sys_mb = sys_path.stat().st_size / (1024 * 1024)

    # The mix is written to a temp ".part" file then renamed by the caller;
    # show the caller-supplied final name (report_path) when given.
    display_name = (report_path or mixed_path).name
    log.info("Output files:")
    log.info("  %-50s %6.1f MB  (mixed - for transcription)", display_name, size_mb)
    log.info("  %-50s %6.1f MB  (your voice)", mic_path.name, mic_mb)
    log.info("  %-50s %6.1f MB  (system/meeting audio)", sys_path.name, sys_mb)
    log.info("Duration: %dm %ds", int(duration // 60), int(duration % 60))
    log.info("Location: %s", mixed_path.parent)
    return True
