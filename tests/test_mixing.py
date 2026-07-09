"""Unit tests for the hardware-free mixdown logic in meeting_recorder.mixing."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from support import read_wav, write_sine_wav

from meeting_recorder.mixing import create_mixed_file


def _duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def test_equal_length_same_rate_mix(tmp_audio_dir: Path) -> None:
    mic = tmp_audio_dir / "mic.wav"
    sys = tmp_audio_dir / "sys.wav"
    out = tmp_audio_dir / "mixed.wav"
    write_sine_wav(mic, duration_s=3.0, rate=48000, freq=440)
    write_sine_wav(sys, duration_s=3.0, rate=48000, freq=880)

    assert create_mixed_file(mic, sys, out, sync_offset="auto") is True

    samples, rate = read_wav(out)
    assert rate == 48000
    # Same length in, same length out (within a frame).
    assert abs(len(samples) - 48000 * 3) <= 1


def test_normalized_to_target_peak(tmp_audio_dir: Path) -> None:
    """Two hot tracks that would clip when summed must be normalized to ~0.95."""
    mic = tmp_audio_dir / "mic.wav"
    sys = tmp_audio_dir / "sys.wav"
    out = tmp_audio_dir / "mixed.wav"
    write_sine_wav(mic, duration_s=1.0, rate=48000, freq=440, amplitude=0.9)
    write_sine_wav(sys, duration_s=1.0, rate=48000, freq=441, amplitude=0.9)

    assert create_mixed_file(mic, sys, out, sync_offset="auto") is True

    samples, _ = read_wav(out)
    peak = float(np.max(np.abs(samples)))
    assert 0.93 <= peak <= 0.96, f"peak={peak}"


def test_auto_alignment_end_anchored(tmp_audio_dir: Path) -> None:
    """A shorter system track (late start) is front-padded so both end together."""
    mic = tmp_audio_dir / "mic.wav"
    sys = tmp_audio_dir / "sys.wav"
    out = tmp_audio_dir / "mixed.wav"
    write_sine_wav(mic, duration_s=5.0, rate=48000, freq=440)
    write_sine_wav(sys, duration_s=3.0, rate=48000, freq=880)  # started ~2s late

    assert create_mixed_file(mic, sys, out, sync_offset="auto") is True

    # Output spans the longer (mic) track; the system audio sits at the END.
    out_samples, rate = read_wav(out)
    assert abs(len(out_samples) - rate * 5) <= 1
    # First second is mic-only (system padded with silence up front); the last
    # second contains both, so its energy is higher than the lead-in.
    lead = out_samples[:rate]
    tail = out_samples[-rate:]
    assert float(np.sqrt(np.mean(tail**2))) > float(np.sqrt(np.mean(lead**2)))


def test_resample_mic_to_system_rate(tmp_audio_dir: Path) -> None:
    """A 16kHz mic (e.g. AirPods) is resampled to the 48kHz system rate."""
    mic = tmp_audio_dir / "mic.wav"
    sys = tmp_audio_dir / "sys.wav"
    out = tmp_audio_dir / "mixed.wav"
    write_sine_wav(mic, duration_s=2.0, rate=16000, freq=300)
    write_sine_wav(sys, duration_s=2.0, rate=48000, freq=600)

    assert create_mixed_file(mic, sys, out, sync_offset="auto") is True

    out_samples, rate = read_wav(out)
    assert rate == 48000  # output takes the system rate
    assert abs(_duration_s(out) - 2.0) < 0.05  # duration preserved (not pitch-shifted)


def test_empty_track_returns_false(tmp_audio_dir: Path) -> None:
    mic = tmp_audio_dir / "mic.wav"
    sys = tmp_audio_dir / "sys.wav"
    out = tmp_audio_dir / "mixed.wav"
    write_sine_wav(mic, duration_s=0.0, rate=48000)  # empty
    write_sine_wav(sys, duration_s=1.0, rate=48000)

    assert create_mixed_file(mic, sys, out, sync_offset="auto") is False
    assert not out.exists()


def test_missing_file_returns_false(tmp_audio_dir: Path) -> None:
    out = tmp_audio_dir / "mixed.wav"
    assert (
        create_mixed_file(
            tmp_audio_dir / "nope-mic.wav",
            tmp_audio_dir / "nope-sys.wav",
            out,
        )
        is False
    )
