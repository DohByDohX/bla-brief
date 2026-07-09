"""Importable test helpers (not a conftest, to avoid name-collision issues).

These are plain functions/classes imported directly by the unit tests. They
live here rather than in ``conftest.py`` because a bare ``import conftest`` can
bind to the wrong module once multiple ``conftest.py`` files exist in the tree.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_sine_wav(
    path: Path,
    duration_s: float,
    rate: int,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> int:
    """Write a mono int16 sine-tone WAV and return the number of frames.

    Used to build deterministic, hardware-free inputs for the mixer tests.
    """
    nframes = int(round(duration_s * rate))
    t = np.arange(nframes, dtype=np.float64) / rate
    samples = (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return nframes


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono int16 WAV as float32 in [-1, 1) plus its sample rate."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


class FakePyAudio:
    """Minimal PyAudio stand-in for testing device-selection logic.

    Only implements the read-only enumeration surface that
    :mod:`meeting_recorder.devices` relies on.
    """

    def __init__(self, devices: list[dict], host_apis: list[dict]) -> None:
        self._devices = devices
        self._host_apis = host_apis

    def get_host_api_count(self) -> int:
        return len(self._host_apis)

    def get_host_api_info_by_index(self, i: int) -> dict:
        return self._host_apis[i]

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, i: int) -> dict:
        return self._devices[i]
