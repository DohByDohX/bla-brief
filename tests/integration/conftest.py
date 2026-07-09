"""Integration-only fixtures: keep real system audio flowing during capture.

The recorder subprocess records the WASAPI **loopback** of the default output
device. On a silent machine that track is empty, which makes the recording
tests environment-dependent (they fail if nothing happens to be playing).

This fixture plays a quiet, continuous tone on the default output device for
the whole integration session, so the loopback always captures real audio and
the recording tests become deterministic. It runs in the pytest process while
the recorder runs in a subprocess; both share the same render endpoint, so the
subprocess's loopback picks up the tone.

If the output stream can't be opened (unusual hardware/CI), the fixture degrades
gracefully — tests then behave as before rather than erroring.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import pyaudiowpatch as pyaudio
except Exception:  # pragma: no cover - PyAudio should be installed
    pyaudio = None


class _TonePlayer:
    """Plays a continuous low-amplitude sine tone on the default WASAPI output."""

    def __init__(self, freq: float = 440.0, amplitude: float = 0.03) -> None:
        self.freq = freq
        self.amplitude = amplitude
        self.rate = 48000
        self._n = 0  # running sample index, for click-free phase continuity
        self._pa = None
        self._stream = None

    def start(self) -> bool:
        if pyaudio is None:
            return False
        try:
            self._pa = pyaudio.PyAudio()
            out_index = self._default_wasapi_output()
            if out_index is not None:
                info = self._pa.get_device_info_by_index(out_index)
                self.rate = int(info["defaultSampleRate"])
            two_pi_f = 2 * np.pi * self.freq

            def _cb(in_data, frame_count, time_info, status):
                t = (self._n + np.arange(frame_count, dtype=np.float64)) / self.rate
                self._n += frame_count
                mono = (np.sin(two_pi_f * t) * self.amplitude).astype(np.float32)
                stereo = np.repeat(mono, 2)  # interleaved L=R
                return (stereo.tobytes(), pyaudio.paContinue)

            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=self.rate,
                output=True,
                output_device_index=out_index,
                frames_per_buffer=1024,
                stream_callback=_cb,
            )
            self._stream.start_stream()
            return True
        except Exception:
            self.stop()
            return False

    def _default_wasapi_output(self) -> int | None:
        for i in range(self._pa.get_host_api_count()):
            api = self._pa.get_host_api_info_by_index(i)
            if "WASAPI" in api["name"]:
                d = api.get("defaultOutputDevice", -1)
                return int(d) if d >= 0 else None
        return None

    def stop(self) -> None:
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass
        self._stream = None
        self._pa = None


@pytest.fixture(scope="session", autouse=True)
def _system_audio_tone():
    """Ensure the system-audio loopback has real content for the whole session."""
    player = _TonePlayer()
    player.start()
    yield
    player.stop()
