"""Synchronized dual-source streaming recorder (single PyAudioWPatch backend).

Both the mic and system-loopback streams use the same audio engine and are
started together, eliminating the multi-second sync offset a dual-backend
design would suffer. Audio is written straight to disk in the callbacks, so
memory stays constant regardless of recording length.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pyaudiowpatch as pyaudio

from meeting_recorder.config import CHUNK_SIZE, SAMPLE_RATE

log = logging.getLogger(__name__)


class StreamingDualRecorder:
    """Record mic + system audio via a single PyAudioWPatch backend."""

    def __init__(
        self,
        mic_path: Path,
        sys_path: Path,
        mic_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.mic_path = mic_path
        self.sys_path = sys_path
        self.sample_rate = sample_rate
        self.mic_index = mic_index

        # Cross-thread stop signal (set by ENTER thread, Ctrl+C handler, etc.).
        self._started = False
        self.stop_event = threading.Event()

        self._pa: pyaudio.PyAudio | None = None
        self._mic_info: dict | None = None
        self._loopback_info: dict | None = None
        self._output_info: dict | None = None

        self._mic_stream: Any = None
        self._sys_stream: Any = None
        self._silence_stream: Any = None

        self._mic_wav: wave.Wave_write | None = None
        self._sys_wav: wave.Wave_write | None = None
        self._mic_lock = threading.Lock()
        self._sys_lock = threading.Lock()
        self.mic_samples = 0
        self.sys_samples = 0

        # Mic sample rate may differ from system rate (e.g., AirPods = 16kHz).
        self._mic_native_rate: int | None = None
        self._mic_channels: int | None = None
        self._sys_native_rate: int | None = None

        # Wall-clock time (perf_counter) of each stream's first delivered
        # callback. The system loopback can begin delivering audio a few
        # seconds after the mic; capturing these lets the mixer align the
        # tracks instead of assuming they both start at frame 0.
        self._mic_t0: float | None = None
        self._sys_t0: float | None = None
        # Seconds the system track started after the mic (positive = sys late).
        self.sync_offset = 0.0

    # -- Control-flow helpers ------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """True while a started recording has not been asked to stop."""
        return self._started and not self.stop_event.is_set()

    def request_stop(self) -> None:
        """Signal the recording to stop (safe to call from any thread)."""
        self.stop_event.set()

    # -- Keepalive -----------------------------------------------------------

    def _start_silence_keepalive(self) -> None:
        """Play inaudible silence on the output so WASAPI loopback stays active."""
        if not self._output_info or self._pa is None:
            return
        try:
            keepalive_rate = int(self._output_info.get("defaultSampleRate", self.sample_rate))
            # Preallocate one stereo-int16 silence buffer and reuse it across
            # callbacks instead of allocating a fresh zero buffer every time.
            silence = bytes(CHUNK_SIZE * 2 * 2)

            def _silence_cb(in_data, frame_count, time_info, status):
                need = frame_count * 2 * 2
                data = silence if need == len(silence) else bytes(need)
                return (data, pyaudio.paContinue)

            self._silence_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=keepalive_rate,
                output=True,
                output_device_index=int(self._output_info["index"]),
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=_silence_cb,
            )
            self._silence_stream.start_stream()
        except Exception:
            log.warning("Keepalive stream failed to start", exc_info=True)

    def _stop_silence_keepalive(self) -> None:
        if self._silence_stream:
            try:
                self._silence_stream.stop_stream()
                self._silence_stream.close()
            except Exception:
                log.debug("Error closing keepalive stream", exc_info=True)

    # -- Capture callbacks ---------------------------------------------------

    def _make_mic_callback(self):
        """Create the mic callback that writes mono frames directly to WAV."""
        mic_ch = self._mic_channels
        assert mic_ch is not None
        mic_lock = self._mic_lock

        def callback(in_data, frame_count, time_info, status):
            if self._mic_t0 is None:
                self._mic_t0 = time.perf_counter()
            samples = np.frombuffer(in_data, dtype=np.int16)
            if mic_ch == 1:
                mono = samples
            else:
                # Drop a trailing partial frame if the buffer isn't a whole
                # number of frames (can happen on an xrun/glitch) — otherwise
                # reshape would raise inside the callback and kill the stream.
                usable = samples.size - (samples.size % mic_ch)
                if usable <= 0:
                    return (None, pyaudio.paContinue)
                samples = samples[:usable].reshape(-1, mic_ch)
                mono = (
                    (samples[:, 0].astype(np.int32) + samples[:, 1].astype(np.int32)) // 2
                ).astype(np.int16)
            with mic_lock:
                if self._mic_wav is not None:
                    self._mic_wav.writeframes(mono.tobytes())
                    self.mic_samples += len(mono)
            return (None, pyaudio.paContinue)

        return callback

    def _make_sys_callback(self):
        """Create the system-audio callback that writes mono frames to WAV."""
        assert self._loopback_info is not None
        sys_ch = self._loopback_info["maxInputChannels"]
        sys_lock = self._sys_lock

        def callback(in_data, frame_count, time_info, status):
            if self._sys_t0 is None:
                self._sys_t0 = time.perf_counter()
            samples = np.frombuffer(in_data, dtype=np.int16)
            if sys_ch == 1:
                mono = samples
            else:
                # Drop a trailing partial frame (see mic callback) to avoid a
                # reshape error crashing the loopback stream.
                usable = samples.size - (samples.size % sys_ch)
                if usable <= 0:
                    return (None, pyaudio.paContinue)
                samples = samples[:usable].reshape(-1, sys_ch)
                mono = (
                    (samples[:, 0].astype(np.int32) + samples[:, 1].astype(np.int32)) // 2
                ).astype(np.int16)
            with sys_lock:
                if self._sys_wav is not None:
                    self._sys_wav.writeframes(mono.tobytes())
                    self.sys_samples += len(mono)
            return (None, pyaudio.paContinue)

        return callback

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open streams and begin synchronized recording."""
        from meeting_recorder.devices import find_devices

        self._pa = pyaudio.PyAudio()
        self._mic_info, self._loopback_info, self._output_info = find_devices(
            self._pa, self.mic_index
        )

        if not self._mic_info:
            raise RuntimeError("No microphone device found.")
        if not self._loopback_info:
            raise RuntimeError("No WASAPI loopback device found.")

        # Determine mic native rate and channels.
        self._mic_native_rate = int(self._mic_info["defaultSampleRate"])
        self._mic_channels = self._mic_info["maxInputChannels"]
        if self._mic_channels < 1:
            self._mic_channels = 1

        # Capture each source at its device's native sample rate so neither
        # track is pitch-shifted / wrong-speed. The loopback device may not
        # run at 48kHz (e.g., a 44.1kHz output), so use its reported rate
        # instead of blindly forcing SAMPLE_RATE.
        mic_rate = self._mic_native_rate
        self._sys_native_rate = int(self._loopback_info["defaultSampleRate"])
        sys_rate = self._sys_native_rate

        log.info("Mic device:    [%d] %s", int(self._mic_info["index"]), self._mic_info["name"])
        log.info(
            "System device: [%d] %s",
            int(self._loopback_info["index"]),
            self._loopback_info["name"],
        )
        if self._output_info:
            log.info(
                "Keepalive on:  [%d] %s",
                int(self._output_info["index"]),
                self._output_info["name"],
            )
        log.info("Mic rate:      %d Hz (%dch)", mic_rate, self._mic_channels)
        log.info("System rate:   %d Hz", sys_rate)
        log.info("Mic file:      %s", self.mic_path.name)
        log.info("System file:   %s", self.sys_path.name)

        # Open WAV files.
        self.mic_path.parent.mkdir(parents=True, exist_ok=True)

        self._mic_wav = wave.open(str(self.mic_path), "wb")
        self._mic_wav.setnchannels(1)
        self._mic_wav.setsampwidth(2)
        self._mic_wav.setframerate(mic_rate)

        self._sys_wav = wave.open(str(self.sys_path), "wb")
        self._sys_wav.setnchannels(1)
        self._sys_wav.setsampwidth(2)
        self._sys_wav.setframerate(sys_rate)

        self.stop_event.clear()
        self._started = True

        # 1. Start silence keepalive (so loopback has data).
        self._start_silence_keepalive()
        time.sleep(0.2)

        # 2. Open BOTH input streams with callbacks (non-blocking, synchronized).
        self._mic_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._mic_channels,
            rate=mic_rate,
            input=True,
            input_device_index=int(self._mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._make_mic_callback(),
        )
        self._sys_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._loopback_info["maxInputChannels"],
            rate=sys_rate,
            input=True,
            input_device_index=int(self._loopback_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._make_sys_callback(),
        )

        # 3. Start both streams as close together as possible.
        self._mic_stream.start_stream()
        self._sys_stream.start_stream()

    def stop(self) -> tuple[int, int]:
        """Stop recording and close all resources safely."""
        self.stop_event.set()
        self._started = False

        # Measure how much later the system stream began delivering audio than
        # the mic (or vice versa). Used by the mixer to align the tracks.
        if self._mic_t0 is not None and self._sys_t0 is not None:
            self.sync_offset = self._sys_t0 - self._mic_t0

        # Stop input streams (callbacks will stop firing).
        for stream in (self._mic_stream, self._sys_stream):
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    log.debug("Error closing input stream", exc_info=True)

        self._stop_silence_keepalive()

        # Close WAV files (finalizes headers).
        with self._mic_lock:
            if self._mic_wav:
                try:
                    self._mic_wav.close()
                except Exception:
                    log.debug("Error closing mic WAV", exc_info=True)
                self._mic_wav = None

        with self._sys_lock:
            if self._sys_wav:
                try:
                    self._sys_wav.close()
                except Exception:
                    log.debug("Error closing system WAV", exc_info=True)
                self._sys_wav = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                log.debug("Error terminating PyAudio", exc_info=True)

        return self.mic_samples, self.sys_samples
