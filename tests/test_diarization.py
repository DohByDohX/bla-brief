"""Unit tests for the diarization module.

VAD, the ECAPA embedder, and clustering are all monkeypatched with fakes so
these tests stay fast, hardware-free, and require none of the optional
diarization dependencies (speechbrain, torch, scikit-learn) to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from meeting_recorder import diarization
from meeting_recorder.diarization import _SpeechSegment, transcribe_with_speakers


@dataclass
class _FakeWord:
    start: float
    end: float
    word: str


@dataclass
class _FakeWhisperSegment:
    words: list[_FakeWord]


class _FakeInfo:
    def __init__(self, language: str = "en", duration: float = 2.0) -> None:
        self.language = language
        self.duration = duration


class _FakeWhisperModel:
    def __init__(self, segments: list[_FakeWhisperSegment]) -> None:
        self._segments = segments

    def transcribe(self, path: str, language: str | None = None, word_timestamps: bool = False):
        assert word_timestamps is True  # diarization always needs word-level timing
        return iter(self._segments), _FakeInfo()


def _patch_pipeline(monkeypatch, segments: list[_SpeechSegment], labels: list[int]) -> None:
    monkeypatch.setattr(diarization, "_register_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(diarization, "_enable_offline", lambda: None)
    # decode_audio is imported lazily inside the function, so patch it where
    # it actually lives rather than on the diarization module.
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda *a, **k: np.zeros(1))
    monkeypatch.setattr(diarization, "_run_vad", lambda audio: segments)
    monkeypatch.setattr(diarization, "_load_embedder", lambda: object())
    monkeypatch.setattr(
        diarization, "_embed_segments", lambda *a, **k: np.zeros((len(segments), 4))
    )
    monkeypatch.setattr(diarization, "_cluster", lambda embeddings: np.array(labels))


def test_transcribe_with_speakers_labels_each_turn(tmp_path: Path, monkeypatch):
    wav = tmp_path / "meeting.wav"
    wav.write_bytes(b"\0")  # only existence is checked before the fake pipeline runs

    segments = [_SpeechSegment(start=0.0, end=1.0), _SpeechSegment(start=1.0, end=2.0)]
    _patch_pipeline(monkeypatch, segments, labels=[0, 1])

    whisper_segments = [
        _FakeWhisperSegment([_FakeWord(0.0, 0.5, "Hello "), _FakeWord(0.5, 1.0, "there.")]),
        _FakeWhisperSegment([_FakeWord(1.0, 1.5, "Hi "), _FakeWord(1.5, 2.0, "back.")]),
    ]
    monkeypatch.setattr(
        diarization, "_load_whisper_model", lambda *a, **k: _FakeWhisperModel(whisper_segments)
    )

    result = transcribe_with_speakers(wav, device="cpu")

    assert result.text == "Speaker 0: Hello there.\nSpeaker 1: Hi back."
    assert result.device == "cpu"
    assert result.language == "en"


def test_transcribe_with_speakers_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        transcribe_with_speakers(Path("does-not-exist.wav"))
