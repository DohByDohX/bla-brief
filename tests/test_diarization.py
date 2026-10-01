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
from meeting_recorder.diarization import (
    _assign_word_labels,
    _render_transcript,
    _SpeechSegment,
    transcribe_with_speakers,
)


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


# -- overlap rendering (word -> label assignment + grouping) -----------------


def test_assign_word_labels_handles_overlap_and_gaps():
    segments = [
        _SpeechSegment(start=0.0, end=1.0, label="Local"),
        _SpeechSegment(start=0.5, end=1.5, label="Speaker 0"),  # overlaps 0.5-1.0
    ]
    words = [
        {"start": 0.0, "end": 0.4, "text": "Hi "},  # Local only
        {"start": 0.6, "end": 0.9, "text": "there "},  # both (overlap)
        {"start": 2.0, "end": 2.4, "text": "gap"},  # neither
    ]

    labeled = _assign_word_labels(words, segments)

    assert labeled[0]["labels"] == ["Local"]
    assert labeled[1]["labels"] == ["Local", "Speaker 0"]
    assert labeled[2]["labels"] == []


def test_render_transcript_overlap_is_separate_lines_same_text():
    words = [
        {"text": "Hi ", "labels": ["Local"]},
        {"text": "there ", "labels": ["Local", "Speaker 0"]},
        {"text": "gap", "labels": []},
    ]

    text = _render_transcript(words)

    assert text == "Local: Hi\nLocal: there\nSpeaker 0: there\nSpeaker None: gap"


# -- channel-aware diarization (mic = Local, only system is clustered) ------


def test_transcribe_with_speakers_channel_aware(tmp_path: Path, monkeypatch):
    mixed = tmp_path / "mixed.wav"
    mic = tmp_path / "mic.wav"
    sys = tmp_path / "sys.wav"
    for p in (mixed, mic, sys):
        p.write_bytes(b"\0")

    monkeypatch.setattr(diarization, "_register_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(diarization, "_enable_offline", lambda: None)

    # Mic ran 1s longer than system -> system started ~1s late (gets padded).
    audio_by_path = {str(mic): np.zeros(32000), str(sys): np.zeros(16000)}
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda path, **k: audio_by_path[path])

    def fake_run_vad(audio: np.ndarray) -> list[_SpeechSegment]:
        if audio.size == 32000:  # mic track
            return [_SpeechSegment(start=0.0, end=1.0)]
        return [_SpeechSegment(start=0.0, end=0.5)]  # sys track, pre-alignment

    monkeypatch.setattr(diarization, "_run_vad", fake_run_vad)
    monkeypatch.setattr(diarization, "_load_embedder", lambda: object())
    monkeypatch.setattr(diarization, "_embed_segments", lambda *a, **k: np.zeros((1, 4)))
    monkeypatch.setattr(diarization, "_cluster", lambda embeddings: np.array([0]))

    whisper_segments = [
        _FakeWhisperSegment([_FakeWord(0.0, 0.5, "Hello "), _FakeWord(1.2, 1.5, "world.")]),
    ]
    monkeypatch.setattr(
        diarization, "_load_whisper_model", lambda *a, **k: _FakeWhisperModel(whisper_segments)
    )

    result = transcribe_with_speakers(mixed, device="cpu", mic_path=mic, sys_path=sys)

    # The system segment (0.0-0.5 raw) is shifted +1.0s to 1.0-1.5 in the
    # mixed timeline, so "world." (at 1.2-1.5) lands on Speaker 0, not Local.
    assert result.text == "Local: Hello\nSpeaker 0: world."
