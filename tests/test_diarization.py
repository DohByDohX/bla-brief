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
from meeting_recorder.config import DIARIZE_MIN_SEGMENT_S
from meeting_recorder.diarization import (
    _assign_word_labels,
    _dedup_consecutive_duplicates,
    _diarize_channel_aware,
    _diarize_single_track,
    _is_near_duplicate,
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


def test_assign_word_labels_uses_overlap_when_midpoint_misses():
    segments = [_SpeechSegment(start=0.0, end=1.0, label="Local")]
    words = [{"start": 0.8, "end": 1.4, "text": "center"}]  # midpoint 1.1, outside

    assert _assign_word_labels(words, segments)[0]["labels"] == ["Local"]


def test_assign_word_labels_prefers_larger_overlap_over_nearer_edge():
    segments = [
        _SpeechSegment(start=0.0, end=1.0, label="Local"),
        _SpeechSegment(start=1.15, end=2.0, label="Speaker 1"),
    ]
    # Overlaps Local by 0.3s and Speaker 1 by 0.05s; midpoint 1.1 is nearer Speaker 1.
    words = [{"start": 0.7, "end": 1.2, "text": "any"}]

    assert _assign_word_labels(words, segments)[0]["labels"] == ["Local"]


def test_assign_word_labels_snaps_word_in_short_gap_to_nearest_segment():
    segments = [
        _SpeechSegment(start=0.0, end=1.0, label="Local"),
        _SpeechSegment(start=2.0, end=3.0, label="Speaker 0"),
    ]
    words = [{"start": 1.1, "end": 1.3, "text": "is"}]  # 0.2s from Local, 0.8s from Speaker 0

    assert _assign_word_labels(words, segments)[0]["labels"] == ["Local"]


def test_assign_word_labels_leaves_far_word_unlabeled():
    segments = [_SpeechSegment(start=0.0, end=1.0, label="Local")]
    words = [{"start": 5.0, "end": 5.4, "text": "Thank you."}]

    assert _assign_word_labels(words, segments)[0]["labels"] == []


def test_assign_word_labels_handles_zero_duration_word_in_gap():
    segments = [_SpeechSegment(start=0.0, end=1.0, label="Local")]
    words = [
        {"start": 1.2, "end": 1.2, "text": "are"},  # within tolerance
        {"start": 1.6, "end": 1.6, "text": "going"},  # beyond tolerance
    ]

    labeled = _assign_word_labels(words, segments)

    assert labeled[0]["labels"] == ["Local"]
    assert labeled[1]["labels"] == []


def test_assign_word_labels_ignores_unlabeled_segments():
    segments = [_SpeechSegment(start=0.0, end=1.0)]
    words = [{"start": 0.2, "end": 0.4, "text": "hi"}]

    assert _assign_word_labels(words, segments)[0]["labels"] == []


def test_render_transcript_overlap_collapses_to_one_line_per_dedup():
    """_assign_word_labels can mark a word as active under two labels at once
    (true overlap), but _render_transcript's dedup pass now collapses the
    resulting duplicate-text lines unconditionally -- see
    _dedup_consecutive_duplicates for why.
    """
    words = [
        {"text": "Hi ", "labels": ["Local"]},
        {"text": "there ", "labels": ["Local", "Speaker 0"]},
        {"text": "gap", "labels": []},
    ]

    text = _render_transcript(words)

    assert text == "Local: Hi\nLocal: there\nSpeaker None: gap"


# -- dedup consecutive identical text (overlap artifact fix) -----------------


def test_dedup_consecutive_duplicates_removes_duplicate_text():
    """Consecutive lines with identical text but different speakers → keep first only."""
    transcript = "Local: decided to like okay\nSpeaker 2: decided to like okay\nLocal: let's wait"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "Local: decided to like okay\nLocal: let's wait"


def test_dedup_consecutive_duplicates_preserves_non_duplicates():
    """Lines with different text or same speaker → unchanged."""
    transcript = "Local: Hello\nSpeaker 2: Hi there\nSpeaker 2: How are you"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == transcript


def test_dedup_consecutive_duplicates_handles_multiple_duplicates():
    """Multiple duplicate groups → all deduplicated."""
    transcript = (
        "Local: No\nSpeaker 2: No\nLocal: no no But that\nSpeaker 0: no no But that\n"
        "Local: was a good call"
    )

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "Local: No\nLocal: no no But that\nLocal: was a good call"


def test_dedup_consecutive_duplicates_empty_or_single_line():
    """Empty or single-line transcripts → unchanged."""
    assert _dedup_consecutive_duplicates("") == ""
    assert _dedup_consecutive_duplicates("Local: Hello") == "Local: Hello"


def test_dedup_consecutive_duplicates_handles_chain_of_three_echoes():
    """Three speakers echoing the same text before the original resumes → all dropped."""
    transcript = "A: X\nB: X\nC: X\nA: Y"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "A: X\nA: Y"


def test_dedup_consecutive_duplicates_drops_echo_even_without_resumption():
    """An echo is dropped even if the *other* (not original) speaker continues --
    two speakers independently saying the same phrase at once isn't a plausible
    coincidence, so it's treated as a bleed-through artifact regardless of who
    continues or how short the echo is.
    """
    transcript = "A: X\nB: X\nB: Y"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "A: X\nB: Y"


def test_dedup_consecutive_duplicates_drops_multiword_echo_even_without_resumption():
    transcript = (
        "Local: No worries, I just got\n"
        "Speaker 0: No worries, I just got\n"
        "Speaker 0: a meeting. So,"
    )

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "Local: No worries, I just got\nSpeaker 0: a meeting. So,"


def test_dedup_consecutive_duplicates_drops_short_echo_without_resumption():
    """Even a short (1-2 word) echo is now dropped unconditionally."""
    transcript = "Local: so get\nSpeaker 0: so get\nSpeaker 0: your rest man"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "Local: so get\nSpeaker 0: your rest man"


def test_is_near_duplicate_matches_minor_punctuation_differences():
    """Trailing punctuation noise shouldn't block a duplicate match."""
    assert _is_near_duplicate("a bit", "a bit.")
    assert _is_near_duplicate("check-in, check", "check-in, check")


def test_is_near_duplicate_rejects_genuinely_different_text():
    """Short shared prefix with substantially different content isn't a duplicate."""
    assert not _is_near_duplicate("load", "load those bots")
    assert not _is_near_duplicate("Hello", "Hi there")


def test_dedup_consecutive_duplicates_handles_near_duplicate_punctuation():
    """Near-duplicate text (punctuation-only diff) under a different speaker is dropped."""
    transcript = "Local: a bit.\nSpeaker 5: a bit.\nLocal: And then, what is repacking?"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "Local: a bit.\nLocal: And then, what is repacking?"


def test_dedup_consecutive_duplicates_handles_chain_of_near_duplicates():
    """A chain where each echo has slightly different trailing punctuation still collapses."""
    transcript = "A: a bit.\nB: a bit,\nC: a bit\nA: moving on now"

    result = _dedup_consecutive_duplicates(transcript)

    assert result == "A: a bit.\nA: moving on now"


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


# -- min-segment filtering (mic channel keeps short slivers, system doesn't) -


def test_diarize_single_track_drops_segments_shorter_than_min_segment(monkeypatch):
    """System-channel path still filters slivers too short to embed reliably."""
    short = _SpeechSegment(start=0.0, end=DIARIZE_MIN_SEGMENT_S / 2)
    long = _SpeechSegment(start=1.0, end=1.0 + DIARIZE_MIN_SEGMENT_S)
    monkeypatch.setattr(diarization, "_run_vad", lambda audio: [short, long])
    monkeypatch.setattr(diarization, "_load_embedder", lambda: object())
    monkeypatch.setattr(diarization, "_embed_segments", lambda *a, **k: np.zeros((1, 4)))
    monkeypatch.setattr(diarization, "_cluster", lambda embeddings: np.array([0]))

    segments = _diarize_single_track(np.zeros(1))

    assert segments == [long]
    assert segments[0].label == "Speaker 0"


def test_diarize_channel_aware_keeps_short_mic_segments(tmp_path: Path, monkeypatch):
    """Mic channel is labeled directly, so it keeps slivers the system channel
    would drop -- fixes short acknowledgments ("Okay", "Yeah") rendering as
    Speaker None.
    """
    mic = tmp_path / "mic.wav"
    sys = tmp_path / "sys.wav"
    for p in (mic, sys):
        p.write_bytes(b"\0")

    audio_by_path = {str(mic): np.zeros(16000), str(sys): np.zeros(16000)}
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda path, **k: audio_by_path[path])

    short_mic_segment = _SpeechSegment(start=0.0, end=DIARIZE_MIN_SEGMENT_S / 2)

    def fake_run_vad(audio: np.ndarray) -> list[_SpeechSegment]:
        return [short_mic_segment]  # same fake segment for both tracks here

    monkeypatch.setattr(diarization, "_run_vad", fake_run_vad)
    monkeypatch.setattr(diarization, "_load_embedder", lambda: object())
    monkeypatch.setattr(diarization, "_embed_segments", lambda *a, **k: np.zeros((1, 4)))
    monkeypatch.setattr(diarization, "_cluster", lambda embeddings: np.array([0]))

    segments = _diarize_channel_aware(mic, sys)

    mic_labels = [s.label for s in segments if s.label == "Local"]
    assert mic_labels == ["Local"]  # short mic segment survived and was labeled
    sys_labels = [s.label for s in segments if s.label != "Local"]
    assert sys_labels == []  # equally short system segment was still dropped
