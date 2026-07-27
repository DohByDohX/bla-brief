"""Unit tests for the local transcription module.

The faster-whisper engine is never loaded here: :func:`_load_whisper_model` is
monkeypatched with a fake so the tests stay fast, hardware-free, and require
neither the optional dependency nor a downloaded model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from meeting_recorder import transcription
from meeting_recorder.transcription import (
    TranscriptionResult,
    _candidate_devices,
    transcribe_file,
    write_transcript,
)


@dataclass
class _FakeSegment:
    text: str


class _FakeInfo:
    def __init__(self, language: str | None, duration: float) -> None:
        self.language = language
        self.duration = duration


class _FakeModel:
    """Stand-in for faster-whisper's WhisperModel."""

    def __init__(self, segments: list[str], language: str = "en", duration: float = 1.0) -> None:
        self._segments = segments
        self._language = language
        self._duration = duration
        self.calls: list[tuple[str, str | None]] = []

    def transcribe(self, path: str, language: str | None = None):
        self.calls.append((path, language))
        segs = [_FakeSegment(t) for t in self._segments]
        return iter(segs), _FakeInfo(self._language, self._duration)


# -- _candidate_devices ------------------------------------------------------


def test_candidate_devices_auto_prefers_cuda_then_cpu():
    assert _candidate_devices("auto") == ["cuda", "cpu"]


def test_candidate_devices_explicit_is_used_as_is():
    assert _candidate_devices("cuda") == ["cuda"]
    assert _candidate_devices("cpu") == ["cpu"]


# -- transcribe_file ---------------------------------------------------------


def test_transcribe_joins_segments_and_reports_metadata(tmp_path: Path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF....")  # existence is all transcribe_file checks
    fake = _FakeModel([" Hello", " world."], language="en", duration=12.5)
    monkeypatch.setattr(transcription, "_load_whisper_model", lambda *a, **k: fake)

    result = transcribe_file(wav, model="base.en", device="cpu", language="en")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello world."  # joined, then stripped
    assert result.language == "en"
    assert result.duration == 12.5
    assert result.device == "cpu"
    assert fake.calls == [(str(wav), "en")]


def test_transcribe_auto_falls_back_to_cpu_when_cuda_fails(tmp_path: Path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")
    fake = _FakeModel([" ok"])
    attempted: list[str] = []

    def loader(model_size: str, device: str, compute_type: str):
        attempted.append(device)
        if device == "cuda":
            raise RuntimeError("no cuDNN")
        return fake

    monkeypatch.setattr(transcription, "_load_whisper_model", loader)

    result = transcribe_file(wav, device="auto")

    assert attempted == ["cuda", "cpu"]  # tried GPU first, then fell back
    assert result.device == "cpu"
    assert result.text == "ok"


def test_transcribe_explicit_cuda_failure_raises(tmp_path: Path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")

    def loader(*_a, **_k):
        raise RuntimeError("no cuDNN")

    monkeypatch.setattr(transcription, "_load_whisper_model", loader)

    # Explicit device request must not silently fall back to CPU.
    with pytest.raises(RuntimeError, match="No transcription backend"):
        transcribe_file(wav, device="cuda")


def test_transcribe_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        transcribe_file(tmp_path / "nope.wav")


# -- write_transcript --------------------------------------------------------


def test_write_transcript_writes_body_and_trailing_newline(tmp_path: Path):
    dest = tmp_path / "notes" / "2026-07-27_1030_sync.md"
    out = write_transcript("Line one.\nLine two.", dest)

    assert out == dest
    assert dest.read_text(encoding="utf-8") == "Line one.\nLine two.\n"


def test_write_transcript_is_atomic_no_part_left(tmp_path: Path):
    dest = tmp_path / "2026-07-27_1030_sync.md"
    write_transcript("body", dest)

    # The hidden in-progress file must be gone after a successful write.
    assert not (tmp_path / ".2026-07-27_1030_sync.md.part").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_write_transcript_normalizes_trailing_newlines(tmp_path: Path):
    dest = tmp_path / "t.md"
    write_transcript("text\n\n\n", dest)
    assert dest.read_text(encoding="utf-8") == "text\n"


# -- _register_cuda_dll_dirs -------------------------------------------------


def test_register_cuda_dll_dirs_is_best_effort(monkeypatch):
    # Should never raise, even when the nvidia packages are absent.
    monkeypatch.setattr(transcription.importlib.util, "find_spec", lambda name: None)
    transcription._register_cuda_dll_dirs()


# -- offline-first enforcement -----------------------------------------------


def test_enable_offline_sets_env_flags(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    transcription._enable_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_transcribe_runs_offline_and_never_online(tmp_path: Path, monkeypatch):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF")
    calls = {"offline": 0, "trust": 0}
    monkeypatch.setattr(transcription, "_enable_offline", lambda: calls.__setitem__("offline", 1))
    monkeypatch.setattr(
        transcription, "_inject_system_trust_store", lambda: calls.__setitem__("trust", 1)
    )
    monkeypatch.setattr(transcription, "_load_whisper_model", lambda *a, **k: _FakeModel([" hi"]))

    transcription.transcribe_file(wav, device="cpu")

    assert calls["offline"] == 1  # offline enforced
    assert calls["trust"] == 0  # no online trust-store path during recording


def test_download_model_goes_online_and_loads_on_cpu(monkeypatch):
    calls: dict[str, object] = {"trust": 0}
    monkeypatch.setattr(
        transcription, "_inject_system_trust_store", lambda: calls.__setitem__("trust", 1)
    )

    def fake_load(model_size: str, device: str, compute_type: str):
        calls["loaded"] = (model_size, device, compute_type)
        return _FakeModel([])

    monkeypatch.setattr(transcription, "_load_whisper_model", fake_load)

    transcription.download_model("base.en")

    assert calls["trust"] == 1  # download validates via OS trust store
    assert calls["loaded"] == ("base.en", "cpu", "int8")
