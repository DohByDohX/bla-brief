"""Unit tests for CLI filename sanitization and path construction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from support import write_sine_wav

from meeting_recorder import transcription
from meeting_recorder.cli import (
    ALL_OUTPUTS,
    _parse_args,
    _produce_outputs,
    _rename_recording,
    _transcribe_recording,
    build_paths,
    parse_output_choice,
    sanitize_name,
)


def test_sanitize_strips_path_traversal():
    safe = sanitize_name("../../etc/passwd")
    assert "/" not in safe
    assert "\\" not in safe
    assert ".." not in safe


def test_sanitize_empty_falls_back_to_recording():
    assert sanitize_name("////") == "recording"
    assert sanitize_name("") == "recording"


def test_sanitize_keeps_safe_characters():
    assert sanitize_name("weekly-sync.v2") == "weekly-sync.v2"


def test_build_paths_layout_named():
    out = Path(r"C:\watch\Recordings")
    paths = build_paths("demo", out, timestamp="2026-07-07_0900")

    # Mixed final lands in the watch folder; tracks live in a sibling folder.
    assert paths.mixed_final == out / "2026-07-07_0900_demo.wav"
    assert paths.work_dir == out.parent / "Recordings - tracks"
    assert paths.mic_path == paths.work_dir / "2026-07-07_0900_demo_mic-only.wav"
    assert paths.sys_path == paths.work_dir / "2026-07-07_0900_demo_system-only.wav"

    # In-progress mix uses a hidden .part name so *.wav watchers ignore it.
    assert paths.mixed_tmp.name.endswith(".wav.part")
    assert paths.mixed_tmp.name.startswith(".")


def test_build_paths_default_timestamp_only_name():
    out = Path(r"C:\watch\Recordings")
    paths = build_paths(None, out, timestamp="2026-07-07_0900")
    assert paths.mixed_final.stem == "2026-07-07_0900"
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}$", paths.mixed_final.stem)


def test_build_paths_custom_tracks_dir():
    out = Path(r"C:\watch\Recordings")
    custom = Path(r"D:\raw-tracks")
    paths = build_paths("demo", out, tracks_dir=custom, timestamp="2026-07-07_0900")
    assert paths.work_dir == custom
    assert paths.mic_path.parent == custom


def test_parse_output_choice_empty_keeps_all():
    assert parse_output_choice("") == set(ALL_OUTPUTS)
    assert parse_output_choice("   ") == set(ALL_OUTPUTS)


def test_parse_output_choice_letters():
    assert parse_output_choice("m") == {"mixed"}
    assert parse_output_choice("v") == {"mic"}
    assert parse_output_choice("s") == {"system"}


def test_parse_output_choice_multi_and_words():
    assert parse_output_choice("m, v") == {"mixed", "mic"}
    assert parse_output_choice("voice system") == {"mic", "system"}
    assert parse_output_choice("mixed,mic,system") == set(ALL_OUTPUTS)


def test_parse_output_choice_unknown_falls_back_to_all():
    assert parse_output_choice("xyz") == set(ALL_OUTPUTS)


def test_rename_recording_renames_tracks_and_returns_named_paths(tmp_path: Path):
    out = tmp_path / "Recordings"
    out.mkdir()
    ts = "2026-07-09_1204"
    original = build_paths(None, out, timestamp=ts)  # timestamp-only names
    original.mic_path.parent.mkdir(parents=True, exist_ok=True)
    original.mic_path.write_bytes(b"mic")
    original.sys_path.write_bytes(b"sys")

    renamed = _rename_recording(original, "team-sync", out, None, ts)

    # New paths embed the name; old files were moved to the new names.
    assert renamed.mic_path.name == f"{ts}_team-sync_mic-only.wav"
    assert renamed.sys_path.name == f"{ts}_team-sync_system-only.wav"
    assert renamed.mixed_final.name == f"{ts}_team-sync.wav"
    assert renamed.mic_path.exists() and renamed.sys_path.exists()
    assert not original.mic_path.exists() and not original.sys_path.exists()


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {"mic_gain": 1.0, "sys_gain": 1.0, "discard_tracks": False}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _paths_with_tracks(tmp_path: Path, *, mic_s: float, sys_s: float):
    """Build a recording layout with mic/sys tracks of the given durations."""
    out = tmp_path / "Recordings"
    out.mkdir()
    paths = build_paths("demo", out, timestamp="2026-07-09_1204")
    write_sine_wav(paths.mic_path, mic_s, 16000)
    write_sine_wav(paths.sys_path, sys_s, 48000)
    return paths


def test_produce_outputs_mixed_only_prunes_raw_tracks(tmp_path: Path):
    paths = _paths_with_tracks(tmp_path, mic_s=0.5, sys_s=0.5)
    _produce_outputs(paths, _args(), keep={"mixed"})

    # Only the published mixed file survives; the raw tracks are removed.
    assert paths.mixed_final.exists()
    assert not paths.mic_path.exists()
    assert not paths.sys_path.exists()
    assert not paths.mixed_tmp.exists()  # no stray .part left behind


def test_produce_outputs_keep_all_retains_every_file(tmp_path: Path):
    paths = _paths_with_tracks(tmp_path, mic_s=0.5, sys_s=0.5)
    _produce_outputs(paths, _args(), keep=set(ALL_OUTPUTS))

    assert paths.mixed_final.exists()
    assert paths.mic_path.exists()
    assert paths.sys_path.exists()


def test_produce_outputs_keeps_usable_track_when_mix_impossible(tmp_path: Path):
    # System track empty (header only) -> cannot mix; the mic track must be
    # kept as a fallback even though the user asked only for "mixed".
    paths = _paths_with_tracks(tmp_path, mic_s=0.5, sys_s=0.0)
    _produce_outputs(paths, _args(), keep={"mixed"})

    assert not paths.mixed_final.exists()
    assert paths.mic_path.exists()  # preserved fallback


def test_produce_outputs_discard_tracks_forces_mixed_only(tmp_path: Path):
    paths = _paths_with_tracks(tmp_path, mic_s=0.5, sys_s=0.5)
    _produce_outputs(paths, _args(discard_tracks=True), keep=set(ALL_OUTPUTS))

    assert paths.mixed_final.exists()
    assert not paths.mic_path.exists()
    assert not paths.sys_path.exists()


# -- Transcription flags -----------------------------------------------------


def test_parse_args_transcription_defaults():
    args = _parse_args([])
    assert args.transcribe is True
    assert args.stt_device == "auto"
    assert args.keep_audio is False


def test_parse_args_no_transcribe():
    assert _parse_args(["--no-transcribe"]).transcribe is False


# -- _transcribe_recording ---------------------------------------------------


def _stt_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults = {
        "transcript_dir": str(tmp_path / "Raw"),
        "stt_model": "base.en",
        "stt_device": "cpu",
        "stt_language": "en",
        "keep_audio": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mixed(tmp_path: Path):
    """Produce a recording layout whose mixed file exists on disk."""
    paths = _paths_with_tracks(tmp_path, mic_s=0.5, sys_s=0.5)
    _produce_outputs(paths, _args(), keep={"mixed"})
    assert paths.mixed_final.exists()
    return paths


def test_transcribe_writes_md_and_deletes_wav(tmp_path: Path, monkeypatch):
    paths = _make_mixed(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_file",
        lambda *a, **k: transcription.TranscriptionResult("Hello team.", "en", 1.0, "cpu"),
    )

    _transcribe_recording(paths, _stt_args(tmp_path))

    md = Path(tmp_path / "Raw" / f"{paths.mixed_final.stem}.md")
    assert md.read_text(encoding="utf-8") == "Hello team.\n"
    assert not paths.mixed_final.exists()  # wav removed after success


def test_transcribe_keep_audio_retains_wav(tmp_path: Path, monkeypatch):
    paths = _make_mixed(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_file",
        lambda *a, **k: transcription.TranscriptionResult("kept.", "en", 1.0, "cpu"),
    )

    _transcribe_recording(paths, _stt_args(tmp_path, keep_audio=True))

    assert paths.mixed_final.exists()  # audio preserved


def test_transcribe_failure_keeps_wav_and_writes_nothing(tmp_path: Path, monkeypatch):
    paths = _make_mixed(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("engine down")

    monkeypatch.setattr(transcription, "transcribe_file", boom)

    _transcribe_recording(paths, _stt_args(tmp_path))

    assert paths.mixed_final.exists()  # never lose audio on failure
    assert not (tmp_path / "Raw").exists() or list((tmp_path / "Raw").glob("*.md")) == []


def test_transcribe_empty_text_keeps_wav_and_writes_nothing(tmp_path: Path, monkeypatch):
    paths = _make_mixed(tmp_path)
    monkeypatch.setattr(
        transcription,
        "transcribe_file",
        lambda *a, **k: transcription.TranscriptionResult("   ", "en", 0.0, "cpu"),
    )

    _transcribe_recording(paths, _stt_args(tmp_path))

    assert paths.mixed_final.exists()
    assert not (tmp_path / "Raw").exists() or list((tmp_path / "Raw").glob("*.md")) == []


def test_transcribe_no_mixed_file_is_noop(tmp_path: Path, monkeypatch):
    paths = build_paths("demo", tmp_path / "Recordings", timestamp="2026-07-09_1204")
    called = False

    def spy(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError("should not be called")

    monkeypatch.setattr(transcription, "transcribe_file", spy)

    _transcribe_recording(paths, _stt_args(tmp_path))

    assert called is False
