"""Unit tests for CLI filename sanitization and path construction."""

from __future__ import annotations

import re
from pathlib import Path

from meeting_recorder.cli import (
    ALL_OUTPUTS,
    _rename_recording,
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
