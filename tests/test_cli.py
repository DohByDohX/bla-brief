"""Unit tests for CLI filename sanitization and path construction."""

from __future__ import annotations

import re
from pathlib import Path

from meeting_recorder.cli import build_paths, sanitize_name


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
