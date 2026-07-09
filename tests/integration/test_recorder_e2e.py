"""End-to-end integration tests for the meeting recorder (v4.2).

These spawn the real recorder (``python -m meeting_recorder``) and require a
working microphone AND a WASAPI loopback device, so they are marked
``integration`` and excluded from the default test run. Run them explicitly:

    pytest -m integration

Some tests reuse the output of earlier tests (T6 reads T3's files, T11 reads
T1's), so they rely on pytest's in-file execution order.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

PYTHON = sys.executable
RECORDER_CMD = [PYTHON, "-m", "meeting_recorder"]
TEST_DIR = Path(r"C:\Users\pchavan\AppData\Local\Temp\opencode\qa_tests")


# -- Session setup -----------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _clean_test_dir():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield


# -- Helpers -----------------------------------------------------------------


def validate_wav(path: Path, min_duration_s: float = 0) -> tuple[bool, str]:
    """Validate a WAV file is readable and has expected properties."""
    if not path.exists():
        return False, f"File does not exist: {path}"
    if path.stat().st_size < 44:
        return False, f"File too small ({path.stat().st_size} bytes)"
    try:
        with wave.open(str(path), "rb") as wf:
            issues = []
            ch, sw, fr, nf = (
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate(),
                wf.getnframes(),
            )
            dur = nf / fr if fr > 0 else 0
            if ch != 1:
                issues.append(f"channels={ch}")
            if sw != 2:
                issues.append(f"sampwidth={sw}")
            if fr not in (16000, 44100, 48000):
                issues.append(f"rate={fr}")
            if min_duration_s > 0 and dur < min_duration_s * 0.8:
                issues.append(f"dur={dur:.1f}s<{min_duration_s}s")
            wf.rewind()
            raw = wf.readframes(min(nf, fr))
            peak = float(np.max(np.abs(np.frombuffer(raw, dtype=np.int16)))) / 32768.0
            if issues:
                return False, "; ".join(issues)
            return True, f"{dur:.1f}s, peak={peak:.4f}, {path.stat().st_size / 1024:.0f}KB"
    except Exception as e:
        return False, f"Cannot read: {e}"


def run_recorder(args, record_seconds=10, timeout=60):
    """Run the recorder as a subprocess, record, stop via ENTER, return proc."""
    proc = subprocess.Popen(
        RECORDER_CMD + args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(record_seconds)
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        proc.communicate()
    return proc


def find_tracks_dir(out_dir: Path) -> Path:
    """Derive the default tracks directory (sibling '<name> - tracks')."""
    return out_dir.parent / f"{out_dir.name} - tracks"


# =============================================================================
# T1: Short recording — watch-folder layout + .part atomicity
# =============================================================================
def test_short_recording():
    out_dir = TEST_DIR / "test1" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = run_recorder(["-n", "qa-test1", "-o", str(out_dir)], record_seconds=10)
    assert proc.returncode == 0

    mixed = [f for f in out_dir.glob("*qa-test1*.wav") if ".part" not in f.name]
    assert len(mixed) == 1, [f.name for f in mixed]

    assert not list(out_dir.glob("*.part")), "stray .part left behind"
    assert not list(out_dir.glob("*mic-only*")), "raw mic track leaked into watch folder"
    assert not list(out_dir.glob("*system-only*")), "raw system track leaked into watch folder"

    mic_tracks = list(tracks_dir.glob("*mic-only*"))
    sys_tracks = list(tracks_dir.glob("*system-only*"))
    assert len(mic_tracks) == 1, f"tracks_dir={tracks_dir}"
    assert len(sys_tracks) == 1, f"tracks_dir={tracks_dir}"

    ok, detail = validate_wav(mixed[0], min_duration_s=8)
    assert ok, detail
    ok, detail = validate_wav(mic_tracks[0], min_duration_s=8)
    assert ok, detail
    ok, detail = validate_wav(sys_tracks[0], min_duration_s=8)
    assert ok, detail


# =============================================================================
# T2: Audio content quality
# =============================================================================
def test_audio_quality():
    out_dir = TEST_DIR / "test2" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = run_recorder(["-n", "qa-test2", "-o", str(out_dir)], record_seconds=8)
    assert proc.returncode == 0

    mic_files = list(tracks_dir.glob("*mic-only*"))
    assert mic_files, "No mic file found"
    with wave.open(str(mic_files[0]), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    raw_int = np.frombuffer(raw, dtype=np.int16)
    samples = raw_int.astype(np.float32) / 32768.0
    assert float(np.max(np.abs(samples))) > 0.00001, "mic silent (below noise floor)"
    clipped = int(np.sum(np.abs(raw_int) >= 32767))
    assert clipped < len(raw_int) * 0.01, f"{clipped}/{len(raw_int)} clipped"

    mixed = [f for f in out_dir.glob("*qa-test2*.wav") if ".part" not in f.name]
    assert mixed
    with wave.open(str(mixed[0]), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    peak = float(np.max(np.abs(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)))
    assert peak > 0.0001, "mixed file has no audio"
    assert peak <= 0.96, f"mixed not normalized (peak={peak})"


# =============================================================================
# T3: Memory stability (60s)
# =============================================================================
def test_memory_stability():
    out_dir = TEST_DIR / "test3" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = subprocess.Popen(
        RECORDER_CMD + ["-n", "qa-test3", "-o", str(out_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = proc.pid
    mem_samples = []
    for _ in range(6):
        time.sleep(10)
        try:
            r = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64 / 1MB",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            mem_samples.append(float(r.stdout.strip()) if r.stdout.strip() else 0)
        except Exception:
            mem_samples.append(0)

    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.communicate(timeout=60)
    except Exception:
        proc.kill()
        proc.communicate()
        pytest.fail("60s recording crashed or timed out")

    assert proc.returncode == 0

    if len(mem_samples) >= 2 and all(m > 0 for m in mem_samples):
        growth = mem_samples[-1] - mem_samples[0]
        assert growth < 50, f"memory grew {growth:+.1f}MB: {mem_samples}"
        assert max(mem_samples) < 200, f"peak {max(mem_samples):.0f}MB"

    mixed = [f for f in out_dir.glob("*qa-test3*.wav") if ".part" not in f.name]
    assert len(mixed) == 1
    ok, detail = validate_wav(mixed[0], min_duration_s=55)
    assert ok, detail
    for track in sorted(tracks_dir.glob("*qa-test3*")):
        ok, detail = validate_wav(track, min_duration_s=55)
        assert ok, f"{track.name}: {detail}"


# =============================================================================
# T4: Crash safety — kill mid-recording
# =============================================================================
def test_crash_safety():
    out_dir = TEST_DIR / "test4" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = subprocess.Popen(
        RECORDER_CMD + ["-n", "qa-test4", "-o", str(out_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(10)
    proc.kill()
    proc.wait()

    mic_files = list(tracks_dir.glob("*mic-only*"))
    sys_files = list(tracks_dir.glob("*system-only*"))
    assert mic_files, "mic file missing after crash"
    assert sys_files, "system file missing after crash"

    assert mic_files[0].stat().st_size > 1000, "mic file empty after crash"
    with wave.open(str(mic_files[0]), "rb") as wf:
        assert wf.getnframes() > 0, "mic WAV header invalid after crash"
    assert sys_files[0].stat().st_size > 1000, "system file empty after crash"


# =============================================================================
# T5: Ctrl+C graceful shutdown
# =============================================================================
def test_ctrl_c_shutdown():
    out_dir = TEST_DIR / "test5" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = subprocess.Popen(
        RECORDER_CMD + ["-n", "qa-test5", "-o", str(out_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    time.sleep(8)
    try:
        ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, proc.pid)
    except Exception:
        try:
            os.kill(proc.pid, signal.SIGINT)
        except Exception:
            proc.terminate()

    try:
        proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail("graceful shutdown timed out")

    tracks = list(tracks_dir.glob("*qa-test5*"))
    assert tracks, "no tracks written before Ctrl+C"
    for track in tracks:
        ok, detail = validate_wav(track, min_duration_s=5)
        assert ok, f"{track.name}: {detail}"


# =============================================================================
# T6: File integrity deep check (reads T3 output)
# =============================================================================
def test_file_integrity():
    test3_out = TEST_DIR / "test3" / "output"
    test3_tracks = find_tracks_dir(test3_out)
    all_files = list(test3_out.glob("*qa-test3*.wav")) + list(test3_tracks.glob("*qa-test3*"))
    all_files = [f for f in all_files if ".part" not in f.name and f.stat().st_size > 44]
    assert all_files, "No T3 files found — run test_memory_stability first"

    for f in sorted(all_files):
        with wave.open(str(f), "rb") as wf:
            total = wf.getnframes()
            ch, sw = wf.getnchannels(), wf.getsampwidth()
            read = 0
            while read < total:
                n = min(48000, total - read)
                raw = wf.readframes(n)
                assert len(raw) == n * ch * sw, f"{f.name}: short read at frame {read}"
                read += n


# =============================================================================
# T7: File access during recording; no partial .wav in watch dir
# =============================================================================
def test_concurrent_access():
    out_dir = TEST_DIR / "test7" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = subprocess.Popen(
        RECORDER_CMD + ["-n", "qa-test7", "-o", str(out_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(5)

    mic_files = list(tracks_dir.glob("*mic-only*"))
    assert mic_files, "mic track not created during recording"
    s1 = mic_files[0].stat().st_size
    time.sleep(2)
    s2 = mic_files[0].stat().st_size
    assert s2 > s1, "file not growing during recording"

    assert not list(out_dir.glob("*.wav")), "finished .wav appeared in watch dir mid-recording"

    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.communicate(timeout=30)
    except Exception:
        proc.kill()
        proc.communicate()
    assert proc.returncode == 0


# =============================================================================
# T8: --discard-tracks flag
# =============================================================================
def test_discard_tracks():
    out_dir = TEST_DIR / "test8" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir = find_tracks_dir(out_dir)

    proc = run_recorder(
        ["-n", "qa-test8", "-o", str(out_dir), "--discard-tracks"], record_seconds=8
    )
    assert proc.returncode == 0

    mixed = [f for f in out_dir.glob("*qa-test8*.wav") if ".part" not in f.name]
    assert len(mixed) == 1
    ok, detail = validate_wav(mixed[0], min_duration_s=5)
    assert ok, detail

    remaining = list(tracks_dir.glob("*qa-test8*"))
    assert not remaining, f"raw tracks not deleted: {[f.name for f in remaining]}"


# =============================================================================
# T9: --tracks-dir custom directory
# =============================================================================
def test_custom_tracks_dir():
    out_dir = TEST_DIR / "test9" / "output"
    custom_tracks = TEST_DIR / "test9" / "my-custom-tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = run_recorder(
        ["-n", "qa-test9", "-o", str(out_dir), "--tracks-dir", str(custom_tracks)], record_seconds=8
    )
    assert proc.returncode == 0

    default_tracks = find_tracks_dir(out_dir)
    in_default = list(default_tracks.glob("*qa-test9*")) if default_tracks.exists() else []
    in_custom = list(custom_tracks.glob("*qa-test9*")) if custom_tracks.exists() else []
    assert len(in_custom) >= 2, f"expected tracks in {custom_tracks.name}"
    assert not in_default, "tracks leaked into default dir"


# =============================================================================
# T10: Filename sanitization (dangerous --name)
# =============================================================================
def test_filename_sanitization():
    out_dir = TEST_DIR / "test10" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = run_recorder(["-n", "../../etc/passwd", "-o", str(out_dir)], record_seconds=5)
    assert proc.returncode == 0

    all_files = list(out_dir.glob("*.wav")) + list(find_tracks_dir(out_dir).glob("*.wav"))
    assert all_files, "no files produced"
    for f in all_files:
        assert not (".." in f.name or "/" in f.name or "\\" in f.name), f"traversal in {f.name}"


# =============================================================================
# T11: Track sync alignment (reads T1 output)
# =============================================================================
def test_sync_alignment():
    test1_out = TEST_DIR / "test1" / "output"
    test1_tracks = find_tracks_dir(test1_out)
    mic_files = list(test1_tracks.glob("*mic-only*")) if test1_tracks.exists() else []
    sys_files = list(test1_tracks.glob("*system-only*")) if test1_tracks.exists() else []
    mixed_files = (
        [f for f in test1_out.glob("*.wav") if ".part" not in f.name] if test1_out.exists() else []
    )
    assert mic_files and sys_files, "run test_short_recording first"

    with wave.open(str(mic_files[0]), "rb") as wf:
        mic_dur = wf.getnframes() / wf.getframerate()
    with wave.open(str(sys_files[0]), "rb") as wf:
        sys_dur = wf.getnframes() / wf.getframerate()
    assert abs(mic_dur - sys_dur) < 2.0, f"mic={mic_dur:.2f}s sys={sys_dur:.2f}s"

    if mixed_files:
        with wave.open(str(mixed_files[0]), "rb") as wf:
            mixed_dur = wf.getnframes() / wf.getframerate()
        assert mixed_dur >= max(mic_dur, sys_dur) * 0.95, "mixed shorter than longest track"


# =============================================================================
# T12: Default (timestamp-only) filename
# =============================================================================
def test_default_name():
    out_dir = TEST_DIR / "test12" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = run_recorder(["-o", str(out_dir)], record_seconds=5)
    assert proc.returncode == 0

    mixed = [f for f in out_dir.glob("*.wav") if ".part" not in f.name]
    assert len(mixed) == 1
    assert re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}$", mixed[0].stem), f"name='{mixed[0].stem}'"
