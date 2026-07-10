"""Command-line interface and run orchestration for the meeting recorder."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from meeting_recorder.config import OUTPUT_DIR, SAMPLE_RATE
from meeting_recorder.devices import list_devices, select_devices
from meeting_recorder.logging_setup import configure_logging
from meeting_recorder.mixing import create_mixed_file
from meeting_recorder.recorder import StreamingDualRecorder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingPaths:
    """Resolved on-disk locations for one recording."""

    work_dir: Path  # holds the raw mic/system tracks (outside the watch folder)
    mic_path: Path
    sys_path: Path
    mixed_tmp: Path  # in-progress ".part" file (ignored by *.wav watchers)
    mixed_final: Path  # atomically published final file in the watch folder


def sanitize_name(name: str) -> str:
    """Reduce an arbitrary recording name to a safe filename token.

    Keeps only word chars, dots and dashes so a name can never escape the
    output directory (e.g. ``"../.."``) or contain invalid characters. Falls
    back to ``"recording"`` if nothing usable remains.
    """
    return re.sub(r"[^\w.-]", "_", name).strip("._") or "recording"


#: The three producible outputs, used as the interactive keep-set vocabulary.
ALL_OUTPUTS = frozenset({"mixed", "mic", "system"})

_OUTPUT_ALIASES = {
    "m": "mixed",
    "mix": "mixed",
    "mixed": "mixed",
    "v": "mic",
    "voice": "mic",
    "mic": "mic",
    "s": "system",
    "sys": "system",
    "system": "system",
}


def parse_output_choice(raw: str) -> set[str]:
    """Parse an interactive output selection into a keep-set.

    Accepts comma/space separated tokens using either letters ([m]ixed,
    [v]oice, [s]ystem) or full words. Empty or fully-unrecognized input keeps
    all three outputs (the safe default).
    """
    tokens = re.split(r"[,\s]+", raw.strip().lower())
    chosen = {_OUTPUT_ALIASES[t] for t in tokens if t in _OUTPUT_ALIASES}
    return chosen or set(ALL_OUTPUTS)


def _prompt(message: str) -> str:
    """Read a line of input, returning "" on EOF/Ctrl+C (never raising)."""
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def build_paths(
    name: str | None,
    output_dir: Path,
    tracks_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> RecordingPaths:
    """Compute all file paths for a recording.

    Watch-folder safety: raw tracks live in a sibling ``"<output> - tracks"``
    folder OUTSIDE the watch folder, and the mixed file is built under a
    ``.part`` name then atomically renamed, so a ``*.wav`` watcher only ever
    sees a single finished file.
    """
    timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    base = f"{timestamp}_{sanitize_name(name)}" if name else timestamp

    work_dir = Path(tracks_dir) if tracks_dir else output_dir.parent / f"{output_dir.name} - tracks"
    return RecordingPaths(
        work_dir=work_dir,
        mic_path=work_dir / f"{base}_mic-only.wav",
        sys_path=work_dir / f"{base}_system-only.wav",
        mixed_tmp=output_dir / f".{base}.wav.part",
        mixed_final=output_dir / f"{base}.wav",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="meeting-recorder",
        description="Record mic + system audio simultaneously for meeting notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  meeting-recorder                        Start recording (auto-named)
  meeting-recorder -n "jon-headcount"     Named recording
  meeting-recorder --mic 15               Use AirPods mic (device 15)
  meeting-recorder --mic-gain 1.3         Boost your mic volume
  meeting-recorder -l                     List available devices""",
    )
    parser.add_argument(
        "--name", "-n", default=None, help="Recording name (default: YYYY-MM-DD_HHMM)"
    )
    parser.add_argument(
        "--mic", type=int, default=None, help="Mic device ID (default: Windows default input)"
    )
    parser.add_argument(
        "--system",
        type=int,
        default=None,
        help="System/loopback device ID (default: auto, follows the default output)",
    )
    parser.add_argument(
        "--list-devices", "-l", action="store_true", help="List available devices and exit"
    )
    parser.add_argument(
        "--mic-gain", type=float, default=1.0, help="Mic volume multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--sys-gain", type=float, default=1.0, help="System volume multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tracks-dir",
        default=None,
        help="Where to keep the raw mic/system tracks (default: a '<output> - tracks' "
        "folder next to the output dir, kept OUTSIDE the watch folder).",
    )
    parser.add_argument(
        "--discard-tracks",
        action="store_true",
        help="Delete the raw mic/system tracks after a successful mix (keep only the mixed file).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt for input/system devices before recording, and for a name and "
        "which outputs to keep after recording.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args(argv)


def _print_banner() -> None:
    print()
    print("  +================================================+")
    print("  |        MEETING RECORDER - Dual Capture  v4      |")
    print("  |  Mic (your voice) + System (meeting audio)      |")
    print("  |  Synchronized capture - streams to disk         |")
    print("  +================================================+")
    print()


def _run_recording_ui(recorder: StreamingDualRecorder, paths: RecordingPaths) -> None:
    """Drive the interactive timer + ENTER/Ctrl+C stop loop while recording."""
    print("  RECORDING... Press ENTER to stop.\n")
    start_time = time.time()

    def timer_display() -> None:
        while recorder.is_recording:
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            hrs, mins = divmod(mins, 60)
            try:
                mic_sz = paths.mic_path.stat().st_size / (1024 * 1024)
                sys_sz = paths.sys_path.stat().st_size / (1024 * 1024)
            except OSError:
                mic_sz = sys_sz = 0
            print(
                f"\r  Elapsed: {hrs:02d}:{mins:02d}:{secs:02d}  |  "
                f"Mic: {mic_sz:.1f}MB  Sys: {sys_sz:.1f}MB",
                end="",
                flush=True,
            )
            time.sleep(2)

    def wait_for_enter() -> None:
        # ENTER and the SIGINT handler both request stop, so the poll loop
        # below exits cleanly either way (no second keypress needed).
        try:
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            pass
        recorder.request_stop()

    threading.Thread(target=timer_display, daemon=True).start()
    threading.Thread(target=wait_for_enter, daemon=True).start()

    try:
        while recorder.is_recording:
            time.sleep(0.1)
    except KeyboardInterrupt:
        recorder.request_stop()


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        log.debug("Could not delete %s", p, exc_info=True)


def _rename_recording(
    old: RecordingPaths,
    name: str,
    output_dir: Path,
    tracks_dir: str | None,
    timestamp: str,
) -> RecordingPaths:
    """Rename the already-written raw tracks to embed a chosen name.

    Returns updated paths (including the new mixed-file names). The same
    ``timestamp`` is reused so the renamed files match the recording instant.
    """
    new = build_paths(name, output_dir, tracks_dir, timestamp=timestamp)
    for src, dst in ((old.mic_path, new.mic_path), (old.sys_path, new.sys_path)):
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
    return new


def _produce_outputs(paths: RecordingPaths, args: argparse.Namespace, keep: set[str]) -> None:
    """Mix (when requested), prune unselected tracks, and report what was kept."""
    mic_ok = paths.mic_path.exists() and paths.mic_path.stat().st_size > 44
    sys_ok = paths.sys_path.exists() and paths.sys_path.stat().st_size > 44

    want_mixed = "mixed" in keep
    # --discard-tracks still forces raw tracks away after a mix (legacy flag).
    keep_mic = "mic" in keep and not args.discard_tracks
    keep_sys = "system" in keep and not args.discard_tracks

    mixed_made = False
    if want_mixed and mic_ok and sys_ok:
        log.info("Creating mixed file...")
        if create_mixed_file(
            paths.mic_path,
            paths.sys_path,
            paths.mixed_tmp,
            args.mic_gain,
            args.sys_gain,
            sync_offset="auto",
        ):
            # Publish atomically: a rename appears to a watch-folder transcriber
            # as a single finished file (never "still being written").
            paths.mixed_final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(paths.mixed_tmp, paths.mixed_final)
            mixed_made = True
        else:
            _safe_unlink(paths.mixed_tmp)  # no stray .part left behind
    elif want_mixed and mic_ok:
        log.warning("System audio track is empty; cannot mix. Keeping the mic track instead.")
        keep_mic = True  # fall back to the one usable track so nothing is lost
    elif want_mixed and sys_ok:
        log.warning("Mic track is empty; cannot mix. Keeping the system track instead.")
        keep_sys = True
    elif want_mixed:
        log.error("Both tracks are empty; nothing to mix!")

    # Prune the raw tracks that are not being kept.
    if not keep_mic:
        _safe_unlink(paths.mic_path)
    if not keep_sys:
        _safe_unlink(paths.sys_path)

    _report_outputs(paths, mixed_made, keep_mic, keep_sys)


def _report_outputs(
    paths: RecordingPaths, mixed_made: bool, kept_mic: bool, kept_sys: bool
) -> None:
    """Summarize only the files that were actually left on disk."""
    entries: list[tuple[Path, str]] = []
    if mixed_made and paths.mixed_final.exists():
        entries.append((paths.mixed_final, "mixed - for transcription"))
    if kept_mic and paths.mic_path.exists():
        entries.append((paths.mic_path, "your voice"))
    if kept_sys and paths.sys_path.exists():
        entries.append((paths.sys_path, "system/meeting audio"))

    if not entries:
        log.warning("No output files were produced.")
        return

    log.info("Output files:")
    for path, label in entries:
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info("  %-52s %6.1f MB  (%s)", path.name, size_mb, label)
    if mixed_made:
        log.info("Mixed file published to watch folder:\n    %s", paths.mixed_final)
    if kept_mic or kept_sys:
        log.info("Raw tracks kept in:\n    %s", paths.work_dir)


def _finalize(
    recorder: StreamingDualRecorder,
    paths: RecordingPaths,
    args: argparse.Namespace,
    timestamp: str,
    interactive: bool,
) -> None:
    """Stop the recorder, then (optionally interactively) name and emit outputs."""
    print("\n\n  Stopping recording...")
    mic_samples, sys_samples = recorder.stop()
    mic_rate = recorder._mic_native_rate or SAMPLE_RATE
    sys_rate = recorder._sys_native_rate or SAMPLE_RATE
    log.info("Mic: %s samples (%.1fs)", f"{mic_samples:,}", mic_samples / mic_rate)
    log.info("Sys: %s samples (%.1fs)", f"{sys_samples:,}", sys_samples / sys_rate)

    if abs(recorder.sync_offset) > 0.05:
        late = "system" if recorder.sync_offset > 0 else "mic"
        log.info("Measured start offset: %s began %.2fs late.", late, abs(recorder.sync_offset))

    # Feature: choose a name after recording (only if not preset via -n).
    if interactive and not args.name:
        name = _prompt("\n  Meeting name (Enter = keep timestamp): ")
        if name:
            paths = _rename_recording(
                paths, name, Path(args.output_dir), args.tracks_dir, timestamp
            )

    # Feature: choose which outputs to keep.
    keep = set(ALL_OUTPUTS)
    if interactive:
        keep = parse_output_choice(
            _prompt("  Keep which outputs? [m]ixed  [v]oice  [s]ystem  (Enter = all): ")
        )
        log.info("Keeping: %s", ", ".join(sorted(keep)))

    _produce_outputs(paths, args, keep)

    print("\n  Done!\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_logging(args.debug)

    if args.list_devices:
        list_devices()
        return

    _print_banner()

    # Interactive device selection (Enter accepts the auto-detected default).
    mic_index = args.mic
    loopback_index = args.system
    if args.interactive:
        picked_mic, picked_sys = select_devices()
        if args.mic is None:
            mic_index = picked_mic
        if args.system is None:
            loopback_index = picked_sys

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    paths = build_paths(args.name, Path(args.output_dir), args.tracks_dir, timestamp=timestamp)

    recorder = StreamingDualRecorder(
        mic_path=paths.mic_path,
        sys_path=paths.sys_path,
        mic_index=mic_index,
        loopback_index=loopback_index,
        sample_rate=SAMPLE_RATE,
    )

    try:
        recorder.start()
    except Exception as e:
        log.error("Error starting recording: %s", e)
        log.error(
            "Run with -l to see available devices, then use --mic <ID> to specify the microphone."
        )
        recorder.stop()  # release any partially-opened handles
        sys.exit(1)

    def handle_signal(sig, frame):
        print("\n\n  Caught Ctrl+C - stopping gracefully...")
        recorder.request_stop()

    signal.signal(signal.SIGINT, handle_signal)

    _run_recording_ui(recorder, paths)
    _finalize(recorder, paths, args, timestamp, args.interactive)


if __name__ == "__main__":
    main()
