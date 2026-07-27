"""Command-line interface and run orchestration for the meeting recorder."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from meeting_recorder import __version__, transcription, ui
from meeting_recorder.config import (
    OUTPUT_DIR,
    POST_TRANSCRIBE_SCRIPT,
    SAMPLE_RATE,
    STT_DEVICE,
    STT_LANGUAGE,
    STT_MODEL,
    TRANSCRIPT_DIR,
)
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
    parser.add_argument(
        "--transcribe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transcribe the mixed file after recording (default: on). Use --no-transcribe "
        "to skip.",
    )
    parser.add_argument(
        "--stt-model", default=STT_MODEL, help=f"Whisper model to use (default: {STT_MODEL})"
    )
    parser.add_argument(
        "--stt-language",
        default=STT_LANGUAGE,
        help=f"Transcription language hint (default: {STT_LANGUAGE})",
    )
    parser.add_argument(
        "--stt-device",
        default=STT_DEVICE,
        choices=("auto", "cuda", "cpu"),
        help=f"Transcription backend (default: {STT_DEVICE}; 'auto' tries GPU then CPU).",
    )
    parser.add_argument(
        "--transcript-dir",
        default=str(TRANSCRIPT_DIR),
        help=f"Where to write the .md transcript (default: {TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the mixed .wav after a successful transcription (default: delete it).",
    )
    parser.add_argument(
        "--automation",
        dest="run_automation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After a successful transcription, fire the meeting catch-up automation "
        "(default: on). Use --no-automation to skip.",
    )
    parser.add_argument(
        "--automation-script",
        default=str(POST_TRANSCRIBE_SCRIPT),
        help=f"PowerShell script fired after transcription (default: {POST_TRANSCRIBE_SCRIPT}).",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Fetch the transcription model into the local cache and exit (the only "
        "step that goes online). Run once on an approved network; recordings then "
        "transcribe fully offline.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args(argv)


def _print_banner() -> None:
    ui.print_banner(__version__)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _track_sizes(paths: RecordingPaths) -> tuple[float, float]:
    """Current on-disk sizes (MB) of the mic and system tracks."""
    try:
        mic_mb = paths.mic_path.stat().st_size / (1024 * 1024)
        sys_mb = paths.sys_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0, 0.0
    return mic_mb, sys_mb


def _device_line(recorder: StreamingDualRecorder) -> str:
    """Short 'mic -> system' description shown in the recording panel."""

    def _short(info: dict | None, fallback: str) -> str:
        if not info:
            return fallback
        name = str(info["name"])
        return name if len(name) <= 24 else name[:23] + "…"

    mic_k = (recorder._mic_native_rate or SAMPLE_RATE) // 1000
    sys_k = (recorder._sys_native_rate or SAMPLE_RATE) // 1000
    mic = _short(recorder._mic_info, "mic")
    system = _short(recorder._loopback_info, "system")
    return f"{mic} {mic_k}k → {system} {sys_k}k"


def _run_recording_ui(recorder: StreamingDualRecorder, paths: RecordingPaths) -> None:
    """Drive the live recording view + ENTER/Ctrl+C stop loop while recording."""
    start_time = time.time()

    def wait_for_enter() -> None:
        # ENTER and the SIGINT handler both request stop, so the poll loop
        # below exits cleanly either way (no second keypress needed).
        try:
            sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            pass
        recorder.request_stop()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    tty = ui.supports_ui()
    with ui.RecordingView(_device_line(recorder)) as view:
        try:
            while recorder.is_recording:
                elapsed = time.time() - start_time
                mic_mb, sys_mb = _track_sizes(paths)
                if tty:
                    view.update(_fmt_elapsed(elapsed), mic_mb, sys_mb)
                    time.sleep(0.25)
                else:
                    print(
                        f"\r  Elapsed: {_fmt_elapsed(elapsed)}  |  "
                        f"Mic: {mic_mb:.1f}MB  Sys: {sys_mb:.1f}MB",
                        end="",
                        flush=True,
                    )
                    time.sleep(2)
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
    files: list[tuple[Path, str]] = []
    if mixed_made and paths.mixed_final.exists():
        files.append((paths.mixed_final, "mixed - for transcription"))
    if kept_mic and paths.mic_path.exists():
        files.append((paths.mic_path, "your voice"))
    if kept_sys and paths.sys_path.exists():
        files.append((paths.sys_path, "system/meeting audio"))

    entries = [(p.name, p.stat().st_size / (1024 * 1024), label) for p, label in files]
    published = str(paths.mixed_final) if mixed_made else None
    tracks_dir = str(paths.work_dir) if (kept_mic or kept_sys) else None
    ui.output_summary(entries, published=published, tracks_dir=tracks_dir)


def _transcribe_recording(paths: RecordingPaths, args: argparse.Namespace) -> None:
    """Transcribe the mixed file, write the .md, and remove the wav on success.

    Best-effort: any failure is logged and the audio is left in place so a
    recording is never lost to a transcription problem. Does nothing when there
    is no mixed file (e.g. the user chose not to keep it).
    """
    if not paths.mixed_final.exists():
        log.warning("No mixed file to transcribe; skipping transcription.")
        return

    dest_md = Path(args.transcript_dir) / f"{paths.mixed_final.stem}.md"
    try:
        result = transcription.transcribe_file(
            paths.mixed_final,
            model=args.stt_model,
            device=args.stt_device,
            language=args.stt_language,
        )
    except Exception as exc:  # noqa: BLE001 - never lose audio to a transcription error
        log.error("Transcription failed (%s); keeping the audio file.", exc)
        return

    if not result.text.strip():
        log.warning("Transcription produced no text; keeping the audio, not writing a transcript.")
        return

    md_path = transcription.write_transcript(result.text, dest_md)
    log.info("Transcript written to %s (%s).", md_path, result.device)
    print(f"\n  Transcript: {md_path}")

    if not args.keep_audio:
        _safe_unlink(paths.mixed_final)
        log.info("Removed mixed audio %s after transcription.", paths.mixed_final.name)

    if args.run_automation:
        _fire_automation(Path(args.automation_script))


def _fire_automation(script: Path) -> None:
    """Launch the meeting catch-up automation detached and return immediately.

    The wrapper is self-gating and locked, so firing it unconditionally is safe.
    It is started fully detached (its own process group, no inherited streams)
    so the recorder can exit without waiting on the (potentially long) run; the
    script logs its own progress to ``run.log``.
    """
    if not script.exists():
        log.warning("Automation script not found (%s); skipping automation.", script)
        return
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    creationflags = 0
    if os.name == "nt":
        # No console window + its own process group (so Ctrl+C to the recorder
        # does not propagate). DETACHED_PROCESS is intentionally NOT used: it
        # prevents PowerShell from executing under these conditions on Windows.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
        log.info("Fired catch-up automation (detached): %s", script.name)
    except OSError as exc:
        log.error("Could not launch automation (%s): %s", script, exc)


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
        name = ui.prompt("meeting name", default="keep timestamp")
        if name:
            paths = _rename_recording(
                paths, name, Path(args.output_dir), args.tracks_dir, timestamp
            )

    # Feature: choose which outputs to keep.
    keep = set(ALL_OUTPUTS)
    if interactive:
        keep = parse_output_choice(ui.prompt("keep [m]ixed [v]oice [s]ystem", default="all"))
        log.info("Keeping: %s", ", ".join(sorted(keep)))

    _produce_outputs(paths, args, keep)

    if args.transcribe:
        _transcribe_recording(paths, args)

    print("\n  Done!\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_logging(args.debug)

    if args.list_devices:
        list_devices()
        return

    if args.download_model:
        # The single sanctioned online step; populates the cache, then exits.
        transcription.download_model(args.stt_model)
        print(f"\n  Model '{args.stt_model}' cached. Recordings now transcribe offline.\n")
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
