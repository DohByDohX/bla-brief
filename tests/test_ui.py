"""Unit tests for the console UI helpers (rendering + non-TTY fallback)."""

from __future__ import annotations

import io

from rich.console import Console

from meeting_recorder import ui


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    """A themed console writing to an in-memory buffer, TTY or not."""
    sio = io.StringIO()
    con = Console(file=sio, force_terminal=terminal, width=100, theme=ui.THEME, highlight=False)
    return con, sio


def _dev(index: int, name: str, rate: int) -> dict:
    return {
        "index": index,
        "name": name,
        "maxInputChannels": 2,
        "isLoopbackDevice": False,
        "defaultSampleRate": rate,
    }


# -- helpers -----------------------------------------------------------------


def test_fmt_rate_formats_khz():
    assert ui._fmt_rate(48000) == "48.0 kHz"
    assert ui._fmt_rate(16000) == "16.0 kHz"


def test_supports_ui_follows_console(monkeypatch):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    assert ui.supports_ui() is False
    con, _ = _console(terminal=True)
    monkeypatch.setattr(ui, "console", con)
    assert ui.supports_ui() is True


# -- banner ------------------------------------------------------------------


def test_banner_contains_brand_and_version(monkeypatch):
    con, sio = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    ui.print_banner("9.9.9")
    out = sio.getvalue()
    assert "MEETING RECORDER" in out
    assert "v9.9.9" in out
    assert "\x1b[" not in out  # non-tty: no ANSI


# -- device panel ------------------------------------------------------------


def test_device_panel_plain_fallback(monkeypatch, capsys):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    options = [_dev(2, "Headset (OtoKami)", 44100), _dev(14, "Microphone Array", 48000)]
    ui.render_device_panel("mic device", options, 14)
    out = capsys.readouterr().out
    assert "\x1b[" not in out
    assert "Headset (OtoKami)" in out
    assert "48.0 kHz" in out
    assert "default" in out  # the "<-- default" marker on device 14
    assert "[14]" in out


def test_device_panel_tty_renders_title_and_names(monkeypatch):
    con, sio = _console(terminal=True)
    monkeypatch.setattr(ui, "console", con)
    options = [_dev(2, "Headset", 44100), _dev(14, "Microphone Array", 48000)]
    ui.render_device_panel("mic device", options, 14)
    out = sio.getvalue()
    assert "mic device" in out
    assert "Microphone Array" in out


def test_device_panel_long_name_keeps_index_and_rate(monkeypatch):
    # A long device name must not squeeze the index/rate columns out.
    sio = io.StringIO()
    con = Console(file=sio, force_terminal=True, width=72, theme=ui.THEME, highlight=False)
    monkeypatch.setattr(ui, "console", con)
    long_name = "Microphone Array (Intel Smart Sound Technology for Digital Microphones)"
    ui.render_device_panel("mic device", [_dev(9, long_name, 48000)], 9)
    out = sio.getvalue()
    assert "9" in out  # index survives
    assert "48.0 kHz" in out  # full rate, not truncated to "48.…"
    assert "…" in out  # the long name is the thing that gets ellipsized


# -- prompt ------------------------------------------------------------------


def test_prompt_returns_stripped(monkeypatch):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    monkeypatch.setattr("builtins.input", lambda: "  hello  ")
    assert ui.prompt("name") == "hello"


def test_prompt_eof_returns_empty(monkeypatch):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)

    def _raise() -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert ui.prompt("name") == ""


def test_prompt_preserves_bracketed_message(monkeypatch):
    # Bracketed hints like [m]ixed must not be swallowed as rich markup.
    con, sio = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    monkeypatch.setattr("builtins.input", lambda: "m")
    assert ui.prompt("keep [m]ixed [v]oice [s]ystem", default="all") == "m"
    out = sio.getvalue()
    assert "[m]ixed [v]oice [s]ystem" in out
    assert "[all]" in out
    assert "\x1b[" not in out


# -- output summary ----------------------------------------------------------


def test_output_summary_plain_matches_shape(monkeypatch, capsys):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    entries = [("meeting.wav", 133.0, "mixed - for transcription")]
    ui.output_summary(entries, published=r"C:\w\meeting.wav", tracks_dir=None)
    out = capsys.readouterr().out
    assert "\x1b[" not in out
    assert "Output files:" in out
    assert "meeting.wav" in out
    assert "133.0 MB" in out
    assert "(mixed - for transcription)" in out
    assert "Mixed file published to watch folder:" in out
    assert r"C:\w\meeting.wav" in out


def test_output_summary_plain_reports_kept_tracks(monkeypatch, capsys):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    entries = [("mic.wav", 5.0, "your voice")]
    ui.output_summary(entries, published=None, tracks_dir=r"D:\tracks")
    out = capsys.readouterr().out
    assert "Raw tracks kept in:" in out
    assert r"D:\tracks" in out
    assert "Mixed file published" not in out


def test_output_summary_empty(monkeypatch):
    con, sio = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    ui.output_summary([], published=None, tracks_dir=None)
    assert "No output files were produced" in sio.getvalue()


def test_output_summary_tty_renders_panel(monkeypatch):
    con, sio = _console(terminal=True)
    monkeypatch.setattr(ui, "console", con)
    ui.output_summary(
        [("meeting.wav", 133.0, "mixed - for transcription")], published=None, tracks_dir=None
    )
    out = sio.getvalue()
    assert "output" in out  # panel title
    assert "meeting.wav" in out


# -- recording view ----------------------------------------------------------


def test_recording_panel_render_contains_timer(monkeypatch):
    con, sio = _console(terminal=True)
    monkeypatch.setattr(ui, "console", con)
    view = ui.RecordingView("Headset → Speakers")
    con.print(view._render("00:12:34", 10.0, 20.0))
    out = sio.getvalue()
    assert "REC" in out
    assert "00:12:34" in out
    assert "Headset" in out


def test_recording_view_non_tty_prints_hint(monkeypatch, capsys):
    con, _ = _console(terminal=False)
    monkeypatch.setattr(ui, "console", con)
    with ui.RecordingView("mic → sys") as view:
        view.update("00:00:01", 1.0, 2.0)  # no-op off-TTY, must not raise
    assert "RECORDING" in capsys.readouterr().out


def test_recording_view_tty_context_does_not_raise(monkeypatch):
    con, _ = _console(terminal=True)
    monkeypatch.setattr(ui, "console", con)
    with ui.RecordingView("mic → sys") as view:
        view.update("00:00:05", 1.0, 2.0)
