"""Console styling for the meeting recorder (Variant B · Framed).

All user-facing presentation lives here so the rest of the package stays
logic-only. Styling is intentionally minimal: a single Tesla-red accent
(:data:`RED`) used sparingly, framed panels for device selection / recording /
output, and no emoji.

Everything degrades gracefully: when stdout is not a real terminal (piped,
redirected to a watch-folder log, etc.) :func:`supports_ui` is False and each
helper falls back to plain text so automation output stays simple and stable.
"""

from __future__ import annotations

from types import TracebackType

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

#: The single accent color (Tesla red). Used only for the wordmark, the REC
#: state, the live timer, the default-device marker, and the input caret.
RED = "#E82127"

THEME = Theme(
    {
        "accent": RED,
        "rec": f"bold {RED}",
        "default": RED,
        "dim": "grey54",
        "bar": "grey62",
        "track": "grey30",
    }
)

console = Console(theme=THEME, highlight=False)


def supports_ui() -> bool:
    """True when the styled UI should render (stdout is an interactive TTY)."""
    return bool(console.is_terminal)


def _fmt_rate(hz: float) -> str:
    """Format a sample rate as a compact kHz string (e.g. ``48.0 kHz``)."""
    return f"{hz / 1000:.1f} kHz"


# -- Banner ------------------------------------------------------------------


def print_banner(version: str) -> None:
    """Print the minimalistic banner: a red rule + title + dim subtitle."""
    console.print()
    console.rule(style="accent")
    console.print(f"  [accent]MEETING RECORDER[/accent] · Dual Capture   [dim]v{version}[/dim]")
    console.print("  [dim]mic + system · synchronized · streams to disk[/dim]")
    console.print()


# -- Device selection --------------------------------------------------------


def render_device_panel(title: str, options: list[dict], default_index: int | None) -> None:
    """Render a framed table of devices, marking the auto-detected default.

    Falls back to a plain aligned list when the styled UI is unavailable.
    """
    if not supports_ui():
        print(f"\n  {title}\n")
        for d in options:
            idx = int(d["index"])
            mark = "  <-- default" if default_index is not None and idx == default_index else ""
            print(f"  [{idx:>2}] {d['name']:<50} {_fmt_rate(d['defaultSampleRate'])}{mark}")
        return

    table = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    table.add_column(justify="right", no_wrap=True)  # index
    table.add_column(no_wrap=True)  # name
    table.add_column(justify="right", no_wrap=True)  # rate
    table.add_column(no_wrap=True)  # marker

    for d in options:
        idx = int(d["index"])
        is_default = default_index is not None and idx == default_index
        style = "default" if is_default else ""
        marker = "[default]●[/default]" if is_default else ""
        table.add_row(
            Text(str(idx), style="default" if is_default else "dim"),
            Text(d["name"], style=style),
            Text(_fmt_rate(d["defaultSampleRate"]), style=style),
            marker,
        )

    console.print(
        Panel(
            table,
            title=f"[dim]{title}[/dim]",
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def prompt(message: str, default: str | None = None) -> str:
    """Show a styled prompt and return the entered line (``""`` on EOF/Ctrl+C).

    The ``message`` is rendered as literal text (never interpreted as markup),
    so bracketed hints like ``[m]ixed`` survive intact.
    """
    line = Text("  ")
    line.append(message)
    if default:
        line.append(f" [{default}]", style="dim")
    line.append(" ")
    line.append("›", style="accent")
    line.append(" ")
    try:
        console.print(line, end="")
        raw = input()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""
    return raw.strip()


# -- Live recording ----------------------------------------------------------


def _bar(fill: float, width: int = 18) -> Text:
    """A fixed-width grayscale meter bar for the given fill fraction [0, 1]."""
    fill = max(0.0, min(1.0, fill))
    filled = int(round(fill * width))
    bar = Text()
    bar.append("█" * filled, style="bar")
    bar.append("░" * (width - filled), style="track")
    return bar


class RecordingView:
    """Live, in-place recording panel (falls back to a single line off-TTY)."""

    def __init__(self, device_line: str) -> None:
        self._device_line = device_line
        self._live: Live | None = None
        self._pulse = False

    def __enter__(self) -> RecordingView:
        if supports_ui():
            # screen=True renders on the alternate screen buffer, which fully
            # repaints every frame — this is resize-safe (an inline Live leaves
            # stale, stacked panels when the terminal width changes mid-render).
            self._live = Live(
                self._render("00:00:00", 0.0, 0.0),
                console=console,
                refresh_per_second=4,
                screen=True,
            )
            self._live.__enter__()
        else:
            print("  RECORDING... Press ENTER to stop.\n")
        return self

    def update(self, timer: str, mic_mb: float, sys_mb: float) -> None:
        """Refresh the panel (TTY) or do nothing (plain line handled by caller)."""
        if self._live is not None:
            self._pulse = not self._pulse
            self._live.update(self._render(timer, mic_mb, sys_mb))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)

    def _render(self, timer: str, mic_mb: float, sys_mb: float) -> Panel:
        peak = max(mic_mb, sys_mb, 0.001)
        dot_style = "rec" if self._pulse else "dim"

        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        header.add_row(
            Text.from_markup(f"[{dot_style}]●[/] [rec]REC[/rec]"),
            Text(timer, style="rec"),
        )

        meters = Table.grid(padding=(0, 1))
        meters.add_column(justify="left", no_wrap=True)
        meters.add_column(no_wrap=True)
        meters.add_column(justify="right", no_wrap=True)
        meters.add_row("[dim]mic[/dim]", _bar(mic_mb / peak), f"{mic_mb:5.1f} MB")
        meters.add_row("[dim]sys[/dim]", _bar(sys_mb / peak), f"{sys_mb:5.1f} MB")

        body = Group(header, Text(""), meters, Text(""), Text(self._device_line, style="dim"))
        return Panel(
            body,
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
            subtitle="[dim]press ENTER to stop[/dim]",
            subtitle_align="right",
        )


# -- Output summary ----------------------------------------------------------


def output_summary(
    entries: list[tuple[str, float, str]],
    published: str | None,
    tracks_dir: str | None,
) -> None:
    """Report the files actually kept on disk.

    ``entries`` are ``(filename, size_mb, label)`` tuples. Renders a framed
    table on a TTY; otherwise prints the plain ``Output files:`` block used by
    non-interactive / watch-folder runs.
    """
    if not entries:
        console.print("  [dim]No output files were produced.[/dim]")
        return

    if not supports_ui():
        print("Output files:")
        for name, size_mb, label in entries:
            print(f"  {name:<50} {size_mb:6.1f} MB  ({label})")
        if published:
            print(f"Mixed file published to watch folder:\n  {published}")
        if tracks_dir:
            print(f"Raw tracks kept in:\n  {tracks_dir}")
        return

    table = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    table.add_column(no_wrap=True)  # name
    table.add_column(justify="right", no_wrap=True)  # size
    table.add_column(no_wrap=True)  # kind
    for name, size_mb, label in entries:
        table.add_row(Text(name, style="accent"), f"{size_mb:.1f} MB", f"[dim]{label}[/dim]")

    console.print(
        Panel(
            table,
            title="[dim]output[/dim]",
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    if published:
        console.print(f"  [dim]published ›[/dim] {published}")
    if tracks_dir:
        console.print(f"  [dim]raw tracks ›[/dim] {tracks_dir}")
