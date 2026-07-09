"""Logging configuration for the meeting recorder.

Library modules (devices, recorder, mixing) log through ``logging.getLogger``
so the same code can be quiet by default, verbose with ``--debug``, and later
redirected to a file without touching any call sites. The interactive console
UI (banner, live timer) stays as ``print`` in :mod:`meeting_recorder.cli`.
"""

import logging


def configure_logging(debug: bool = False) -> None:
    """Configure root logging for a CLI run.

    Args:
        debug: When True, emit DEBUG-level records with timestamps. Otherwise
            show a clean INFO-and-above stream without noisy prefixes.
    """
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s" if debug else "  %(message)s"
    logging.basicConfig(level=level, format=fmt)
