"""Shared pytest fixtures and marker registration for the test suite.

Importable helper *functions/classes* live in ``tests/support.py`` instead of
here, so the unit tests can ``from support import ...`` without risking a
collision between multiple ``conftest.py`` modules in the tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers programmatically.

    This guarantees the ``integration`` marker is known even when pytest is
    invoked from a directory where it doesn't discover ``pyproject.toml``
    (e.g. a parent workspace folder), avoiding a PytestUnknownMarkWarning.
    """
    config.addinivalue_line(
        "markers",
        "integration: end-to-end tests that require real audio devices "
        "(deselect with -m 'not integration')",
    )


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """A clean temp directory for generated audio files."""
    d = tmp_path / "audio"
    d.mkdir()
    return d
