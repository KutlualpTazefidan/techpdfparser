"""Shared pytest fixtures for the extraction test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

import extraction.adapters  # noqa: F401 — trigger adapter registration

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text_pdf() -> Path:
    """Path to the small text-heavy fixture PDF.

    Provided out-of-band by the maintainer (see plan Prerequisites).
    Tests that require it will skip if it is missing, so CI without the
    fixture stays green.
    """
    path = FIXTURES_DIR / "sample_text.pdf"
    if not path.exists():
        pytest.skip(f"Fixture PDF not found: {path}")
    return path


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test isolated output dir; avoids pipeline._assert_output_dir_clean."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out
