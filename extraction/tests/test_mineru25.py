"""Tests for the mineru25 adapter.

Unit tests (CPU-only) verify registration under both roles and that
the class can be constructed without triggering any mineru imports.
Integration tests (GPU-gated via -m integration) appear in later tasks.
"""
from __future__ import annotations

from extraction.registry import get_segmenter, get_table_extractor


def test_mineru25_segmenter_tool_name() -> None:
    assert get_segmenter("mineru25").tool_name == "mineru25"


def test_mineru25_registers_as_table_extractor() -> None:
    assert get_table_extractor("mineru25").tool_name == "mineru25"


def test_mineru25_construction_does_not_import_mineru() -> None:
    # Construction must be lazy; importing/instantiating the class
    # must not require mineru or torch at class-definition time.
    import sys

    # Force a clean import state for this check.
    before = set(sys.modules.keys())
    adapter = get_segmenter("mineru25")
    after = set(sys.modules.keys())
    newly_imported = after - before
    # No heavy dependency should have been pulled in by mere construction.
    assert not any(m.startswith("mineru") for m in newly_imported)
    assert not any(m.startswith("torch") for m in newly_imported)
    assert not any(m.startswith("transformers") for m in newly_imported)
    assert adapter is not None


def test_mineru25_loaded_flag_starts_false() -> None:
    adapter = get_segmenter("mineru25")
    # Adapter exposes a `_loaded` flag that later lifecycle tests use
    # as a lightweight observable of load state (full VRAM verification
    # happens in the integration tests).
    assert adapter._loaded is False
