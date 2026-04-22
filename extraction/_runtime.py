"""Runtime helpers used by the heavy VLM/OCR adapters."""
from __future__ import annotations

import gc
import sys
from typing import Any


def release_runtime_resources(*adapters: Any) -> None:
    """Drop adapter-held model references and clear CUDA cache if available."""
    for adapter in adapters:
        if adapter is None:
            continue
        unload = getattr(adapter, "unload", None)
        if callable(unload):
            unload()
    gc.collect()
    _clear_cuda_cache()


def _clear_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def is_cuda_oom(exc: BaseException) -> bool:
    torch = sys.modules.get("torch")
    if torch is not None:
        oom_type = getattr(torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message
