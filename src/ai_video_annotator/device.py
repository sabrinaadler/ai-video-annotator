"""PyTorch device selection helpers."""

from __future__ import annotations

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Select a PyTorch device.

    Parameters
    ----------
    preference:
        One of ``auto``, ``mps``, ``cuda``, or ``cpu``.

    ``auto`` prefers Apple Metal (MPS), then CUDA, then CPU. This ordering is
    deliberate because the primary development target is an Apple Silicon Mac.
    """

    normalized = preference.lower().strip()
    allowed = {"auto", "mps", "cuda", "cpu"}
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unknown device preference {preference!r}. Expected one of: {choices}")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
