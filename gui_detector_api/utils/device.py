from __future__ import annotations

from typing import Literal

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
ResolvedDeviceName = Literal["cpu", "cuda", "mps"]


class DeviceResolutionError(Exception):
    """Raised when the requested inference device is unavailable."""


def _load_torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_is_available(torch_module) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    checker = getattr(cuda, "is_available", None)
    return bool(checker and checker())


def _mps_is_available(torch_module) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    checker = getattr(mps, "is_available", None)
    return bool(checker and checker())


def resolve_device(
    requested: DeviceName,
    *,
    torch_module=None,
) -> ResolvedDeviceName:
    torch_module = _load_torch_module() if torch_module is None else torch_module

    if requested == "cpu":
        return "cpu"

    if torch_module is None:
        if requested == "auto":
            return "cpu"
        raise DeviceResolutionError(
            f"Requested device '{requested}' cannot be validated because torch is not installed."
        )

    cuda_available = _cuda_is_available(torch_module)
    mps_available = _mps_is_available(torch_module)

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested == "cuda" and not cuda_available:
        raise DeviceResolutionError("Requested device 'cuda' is not available in the current environment.")
    if requested == "mps" and not mps_available:
        raise DeviceResolutionError("Requested device 'mps' is not available in the current environment.")

    return requested
