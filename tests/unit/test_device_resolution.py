from __future__ import annotations

import types

import pytest

from gui_detector_api.utils.device import DeviceResolutionError, resolve_device


def _build_torch_module(*, cuda_available: bool, mps_available: bool):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(
                is_available=lambda: mps_available,
                is_built=lambda: mps_available,
            )
        ),
    )


def test_resolve_device_prefers_cuda_for_auto():
    torch_module = _build_torch_module(cuda_available=True, mps_available=True)
    assert resolve_device("auto", torch_module=torch_module) == "cuda"


def test_resolve_device_uses_mps_when_cuda_is_unavailable():
    torch_module = _build_torch_module(cuda_available=False, mps_available=True)
    assert resolve_device("auto", torch_module=torch_module) == "mps"


def test_resolve_device_falls_back_to_cpu():
    torch_module = _build_torch_module(cuda_available=False, mps_available=False)
    assert resolve_device("auto", torch_module=torch_module) == "cpu"


def test_resolve_device_rejects_unavailable_mps():
    torch_module = _build_torch_module(cuda_available=False, mps_available=False)
    with pytest.raises(DeviceResolutionError):
        resolve_device("mps", torch_module=torch_module)


def test_resolve_device_rejects_unavailable_cuda():
    torch_module = _build_torch_module(cuda_available=False, mps_available=True)
    with pytest.raises(DeviceResolutionError):
        resolve_device("cuda", torch_module=torch_module)
