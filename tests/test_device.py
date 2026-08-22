from types import SimpleNamespace

import pytest

from app.device import DeviceUnavailableError, resolve_device


class FakeCuda:
    def __init__(self, available: bool, count: int = 0):
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


class FakeMps:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self) -> bool:
        return self.available


def fake_torch(cuda: bool, mps: bool, cuda_count: int = 0) -> object:
    return SimpleNamespace(
        cuda=FakeCuda(cuda, cuda_count),
        backends=SimpleNamespace(mps=FakeMps(mps)),
    )


def test_auto_device_priority_is_cuda_then_mps_then_cpu() -> None:
    assert resolve_device("auto", fake_torch(True, True, 1)) == "cuda"
    assert resolve_device("auto", fake_torch(False, True)) == "mps"
    assert resolve_device("auto", fake_torch(False, False)) == "cpu"


def test_explicit_unavailable_device_fails_clearly() -> None:
    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        resolve_device("cuda", fake_torch(False, False))
    with pytest.raises(DeviceUnavailableError, match="MPS"):
        resolve_device("mps", fake_torch(False, False))


def test_cuda_index_is_validated() -> None:
    assert resolve_device("cuda:1", fake_torch(True, False, 2)) == "cuda:1"
    with pytest.raises(DeviceUnavailableError, match="не найдено"):
        resolve_device("cuda:2", fake_torch(True, False, 2))
