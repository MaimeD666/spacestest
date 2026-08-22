from __future__ import annotations

from typing import Any


class DeviceUnavailableError(RuntimeError):
    pass


def resolve_device(requested: str, torch_module: Any | None = None) -> str:
    """Resolve auto as CUDA -> MPS -> CPU and validate explicit accelerators."""
    requested = requested.strip().lower()
    if requested == "cpu":
        return "cpu"

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            if requested == "auto":
                return "cpu"
            raise DeviceUnavailableError(
                f"DEVICE={requested} требует установленный PyTorch"
            ) from None

    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not cuda_available:
            raise DeviceUnavailableError("Запрошена CUDA, но она недоступна")
        if ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError:
                raise DeviceUnavailableError("Некорректный индекс CUDA") from None
            if index < 0 or index >= torch_module.cuda.device_count():
                raise DeviceUnavailableError(
                    f"CUDA-устройство {index} не найдено"
                )
        return requested
    if requested == "mps":
        if not mps_available:
            raise DeviceUnavailableError("Запрошен MPS, но он недоступен")
        return "mps"
    raise DeviceUnavailableError(f"Неизвестное устройство: {requested}")
