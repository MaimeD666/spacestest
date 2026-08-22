from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.device import DeviceUnavailableError, resolve_device  # noqa: E402
from app.weights import WEIGHT_SPECS, required_weight_paths, sha256_file  # noqa: E402


RUNTIME_PACKAGES = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "google-genai",
    "ultralytics",
    "numpy",
    "Pillow",
    "torch",
)


def collect() -> tuple[list[str], list[str]]:
    lines = [
        f"Python: {platform.python_version()} ({platform.system()} {platform.machine()})",
        f"Project: {PROJECT_ROOT}",
    ]
    errors: list[str] = []
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        errors.append("Поддерживается Python 3.10–3.12")

    for package in RUNTIME_PACKAGES:
        try:
            lines.append(f"Package {package}: {version(package)}")
        except PackageNotFoundError:
            errors.append(f"Не установлен пакет {package}")

    for opencv_package in ("opencv-python-headless", "opencv-python"):
        try:
            lines.append(f"Package {opencv_package}: {version(opencv_package)}")
            break
        except PackageNotFoundError:
            continue
    else:
        errors.append("Не установлен пакет OpenCV")

    try:
        lines.append(f"Device: {resolve_device(settings.device)}")
    except DeviceUnavailableError as error:
        errors.append(str(error))

    lines.append(
        "Google API key: configured"
        if settings.google_api_key
        else "Google API key: missing"
    )
    if not settings.google_api_key:
        errors.append("GOOGLE_API_KEY не задан")

    for path in required_weight_paths(settings):
        if path.is_file():
            lines.append(f"Weight: OK {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
            spec = WEIGHT_SPECS.get(path.name)
            if spec is not None and sha256_file(path) != spec.sha256:
                errors.append(f"Неверная контрольная сумма веса {path}")
        else:
            errors.append(f"Не найден вес {path}")
    return lines, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Диагностика локального окружения")
    parser.add_argument(
        "--strict", action="store_true", help="вернуть ненулевой код при проблемах"
    )
    args = parser.parse_args()
    lines, errors = collect()
    print("\n".join(lines))
    if errors:
        print("\nProblems:")
        print("\n".join(f"- {item}" for item in errors))
    else:
        print("\nEnvironment is ready.")
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
