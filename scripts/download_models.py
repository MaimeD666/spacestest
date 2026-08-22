from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.weights import WEIGHT_SPECS, sha256_file  # noqa: E402


def selected_names(all_detectors: bool) -> list[str]:
    names = [
        settings.yolo_universal_model
        if settings.detection_mode == "universal"
        else settings.yolo_model
    ]
    if settings.detection_mode == "universal":
        names.append("mobileclip2_b.ts")
    if all_detectors:
        names.extend([settings.yolo_model, settings.yolo_universal_model])
        names.append("mobileclip2_b.ts")
    return list(dict.fromkeys(Path(name).name for name in names))


def download(name: str, force: bool) -> None:
    spec = WEIGHT_SPECS.get(name)
    if spec is None:
        raise SystemExit(
            f"Для {name} нет зафиксированного источника. "
            "Положите этот пользовательский файл в WEIGHTS_DIR вручную."
        )
    target = settings.weights_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not force:
        actual = sha256_file(target)
        if actual == spec.sha256:
            print(f"OK       {target}")
            return
        raise SystemExit(
            f"Контрольная сумма {target} не совпала. "
            "Проверьте файл и повторите с --force для замены."
        )

    temporary = target.with_suffix(target.suffix + ".download")
    temporary.unlink(missing_ok=True)
    print(f"DOWNLOAD {spec.url}")
    request = urllib.request.Request(
        spec.url, headers={"User-Agent": "video-character-extractor/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
        actual = sha256_file(temporary)
        if actual != spec.sha256:
            raise RuntimeError(
                f"SHA-256 не совпал для {name}: ожидался {spec.sha256}, получен {actual}"
            )
        temporary.replace(target)
        print(f"OK       {target}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Явно скачать и проверить локальные веса Ultralytics"
    )
    parser.add_argument(
        "--all", action="store_true", help="скачать веса обоих режимов детекции"
    )
    parser.add_argument(
        "--force", action="store_true", help="заменить уже существующие файлы"
    )
    args = parser.parse_args()
    for name in selected_names(args.all):
        download(name, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
