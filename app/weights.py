from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


@dataclass(frozen=True, slots=True)
class WeightSpec:
    filename: str
    sha256: str
    url: str


ASSET_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
WEIGHT_SPECS = {
    "yolo26n.pt": WeightSpec(
        "yolo26n.pt",
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        f"{ASSET_BASE_URL}/yolo26n.pt",
    ),
    "yoloe-26n-seg.pt": WeightSpec(
        "yoloe-26n-seg.pt",
        "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b",
        f"{ASSET_BASE_URL}/yoloe-26n-seg.pt",
    ),
    "mobileclip2_b.ts": WeightSpec(
        "mobileclip2_b.ts",
        "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f",
        f"{ASSET_BASE_URL}/mobileclip2_b.ts",
    ),
}


class MissingWeightsError(RuntimeError):
    pass


def required_weight_paths(settings: Settings) -> list[Path]:
    model = (
        settings.yolo_universal_model
        if settings.detection_mode == "universal"
        else settings.yolo_model
    )
    paths = [settings.weight_path(model)]
    if settings.detection_mode == "universal":
        paths.append(settings.weights_dir / "mobileclip2_b.ts")
    return paths


def ensure_weights(settings: Settings) -> list[Path]:
    paths = required_weight_paths(settings)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise MissingWeightsError(
            f"Не найдены локальные веса: {names}. "
            "Запустите: python scripts/download_models.py"
        )
    for path in paths:
        spec = WEIGHT_SPECS.get(path.name)
        if spec is not None and sha256_file(path) != spec.sha256:
            raise MissingWeightsError(
                f"Контрольная сумма веса не совпала: {path}. "
                "Проверьте файл или запустите download_models.py --force"
            )
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
