from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "data" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

DEFAULT_CHARACTER_PROMPTS = (
    "character",
    "person",
    "animated character",
    "cartoon character",
    "anime character",
    "3D game character",
    "animal character",
    "fantasy creature",
    "robot",
    "mascot",
    "humanoid",
    "monster",
    "alien",
    "puppet",
    "anthropomorphic object",
    "vehicle character",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} должен быть целым числом") from None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} должен быть числом") from None


def _env_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    google_api_key: str | None
    gemini_analysis_model: str
    gemini_image_model: str
    yolo_model: str
    yolo_universal_model: str
    detection_mode: str
    character_prompts: tuple[str, ...]
    device: str
    analysis_fps: int
    detection_confidence: float
    max_video_seconds: int
    max_upload_bytes: int
    max_candidates_per_track: int
    max_references_per_character: int
    max_characters: int
    image_size: str
    data_dir: Path
    weights_dir: Path
    tracker_config: Path
    job_retention_hours: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("DATA_DIR", "data/jobs")
        weights_dir = _env_path("WEIGHTS_DIR", "data/weights")
        if data_dir in {PROJECT_ROOT, Path(data_dir.anchor)}:
            raise ValueError("DATA_DIR не может быть корнем проекта или файловой системы")

        # GEMINI_API_KEY is also recognized by Google's SDK and kept as a
        # compatibility fallback for existing local setups.
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        detection_mode = os.getenv("DETECTION_MODE", "universal").strip().lower()
        if detection_mode not in {"universal", "people"}:
            raise ValueError("DETECTION_MODE должен быть universal или people")
        prompt_value = os.getenv("CHARACTER_PROMPTS")
        character_prompts = (
            tuple(item.strip() for item in prompt_value.split("|") if item.strip())
            if prompt_value
            else DEFAULT_CHARACTER_PROMPTS
        )
        if not character_prompts:
            raise ValueError("CHARACTER_PROMPTS не может быть пустым")
        detection_confidence = _env_float("DETECTION_CONFIDENCE", 0.20)
        if not 0 <= detection_confidence <= 1:
            raise ValueError("DETECTION_CONFIDENCE должен быть от 0 до 1")
        device = os.getenv("DEVICE", "auto").strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"} and not device.startswith(
            "cuda:"
        ):
            raise ValueError("DEVICE должен быть auto, cpu, cuda, cuda:N или mps")
        return cls(
            google_api_key=api_key,
            gemini_analysis_model=os.getenv(
                "GEMINI_ANALYSIS_MODEL", "gemini-3.1-flash-lite"
            ),
            gemini_image_model=os.getenv(
                "GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"
            ),
            yolo_model=os.getenv("YOLO_MODEL", "yolo26n.pt"),
            yolo_universal_model=os.getenv(
                "YOLO_UNIVERSAL_MODEL", "yoloe-26n-seg.pt"
            ),
            detection_mode=detection_mode,
            character_prompts=character_prompts,
            device=device,
            analysis_fps=max(1, _env_int("ANALYSIS_FPS", 8)),
            detection_confidence=detection_confidence,
            max_video_seconds=max(1, _env_int("MAX_VIDEO_SECONDS", 15)),
            max_upload_bytes=max(1, _env_int("MAX_UPLOAD_MB", 100)) * 1024 * 1024,
            max_candidates_per_track=max(
                1, _env_int("MAX_CANDIDATES_PER_TRACK", 6)
            ),
            max_references_per_character=min(
                4, max(1, _env_int("MAX_REFERENCES_PER_CHARACTER", 4))
            ),
            max_characters=min(3, max(1, _env_int("MAX_CHARACTERS", 3))),
            image_size=os.getenv("IMAGE_SIZE", "2K"),
            data_dir=data_dir,
            weights_dir=weights_dir,
            tracker_config=(
                PROJECT_ROOT
                / "config"
                / (
                    "tracktrack_universal.yaml"
                    if detection_mode == "universal"
                    else "tracktrack_reid.yaml"
                )
            ),
            job_retention_hours=max(0, _env_int("JOB_RETENTION_HOURS", 168)),
        )

    def weight_path(self, configured_name: str) -> Path:
        """Resolve configured model names without depending on the current directory."""
        path = Path(configured_name).expanduser()
        return (path if path.is_absolute() else self.weights_dir / path).resolve()


settings = Settings.from_env()
