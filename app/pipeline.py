from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from app.config import Settings
from app.device import DeviceUnavailableError
from app.gemini import (
    GeminiConfigurationError,
    GeminiResponseError,
    GeminiService,
)
from app.models import CharacterArtifact, DeduplicationPlan, JobStatus
from app.storage import JobStore
from app.video import VideoAnalysis, VideoAnalyzer, VideoValidationError
from app.weights import MissingWeightsError


ProgressCallback = Callable[[JobStatus, str, int], None]
logger = logging.getLogger(__name__)


class CharacterPipeline:
    def __init__(
        self,
        settings: Settings,
        analyzer: VideoAnalyzer | None = None,
        gemini: GeminiService | None = None,
    ):
        self.settings = settings
        self.analyzer = analyzer or VideoAnalyzer(settings)
        self.gemini = gemini or GeminiService(settings)

    def run(
        self,
        video_path: Path,
        job_dir: Path,
        progress: ProgressCallback,
    ) -> tuple[VideoAnalysis, list[CharacterArtifact]]:
        progress(JobStatus.ANALYZING, "Поиск и трекинг персонажей", 10)
        analysis = self.analyzer.analyze(video_path, job_dir / "references")

        progress(JobStatus.DEDUPLICATING, "Дедупликация и выбор референсов", 55)
        plan = self.gemini.deduplicate(analysis)
        self._write_analysis(job_dir, analysis, plan)

        candidate_map = {
            candidate.reference_id: candidate for candidate in analysis.all_candidates
        }
        artifacts: list[CharacterArtifact] = []
        character_count = len(plan.characters)
        for index, character in enumerate(plan.characters, start=1):
            progress_value = 60 + round(35 * (index - 1) / character_count)
            progress(
                JobStatus.GENERATING,
                f"Генерация персонажа {index} из {character_count}",
                progress_value,
            )
            output_path = job_dir / "output" / f"{character.character_id}.png"
            self.gemini.generate_character(character, candidate_map, output_path)
            artifacts.append(
                CharacterArtifact(
                    character_id=character.character_id,
                    track_ids=character.track_ids,
                    reference_ids=character.reference_ids,
                    appearance=character.appearance,
                    confidence=character.confidence,
                    image_file=str(output_path),
                    image_url=(
                        f"/v1/jobs/{job_dir.name}/characters/"
                        f"{character.character_id}.png"
                    ),
                    media_style=character.media_style,
                    character_form=character.character_form,
                )
            )
        return analysis, artifacts

    @staticmethod
    def _write_analysis(
        job_dir: Path, analysis: VideoAnalysis, plan: DeduplicationPlan
    ) -> None:
        payload = {
            "video": {
                "width": analysis.width,
                "height": analysis.height,
                "fps": analysis.fps,
                "frame_count": analysis.frame_count,
                "duration_seconds": analysis.duration_seconds,
            },
            "tracks": {
                str(track_id): [
                    {
                        "reference_id": candidate.reference_id,
                        "frame_index": candidate.frame_index,
                        "timestamp_seconds": candidate.timestamp_seconds,
                        "quality_score": candidate.quality_score,
                        "detection_label": candidate.detection_label,
                        "file": str(candidate.path),
                    }
                    for candidate in candidates
                ]
                for track_id, candidates in analysis.candidates.items()
            },
            "deduplication": plan.model_dump(mode="json"),
        }
        (job_dir / "analysis.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class JobRunner:
    def __init__(
        self, settings: Settings, store: JobStore, pipeline: CharacterPipeline
    ):
        self.settings = settings
        self.store = store
        self.pipeline = pipeline

    def run(self, job_id: str) -> None:
        record = self.store.get(job_id)
        self.store.update(
            job_id,
            attempts=record.attempts + 1,
            error=None,
            error_code=None,
        )

        def progress(status: JobStatus, stage: str, value: int) -> None:
            self.store.update(job_id, status=status, stage=stage, progress=value)

        try:
            analysis, artifacts = self.pipeline.run(
                Path(record.input_file), self.store.job_dir(job_id), progress
            )
            self.store.update(
                job_id,
                status=JobStatus.COMPLETED,
                stage="Готово",
                progress=100,
                characters=artifacts,
                video={
                    "width": analysis.width,
                    "height": analysis.height,
                    "fps": round(analysis.fps, 3),
                    "frame_count": analysis.frame_count,
                    "duration_seconds": round(analysis.duration_seconds, 3),
                },
            )
        except Exception as error:
            logger.exception("Job %s failed", job_id)
            error_code, public_message = self._public_error(error)
            self.store.update(
                job_id,
                status=JobStatus.FAILED,
                stage="Ошибка обработки",
                error=public_message,
                error_code=error_code,
            )

    @staticmethod
    def _public_error(error: Exception) -> tuple[str, str]:
        if isinstance(error, VideoValidationError):
            return "invalid_video", str(error)
        if isinstance(error, MissingWeightsError):
            return "weights_missing", str(error)
        if isinstance(error, DeviceUnavailableError):
            return "device_unavailable", str(error)
        if isinstance(error, GeminiConfigurationError):
            return "gemini_not_configured", str(error)
        if isinstance(error, GeminiResponseError):
            return "gemini_invalid_response", str(error)

        status_code = getattr(error, "code", None)
        if status_code in {401, 403}:
            return "gemini_auth", "Google API отклонил ключ или доступ к модели"
        if status_code == 429:
            return "gemini_quota", "Исчерпана квота Google API; повторите позже"
        if isinstance(status_code, int) and status_code >= 500:
            return "gemini_unavailable", "Google API временно недоступен"
        return "internal_error", "Внутренняя ошибка обработки; подробности в логе"
