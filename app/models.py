from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    """Python 3.10-compatible string enum."""

    QUEUED = "queued"
    ANALYZING = "analyzing"
    DEDUPLICATING = "deduplicating"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class CharacterPlan(BaseModel):
    character_id: str = Field(description="Stable ID such as character_01")
    track_ids: list[int] = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1, max_length=4)
    appearance: str = Field(
        description="Precise description of every visible identity-defining trait"
    )
    media_style: Literal[
        "photorealistic",
        "2d_animation",
        "3d_animation",
        "stop_motion",
        "other_stylized",
    ]
    character_form: Literal[
        "human", "humanoid", "animal", "creature", "robot", "other"
    ]
    style_description: str = Field(
        description="Visual medium, rendering, linework, shading and proportions"
    )
    confidence: float = Field(ge=0, le=1)


class DeduplicationPlan(BaseModel):
    characters: list[CharacterPlan] = Field(min_length=1, max_length=3)


class CharacterArtifact(BaseModel):
    character_id: str
    track_ids: list[int]
    reference_ids: list[str]
    appearance: str
    confidence: float
    image_file: str
    image_url: str
    media_style: str = "photorealistic"
    character_form: str = "human"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    stage: str
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    input_file: str
    error: str | None = None
    error_code: str | None = None
    attempts: int = Field(default=0, ge=0)
    characters: list[CharacterArtifact] = Field(default_factory=list)
    video: dict[str, int | float] | None = None

    @classmethod
    def queued(cls, job_id: str, input_file: Path) -> "JobRecord":
        now = utc_now()
        return cls(
            job_id=job_id,
            status=JobStatus.QUEUED,
            stage="Видео принято",
            progress=0,
            created_at=now,
            updated_at=now,
            input_file=str(input_file),
        )


class HealthResponse(BaseModel):
    status: str
    google_api_configured: bool
    analysis_model: str
    image_model: str
    detector_model: str
    detection_mode: str
    device: str
    requested_device: str
    queue_mode: str
    weights_ready: bool
