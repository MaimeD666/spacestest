from dataclasses import replace
from pathlib import Path

from PIL import Image

from app.config import settings
from app.models import CharacterPlan, DeduplicationPlan, JobRecord, JobStatus
from app.pipeline import CharacterPipeline, JobRunner
from app.storage import JobStore
from app.video import ReferenceCandidate, VideoAnalysis


class FakeAnalyzer:
    def analyze(self, _video_path: Path, references_dir: Path) -> VideoAnalysis:
        reference_path = references_dir / "track_0001" / "t0001_r01.jpg"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 48), "navy").save(reference_path)
        candidate = ReferenceCandidate(
            reference_id="t0001_r01",
            track_id=1,
            frame_index=2,
            timestamp_seconds=0.2,
            quality_score=0.9,
            path=reference_path,
            detection_label="person",
        )
        return VideoAnalysis(
            width=640,
            height=480,
            fps=10.0,
            frame_count=50,
            duration_seconds=5.0,
            candidates={1: [candidate]},
        )


class FakeGemini:
    def deduplicate(self, _analysis: VideoAnalysis) -> DeduplicationPlan:
        return DeduplicationPlan(
            characters=[
                CharacterPlan(
                    character_id="character_01",
                    track_ids=[1],
                    reference_ids=["t0001_r01"],
                    appearance="dark blue outfit",
                    media_style="photorealistic",
                    character_form="human",
                    style_description="live-action",
                    confidence=0.9,
                )
            ]
        )

    def generate_character(
        self,
        _character: CharacterPlan,
        _references: dict[str, ReferenceCandidate],
        output_path: Path,
    ) -> None:
        Image.new("RGB", (64, 96), "navy").save(output_path, format="PNG")


def test_job_runner_completes_full_offline_pipeline(tmp_path: Path) -> None:
    local_settings = replace(settings, data_dir=tmp_path / "jobs")
    store = JobStore(local_settings.data_dir)
    job_id = "a" * 32
    input_path = store.job_dir(job_id) / "input.mp4"
    store.create(JobRecord.queued(job_id, input_path))
    input_path.write_bytes(b"offline fixture")
    pipeline = CharacterPipeline(
        local_settings,
        analyzer=FakeAnalyzer(),  # type: ignore[arg-type]
        gemini=FakeGemini(),  # type: ignore[arg-type]
    )

    JobRunner(local_settings, store, pipeline).run(job_id)

    result = store.get(job_id)
    assert result.status == JobStatus.COMPLETED
    assert result.progress == 100
    assert result.attempts == 1
    assert result.error is None
    assert len(result.characters) == 1
    assert Path(result.characters[0].image_file).is_file()
    assert (store.job_dir(job_id) / "analysis.json").is_file()
