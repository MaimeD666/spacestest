from pathlib import Path

import pytest

from app.models import JobRecord, JobStatus
from app.storage import JobNotFoundError, JobStore


def test_job_store_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_dir = store.job_dir("abc")
    record = JobRecord.queued("abc", job_dir / "input.mp4")

    store.create(record)
    updated = store.update(
        "abc",
        status=JobStatus.ANALYZING,
        stage="Тест",
        progress=25,
    )

    restored = store.get("abc")
    assert updated.status == JobStatus.ANALYZING
    assert restored.progress == 25
    assert restored.stage == "Тест"
    assert (job_dir / "references").is_dir()
    assert (job_dir / "output").is_dir()


def test_job_store_missing(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    with pytest.raises(JobNotFoundError):
        store.get("missing")


def test_recover_incomplete_job_clears_partial_outputs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = "b" * 32
    job_dir = store.job_dir(job_id)
    input_path = job_dir / "input.mp4"
    store.create(JobRecord.queued(job_id, input_path))
    input_path.write_bytes(b"video")
    (job_dir / "output" / "partial.png").write_bytes(b"partial")
    (job_dir / "analysis.json").write_text("{}", encoding="utf-8")
    store.update(
        job_id,
        status=JobStatus.GENERATING,
        stage="Генерация",
        progress=80,
    )

    recovered = store.recover_incomplete()

    assert [item.job_id for item in recovered] == [job_id]
    restored = store.get(job_id)
    assert restored.status == JobStatus.QUEUED
    assert restored.progress == 0
    assert not (job_dir / "output" / "partial.png").exists()
    assert not (job_dir / "analysis.json").exists()


def test_recover_marks_job_without_input_as_failed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = "c" * 32
    store.create(JobRecord.queued(job_id, store.job_dir(job_id) / "input.mp4"))

    assert store.recover_incomplete() == []
    result = store.get(job_id)
    assert result.status == JobStatus.FAILED
    assert result.error_code == "input_missing"
