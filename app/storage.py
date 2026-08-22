from __future__ import annotations

import json
import shutil
import threading
from datetime import timedelta
from pathlib import Path

from app.models import JobRecord, JobStatus, utc_now


class JobNotFoundError(KeyError):
    pass


class JobStore:
    """Small disk-backed job store suitable for one local server process."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
            raise ValueError("Некорректный ID задания")
        return self.root / job_id

    def create(self, record: JobRecord) -> JobRecord:
        with self._lock:
            directory = self.job_dir(record.job_id)
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "references").mkdir()
            (directory / "output").mkdir()
            self._write(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        state_path = self.job_dir(job_id) / "job.json"
        if not state_path.is_file():
            raise JobNotFoundError(job_id)
        with self._lock:
            return JobRecord.model_validate_json(state_path.read_text("utf-8"))

    def update(self, job_id: str, **changes: object) -> JobRecord:
        with self._lock:
            record = self.get(job_id)
            data = record.model_dump()
            data.update(changes)
            data["updated_at"] = utc_now()
            updated = JobRecord.model_validate(data)
            self._write(updated)
            return updated

    def list_records(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        with self._lock:
            for state_path in sorted(self.root.glob("*/job.json")):
                try:
                    record = JobRecord.model_validate_json(
                        state_path.read_text("utf-8")
                    )
                    if record.job_id == state_path.parent.name:
                        records.append(record)
                except (OSError, ValueError):
                    # A corrupt state must not prevent the service from starting.
                    continue
        return records

    def recover_incomplete(self) -> list[JobRecord]:
        """Reset interrupted work so the serialized worker can retry it."""
        recovered: list[JobRecord] = []
        active = {
            JobStatus.QUEUED,
            JobStatus.ANALYZING,
            JobStatus.DEDUPLICATING,
            JobStatus.GENERATING,
        }
        for record in self.list_records():
            if record.status not in active:
                continue
            input_path = Path(record.input_file)
            if not input_path.is_file():
                self.update(
                    record.job_id,
                    status=JobStatus.FAILED,
                    stage="Восстановление невозможно",
                    error="Исходный MP4 отсутствует",
                    error_code="input_missing",
                )
                continue
            self._clear_derived_files(record.job_id)
            recovered.append(
                self.update(
                    record.job_id,
                    status=JobStatus.QUEUED,
                    stage="Восстановлено после перезапуска",
                    progress=0,
                    error=None,
                    error_code=None,
                    characters=[],
                    video=None,
                )
            )
        return recovered

    def cleanup_expired(self, max_age_hours: int) -> list[str]:
        """Remove only terminal jobs older than the configured retention time."""
        if max_age_hours <= 0:
            return []
        cutoff = utc_now() - timedelta(hours=max_age_hours)
        removed: list[str] = []
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED}
        with self._lock:
            for record in self.list_records():
                if record.status in terminal and record.updated_at < cutoff:
                    shutil.rmtree(self.job_dir(record.job_id), ignore_errors=False)
                    removed.append(record.job_id)
        return removed

    def delete(self, job_id: str) -> None:
        with self._lock:
            directory = self.job_dir(job_id)
            if not (directory / "job.json").is_file():
                raise JobNotFoundError(job_id)
            shutil.rmtree(directory)

    def _clear_derived_files(self, job_id: str) -> None:
        directory = self.job_dir(job_id)
        with self._lock:
            for name in ("references", "output"):
                target = directory / name
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
            analysis = directory / "analysis.json"
            if analysis.exists():
                analysis.unlink()

    def _write(self, record: JobRecord) -> None:
        state_path = self.job_dir(record.job_id) / "job.json"
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(state_path)
