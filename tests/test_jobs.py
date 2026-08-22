import asyncio
import threading
import time
from pathlib import Path

from app.jobs import JobManager
from app.storage import JobStore


class RecordingRunner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.completed: list[str] = []

    def run(self, job_id: str) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
            self.completed.append(job_id)


def test_job_manager_serializes_work(tmp_path: Path) -> None:
    async def scenario() -> tuple[list[str], int]:
        runner = RecordingRunner()
        manager = JobManager(JobStore(tmp_path / "jobs"), runner, 0)  # type: ignore[arg-type]
        await manager.start()
        try:
            await manager.submit("first")
            await manager.submit("second")
            assert manager._queue is not None
            await manager._queue.join()
            return runner.completed, runner.max_active
        finally:
            await manager.stop()

    completed, max_active = asyncio.run(scenario())
    assert completed == ["first", "second"]
    assert max_active == 1
