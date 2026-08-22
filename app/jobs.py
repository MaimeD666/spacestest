from __future__ import annotations

import asyncio
import logging

from app.pipeline import JobRunner
from app.storage import JobStore


logger = logging.getLogger(__name__)


class JobManager:
    """One in-process queue that serializes memory-heavy ML jobs."""

    def __init__(self, store: JobStore, runner: JobRunner, retention_hours: int):
        self.store = store
        self.runner = runner
        self.retention_hours = retention_hours
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._pending: set[str] = set()

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        if self.running:
            return
        self._queue = asyncio.Queue()
        try:
            removed = self.store.cleanup_expired(self.retention_hours)
            if removed:
                logger.info("Removed %d expired jobs", len(removed))
        except OSError:
            logger.exception("Could not clean expired jobs")

        recovered = self.store.recover_incomplete()
        self._worker = asyncio.create_task(self._work(), name="ml-job-worker")
        for record in recovered:
            await self.submit(record.job_id)
        if recovered:
            logger.info("Requeued %d interrupted jobs", len(recovered))

    async def stop(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._queue = None
        self._pending.clear()

    async def submit(self, job_id: str) -> None:
        if self._queue is None or not self.running:
            raise RuntimeError("Очередь обработки не запущена")
        if job_id in self._pending:
            return
        self._pending.add(job_id)
        await self._queue.put(job_id)

    async def _work(self) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                await asyncio.to_thread(self.runner.run, job_id)
            except Exception:
                # JobRunner normally records failures itself. This protects the
                # queue from an unexpected storage/runner failure.
                logger.exception("Unhandled worker error for %s", job_id)
            finally:
                self._pending.discard(job_id)
                self._queue.task_done()
