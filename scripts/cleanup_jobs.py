from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.models import JobStatus, utc_now  # noqa: E402
from app.storage import JobStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Очистка старых завершённых заданий")
    parser.add_argument("--hours", type=int, default=settings.job_retention_hours)
    parser.add_argument(
        "--apply", action="store_true", help="действительно удалить найденные задания"
    )
    args = parser.parse_args()
    if args.hours <= 0:
        raise SystemExit("--hours должен быть больше нуля")

    store = JobStore(settings.data_dir)
    cutoff = utc_now() - timedelta(hours=args.hours)
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED}
    candidates = [
        record
        for record in store.list_records()
        if record.status in terminal and record.updated_at < cutoff
    ]
    for record in candidates:
        print(f"{record.job_id} {record.status.value} {record.updated_at.isoformat()}")
    if not args.apply:
        print(f"Dry run: {len(candidates)} job(s). Добавьте --apply для удаления.")
        return 0
    removed = store.cleanup_expired(args.hours)
    print(f"Удалено заданий: {len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
