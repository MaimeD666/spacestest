from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.device import DeviceUnavailableError, resolve_device
from app.jobs import JobManager
from app.models import HealthResponse, JobRecord, JobStatus
from app.pipeline import CharacterPipeline, JobRunner
from app.storage import JobNotFoundError, JobStore
from app.weights import required_weight_paths


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

store = JobStore(settings.data_dir)
pipeline = CharacterPipeline(settings)
runner = JobRunner(settings, store, pipeline)
manager = JobManager(store, runner, settings.job_retention_hours)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await manager.start()
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(
    title="Video Character Extractor",
    description=(
        "Извлечение уникальных персонажей из короткого видео и генерация "
        "полных изображений с сохранением исходного стиля"
    ),
    version="1.0.0",
    lifespan=lifespan,
)
web_dir = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(web_dir / "index.html", media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device_available = True
    try:
        device = resolve_device(settings.device)
    except DeviceUnavailableError as error:
        device = f"unavailable: {error}"
        device_available = False
    weights_ready = all(path.is_file() for path in required_weight_paths(settings))
    ready = bool(settings.google_api_key) and weights_ready and device_available
    return HealthResponse(
        status="ok" if ready else "degraded",
        google_api_configured=bool(settings.google_api_key),
        analysis_model=settings.gemini_analysis_model,
        image_model=settings.gemini_image_model,
        detector_model=(
            settings.yolo_universal_model
            if settings.detection_mode == "universal"
            else settings.yolo_model
        ),
        detection_mode=settings.detection_mode,
        device=device,
        requested_device=settings.device,
        queue_mode="serialized",
        weights_ready=weights_ready,
    )


@app.post(
    "/v1/jobs",
    response_model=JobRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(video: UploadFile = File(...)) -> JobRecord:
    if not settings.google_api_key:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY не задан в локальном .env",
        )
    filename = video.filename or "video.mp4"
    if Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(status_code=415, detail="Поддерживается только MP4")

    job_id = uuid4().hex
    job_dir = store.job_dir(job_id)
    input_path = job_dir / "input.mp4"
    record = JobRecord.queued(job_id, input_path)
    store.create(record)

    size = 0
    temporary_path = input_path.with_suffix(".mp4.uploading")
    try:
        with temporary_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Файл слишком большой; максимум "
                            f"{settings.max_upload_bytes // 1024 // 1024} МБ"
                        ),
                    )
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Загружен пустой файл")
        with temporary_path.open("rb") as uploaded:
            header = uploaded.read(12)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise HTTPException(
                status_code=400,
                detail="Файл не похож на корректный MP4",
            )
        temporary_path.replace(input_path)
    except HTTPException as error:
        temporary_path.unlink(missing_ok=True)
        store.update(
            job_id,
            status=JobStatus.FAILED,
            stage="Ошибка загрузки",
            error=str(error.detail),
            error_code="upload_rejected",
        )
        raise
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        logger.exception("Upload write failed for %s", job_id)
        store.update(
            job_id,
            status=JobStatus.FAILED,
            stage="Ошибка загрузки",
            error="Не удалось сохранить загруженный файл",
            error_code="upload_io_error",
        )
        raise HTTPException(
            status_code=500, detail="Не удалось сохранить загруженный файл"
        ) from error
    finally:
        await video.close()

    try:
        await manager.submit(job_id)
    except RuntimeError as error:
        store.update(
            job_id,
            status=JobStatus.FAILED,
            stage="Очередь недоступна",
            error="Сервис обработки ещё не готов",
            error_code="queue_unavailable",
        )
        raise HTTPException(status_code=503, detail=str(error)) from error
    return store.get(job_id)


@app.get("/v1/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    return _get_job_or_404(job_id)


@app.get("/v1/jobs/{job_id}/characters/{character_id}.png")
def get_character_image(job_id: str, character_id: str) -> FileResponse:
    record = _get_job_or_404(job_id)
    artifact = next(
        (
            item
            for item in record.characters
            if item.character_id == character_id
        ),
        None,
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    path = Path(artifact.image_file).resolve()
    output_dir = (store.job_dir(job_id) / "output").resolve()
    if path.parent != output_dir:
        logger.error("Rejected artifact path outside job output: %s", path)
        raise HTTPException(status_code=404, detail="Файл изображения отсутствует")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Файл изображения отсутствует")
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.delete("/v1/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str) -> Response:
    record = _get_job_or_404(job_id)
    if record.status not in {JobStatus.COMPLETED, JobStatus.FAILED}:
        raise HTTPException(
            status_code=409,
            detail="Нельзя удалить задание во время обработки",
        )
    store.delete(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _get_job_or_404(job_id: str) -> JobRecord:
    if len(job_id) != 32 or any(
        character not in "0123456789abcdef" for character in job_id
    ):
        raise HTTPException(status_code=404, detail="Задание не найдено")
    try:
        return store.get(job_id)
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Задание не найдено") from error
