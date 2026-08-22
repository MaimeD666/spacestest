from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import Settings
from app.device import resolve_device
from app.weights import ensure_weights


class VideoValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    reference_id: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    quality_score: float
    path: Path
    detection_label: str = "character"


@dataclass(frozen=True, slots=True)
class VideoAnalysis:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    candidates: dict[int, list[ReferenceCandidate]]

    @property
    def all_candidates(self) -> list[ReferenceCandidate]:
        return [candidate for group in self.candidates.values() for candidate in group]


@dataclass(slots=True)
class _BufferedCandidate:
    track_id: int
    frame_index: int
    timestamp_seconds: float
    quality_score: float
    encoded_jpeg: bytes
    detection_label: str


class VideoAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._resolved_device: str | None = None

    def analyze(self, video_path: Path, references_dir: Path) -> VideoAnalysis:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise VideoValidationError("Не удалось открыть MP4-файл")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
                raise VideoValidationError("У видео некорректные FPS или число кадров")
            duration = frame_count / fps
            if duration > self.settings.max_video_seconds:
                raise VideoValidationError(
                    f"Видео длится {duration:.1f} с; максимум — "
                    f"{self.settings.max_video_seconds} с"
                )
            if width < 160 or height < 160:
                raise VideoValidationError("Минимальный размер видео — 160×160")

            candidates, observations = self._track_frames(capture, fps, width, height)
        finally:
            capture.release()

        kept_tracks = {
            track_id: items
            for track_id, items in candidates.items()
            if observations[track_id] >= 2 and items
        }
        if not kept_tracks:
            raise VideoValidationError("Персонажи в видео не найдены")

        saved = self._save_candidates(kept_tracks, references_dir)
        return VideoAnalysis(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            candidates=saved,
        )

    def _track_frames(
        self,
        capture: cv2.VideoCapture,
        fps: float,
        width: int,
        height: int,
    ) -> tuple[dict[int, list[_BufferedCandidate]], dict[int, int]]:
        model = self._load_model()
        sample_step = max(1, round(fps / self.settings.analysis_fps))
        candidates: dict[int, list[_BufferedCandidate]] = defaultdict(list)
        observations: dict[int, int] = defaultdict(int)
        frame_index = -1

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % sample_step:
                continue

            track_options: dict[str, Any] = {
                "persist": True,
                "tracker": str(self.settings.tracker_config),
                "conf": self.settings.detection_confidence,
                "imgsz": 640,
                "device": self._device(),
                "verbose": False,
            }
            if self.settings.detection_mode == "people":
                track_options["classes"] = [0]
            else:
                # Multiple text prompts can describe the same stylized character.
                # Class-agnostic suppression avoids duplicate boxes for one object.
                track_options["agnostic_nms"] = True
            result = model.track(
                frame,
                **track_options,
            )[0]
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            xyxy_values = boxes.xyxy.detach().cpu().numpy()
            id_values = boxes.id.detach().cpu().numpy().astype(int)
            confidence_values = boxes.conf.detach().cpu().numpy()
            class_values = boxes.cls.detach().cpu().numpy().astype(int)
            for xyxy, track_id, confidence, class_id in zip(
                xyxy_values,
                id_values,
                confidence_values,
                class_values,
                strict=True,
            ):
                observations[int(track_id)] += 1
                crop = self._crop_character(frame, xyxy)
                if crop is None:
                    continue
                quality = self.quality_score(
                    crop, xyxy, float(confidence), width, height
                )
                ok, encoded = cv2.imencode(
                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94]
                )
                if not ok:
                    continue
                if isinstance(result.names, dict):
                    detection_label = str(
                        result.names.get(int(class_id), "character")
                    )
                elif 0 <= int(class_id) < len(result.names):
                    detection_label = str(result.names[int(class_id)])
                else:
                    detection_label = "character"
                item = _BufferedCandidate(
                    track_id=int(track_id),
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    quality_score=quality,
                    encoded_jpeg=encoded.tobytes(),
                    detection_label=detection_label,
                )
                self._add_diverse_candidate(candidates[int(track_id)], item, fps)

        return dict(candidates), dict(observations)

    def _load_model(self) -> Any:
        if self._model is None:
            ensure_weights(self.settings)
            model_name = (
                self.settings.yolo_universal_model
                if self.settings.detection_mode == "universal"
                else self.settings.yolo_model
            )
            model_path = self.settings.weight_path(model_name)
            if self.settings.detection_mode == "universal":
                from ultralytics import YOLOE, settings as ultralytics_settings

                # MobileCLIP is loaded by Ultralytics through its weights_dir.
                # Point it at our explicit local directory before set_classes so
                # inference never silently downloads a model at runtime.
                ultralytics_settings.update(
                    {"weights_dir": str(self.settings.weights_dir)}
                )
                self._model = YOLOE(str(model_path))
                self._model.set_classes(list(self.settings.character_prompts))
            else:
                from ultralytics import YOLO

                self._model = YOLO(str(model_path))
        return self._model

    def _device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = resolve_device(self.settings.device)
        return self._resolved_device

    @staticmethod
    def _crop_character(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        box_width, box_height = x2 - x1, y2 - y1
        # Do not assume human proportions: animated characters can be wide,
        # quadrupedal, floating, serpentine or very short.
        if min(box_width, box_height) < 24 or box_width * box_height < 2500:
            return None
        pad_x, pad_y = box_width * 0.06, box_height * 0.04
        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))
        crop = frame[top:bottom, left:right]
        return crop if crop.size else None

    @staticmethod
    def quality_score(
        crop: np.ndarray,
        xyxy: np.ndarray,
        confidence: float,
        frame_width: int,
        frame_height: int,
    ) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(1.0, math.log1p(laplacian_variance) / math.log1p(1200))

        x1, y1, x2, y2 = (float(value) for value in xyxy)
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / (
            frame_width * frame_height
        )
        size_score = min(1.0, math.sqrt(area_ratio / 0.35))

        margin_x = frame_width * 0.015
        margin_y = frame_height * 0.015
        unclipped_edges = sum(
            (
                x1 > margin_x,
                y1 > margin_y,
                x2 < frame_width - margin_x,
                y2 < frame_height - margin_y,
            )
        ) / 4

        mean_luma = float(gray.mean()) / 255
        exposure = max(0.0, 1.0 - abs(mean_luma - 0.5) / 0.5)
        score = (
            0.40 * sharpness
            + 0.25 * size_score
            + 0.20 * unclipped_edges
            + 0.10 * max(0.0, min(1.0, confidence))
            + 0.05 * exposure
        )
        return round(score, 5)

    def _add_diverse_candidate(
        self,
        items: list[_BufferedCandidate],
        candidate: _BufferedCandidate,
        fps: float,
    ) -> None:
        min_gap = max(1, round(fps * 0.30))
        close = next(
            (
                existing
                for existing in items
                if abs(existing.frame_index - candidate.frame_index) < min_gap
            ),
            None,
        )
        if close is not None:
            if candidate.quality_score > close.quality_score:
                items.remove(close)
                items.append(candidate)
        else:
            items.append(candidate)

        items.sort(key=lambda item: item.quality_score, reverse=True)
        del items[self.settings.max_candidates_per_track :]

    @staticmethod
    def _save_candidates(
        groups: dict[int, list[_BufferedCandidate]], references_dir: Path
    ) -> dict[int, list[ReferenceCandidate]]:
        references_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[int, list[ReferenceCandidate]] = {}
        for track_id, items in sorted(groups.items()):
            track_dir = references_dir / f"track_{track_id:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)
            saved[track_id] = []
            for rank, item in enumerate(items, start=1):
                reference_id = f"t{track_id:04d}_r{rank:02d}"
                path = track_dir / f"{reference_id}.jpg"
                path.write_bytes(item.encoded_jpeg)
                saved[track_id].append(
                    ReferenceCandidate(
                        reference_id=reference_id,
                        track_id=track_id,
                        frame_index=item.frame_index,
                        timestamp_seconds=round(item.timestamp_seconds, 3),
                        quality_score=item.quality_score,
                        path=path,
                        detection_label=item.detection_label,
                    )
                )
        return saved
