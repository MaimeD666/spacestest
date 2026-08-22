import cv2
import numpy as np

from app.video import VideoAnalyzer


def test_sharp_large_unclipped_crop_scores_higher() -> None:
    rng = np.random.default_rng(42)
    sharp = rng.integers(0, 255, size=(500, 200, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (41, 41), 0)

    sharp_score = VideoAnalyzer.quality_score(
        sharp,
        np.array([200, 100, 400, 600]),
        confidence=0.9,
        frame_width=800,
        frame_height=800,
    )
    blurred_score = VideoAnalyzer.quality_score(
        blurred,
        np.array([0, 0, 100, 250]),
        confidence=0.6,
        frame_width=800,
        frame_height=800,
    )

    assert sharp_score > blurred_score
    assert 0 <= blurred_score <= 1
    assert 0 <= sharp_score <= 1


def test_wide_nonhuman_character_crop_is_accepted() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    crop = VideoAnalyzer._crop_character(
        frame, np.array([40, 100, 360, 145], dtype=float)
    )

    assert crop is not None
    assert crop.shape[1] > crop.shape[0]
