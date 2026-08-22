from dataclasses import replace
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from app.config import settings
from app.gemini import GeminiResponseError, GeminiService
from app.models import CharacterPlan, DeduplicationPlan
from app.video import ReferenceCandidate


def candidate(reference_id: str, track_id: int, score: float) -> ReferenceCandidate:
    return ReferenceCandidate(
        reference_id=reference_id,
        track_id=track_id,
        frame_index=1,
        timestamp_seconds=0.1,
        quality_score=score,
        path=Path(f"{reference_id}.jpg"),
    )


def test_validate_plan_removes_hallucinated_ids_and_uses_fallback() -> None:
    local_settings = replace(settings, max_references_per_character=2)
    candidates = [
        candidate("t0001_r01", 1, 0.9),
        candidate("t0001_r02", 1, 0.8),
        candidate("t0002_r01", 2, 0.7),
    ]
    plan = DeduplicationPlan(
        characters=[
            CharacterPlan(
                character_id="made_up",
                track_ids=[1, 99],
                reference_ids=["hallucinated"],
                appearance="dark jacket",
                media_style="photorealistic",
                character_form="human",
                style_description="live-action footage",
                confidence=0.8,
            )
        ]
    )

    validated = GeminiService.validate_plan(plan, candidates, local_settings)

    assert validated.characters[0].character_id == "character_01"
    assert validated.characters[0].track_ids == [1]
    assert validated.characters[0].reference_ids == ["t0001_r01", "t0001_r02"]


def test_validate_plan_limits_references() -> None:
    local_settings = replace(settings, max_references_per_character=1)
    candidates = [
        candidate("t0001_r01", 1, 0.9),
        candidate("t0001_r02", 1, 0.8),
    ]
    plan = DeduplicationPlan(
        characters=[
            CharacterPlan(
                character_id="person",
                track_ids=[1],
                reference_ids=["t0001_r01", "t0001_r02"],
                appearance="blue coat",
                media_style="photorealistic",
                character_form="human",
                style_description="live-action footage",
                confidence=0.9,
            )
        ]
    )

    validated = GeminiService.validate_plan(plan, candidates, local_settings)
    assert validated.characters[0].reference_ids == ["t0001_r01"]


def test_generate_character_decodes_sdk_bytes_and_writes_real_png(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jpg"
    Image.new("RGB", (16, 16), "white").save(reference_path)
    reference = ReferenceCandidate(
        reference_id="t0001_r01",
        track_id=1,
        frame_index=1,
        timestamp_seconds=0.1,
        quality_score=0.9,
        path=reference_path,
    )
    character = CharacterPlan(
        character_id="character_01",
        track_ids=[1],
        reference_ids=[reference.reference_id],
        appearance="black jacket",
        media_style="photorealistic",
        character_form="human",
        style_description="live-action footage",
        confidence=0.9,
    )

    generated_buffer = BytesIO()
    Image.new("RGB", (12, 18), "black").save(generated_buffer, format="JPEG")

    class FakeGeneratedImage:
        image_bytes = generated_buffer.getvalue()

    class FakePart:
        inline_data = object()

        @staticmethod
        def as_image() -> FakeGeneratedImage:
            return FakeGeneratedImage()

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs: object) -> object:
            return type("Response", (), {"parts": [FakePart()]})()

    service = GeminiService(replace(settings, google_api_key="test-key"))
    service._client = type("Client", (), {"models": FakeModels()})()
    output_path = tmp_path / "character.png"

    service.generate_character(
        character,
        {reference.reference_id: reference},
        output_path,
    )

    with Image.open(output_path) as result:
        assert result.format == "PNG"
        assert result.size == (12, 18)


def test_image_generation_falls_back_when_optional_config_is_unsupported(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.jpg"
    Image.new("RGB", (16, 16), "white").save(reference_path)
    reference = ReferenceCandidate(
        reference_id="t0001_r01",
        track_id=1,
        frame_index=1,
        timestamp_seconds=0.1,
        quality_score=0.9,
        path=reference_path,
    )
    character = CharacterPlan(
        character_id="character_01",
        track_ids=[1],
        reference_ids=[reference.reference_id],
        appearance="gray coat",
        media_style="photorealistic",
        character_form="human",
        style_description="live-action footage",
        confidence=0.9,
    )
    generated_buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(generated_buffer, format="PNG")

    class UnsupportedConfigError(Exception):
        code = 400

    class FakeGeneratedImage:
        image_bytes = generated_buffer.getvalue()

    class FakePart:
        inline_data = object()

        @staticmethod
        def as_image() -> FakeGeneratedImage:
            return FakeGeneratedImage()

    class FakeModels:
        calls = 0

        @classmethod
        def generate_content(cls, **_kwargs: object) -> object:
            cls.calls += 1
            if cls.calls == 1:
                raise UnsupportedConfigError("image_size parameter is not supported")
            return type("Response", (), {"parts": [FakePart()]})()

    service = GeminiService(replace(settings, google_api_key="test-key"))
    service._client = type("Client", (), {"models": FakeModels()})()
    output_path = tmp_path / "character.png"

    service.generate_character(
        character,
        {reference.reference_id: reference},
        output_path,
    )

    assert FakeModels.calls == 2
    with Image.open(output_path) as result:
        assert result.format == "PNG"


def test_generate_character_reads_documented_inline_data(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jpg"
    Image.new("RGB", (16, 16), "white").save(reference_path)
    reference = ReferenceCandidate(
        reference_id="t0001_r01",
        track_id=1,
        frame_index=1,
        timestamp_seconds=0.1,
        quality_score=0.9,
        path=reference_path,
    )
    character = CharacterPlan(
        character_id="character_01",
        track_ids=[1],
        reference_ids=[reference.reference_id],
        appearance="blue coat",
        media_style="photorealistic",
        character_form="human",
        style_description="live-action footage",
        confidence=0.9,
    )
    generated_buffer = BytesIO()
    Image.new("RGB", (10, 20), "blue").save(generated_buffer, format="JPEG")

    class FakePart:
        inline_data = type("InlineData", (), {"data": generated_buffer.getvalue()})()
        thought = False

        @staticmethod
        def as_image() -> None:
            return None

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs: object) -> object:
            return type("Response", (), {"parts": [FakePart()]})()

    service = GeminiService(replace(settings, google_api_key="test-key"))
    service._client = type("Client", (), {"models": FakeModels()})()
    output_path = tmp_path / "character.png"
    service.generate_character(character, {reference.reference_id: reference}, output_path)

    with Image.open(output_path) as result:
        assert result.format == "PNG"
        assert result.size == (10, 20)


def test_text_only_image_response_reports_model_reason(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jpg"
    Image.new("RGB", (16, 16), "white").save(reference_path)
    reference = ReferenceCandidate(
        reference_id="t0001_r01",
        track_id=1,
        frame_index=1,
        timestamp_seconds=0.1,
        quality_score=0.9,
        path=reference_path,
    )
    character = CharacterPlan(
        character_id="character_01",
        track_ids=[1],
        reference_ids=[reference.reference_id],
        appearance="blue coat",
        media_style="photorealistic",
        character_form="human",
        style_description="live-action footage",
        confidence=0.9,
    )

    class TextPart:
        inline_data = None
        text = "The image request could not be completed."
        thought = False

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs: object) -> object:
            return type("Response", (), {"parts": [TextPart()], "candidates": []})()

    service = GeminiService(replace(settings, google_api_key="test-key"))
    service._client = type("Client", (), {"models": FakeModels()})()

    with pytest.raises(GeminiResponseError, match="could not be completed"):
        service.generate_character(
            character,
            {reference.reference_id: reference},
            tmp_path / "character.png",
        )


def test_compatibility_fallback_does_not_mask_auth_or_quota_errors() -> None:
    class ApiError(Exception):
        def __init__(self, code: int, message: str):
            super().__init__(message)
            self.code = code

    assert GeminiService._is_unsupported_config_error(
        ApiError(400, "image_size parameter is not supported")
    )
    assert not GeminiService._is_unsupported_config_error(
        ApiError(401, "API key is invalid")
    )
    assert not GeminiService._is_unsupported_config_error(
        ApiError(429, "Quota exceeded")
    )


def test_generation_prompt_preserves_2d_nonhuman_style() -> None:
    character = CharacterPlan(
        character_id="character_01",
        track_ids=[1],
        reference_ids=["t0001_r01"],
        appearance="blue quadruped with a long striped tail",
        media_style="2d_animation",
        character_form="creature",
        style_description="flat cel shading and thick black outlines",
        confidence=0.95,
    )

    prompt = GeminiService._generation_prompt(character)

    assert "must remain 2D" in prompt
    assert "Do not photorealize" in prompt
    assert "Do not add a human body" in prompt
    assert "tail" in prompt
