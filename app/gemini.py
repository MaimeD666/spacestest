from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.models import CharacterPlan, DeduplicationPlan
from app.video import ReferenceCandidate, VideoAnalysis


class GeminiConfigurationError(RuntimeError):
    pass


class GeminiResponseError(RuntimeError):
    pass


class GeminiService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Any | None = None

    def deduplicate(self, analysis: VideoAnalysis) -> DeduplicationPlan:
        from google.genai import types

        candidates = analysis.all_candidates
        prompt = self._analysis_prompt(candidates, self.settings.max_characters)
        contents: list[Any] = [prompt]
        opened_images: list[Image.Image] = []
        try:
            for candidate in candidates:
                contents.append(
                    f"Reference ID: {candidate.reference_id}; tracker ID: "
                    f"{candidate.track_id}; detector label: "
                    f"{candidate.detection_label}; local quality: "
                    f"{candidate.quality_score:.3f}"
                )
                image = Image.open(candidate.path)
                opened_images.append(image)
                contents.append(image)

            response = self._get_client().models.generate_content(
                model=self.settings.gemini_analysis_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DeduplicationPlan,
                ),
            )
            if getattr(response, "parsed", None) is not None:
                parsed = response.parsed
                plan = (
                    parsed
                    if isinstance(parsed, DeduplicationPlan)
                    else DeduplicationPlan.model_validate(parsed)
                )
            elif getattr(response, "text", None):
                plan = DeduplicationPlan.model_validate_json(response.text)
            else:
                raise GeminiResponseError("Gemini не вернул план дедупликации")
        finally:
            for image in opened_images:
                image.close()

        return self.validate_plan(plan, candidates, self.settings)

    def generate_character(
        self,
        character: CharacterPlan,
        references: dict[str, ReferenceCandidate],
        output_path: Path,
    ) -> None:
        from google.genai import types

        selected = [references[reference_id] for reference_id in character.reference_ids]
        prompt = self._generation_prompt(character)
        images: list[Image.Image] = []
        try:
            images = [Image.open(candidate.path) for candidate in selected]
            response = self._generate_image_response([prompt, *images], types)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            for part in response.parts or []:
                if getattr(part, "inline_data", None) is not None:
                    generated = part.as_image()
                    if generated is None or generated.image_bytes is None:
                        continue
                    # Google SDK's Image.save() does not accept Pillow options and
                    # may preserve JPEG/WebP bytes under a .png suffix. Decode the
                    # returned bytes with Pillow and always encode a real PNG.
                    with Image.open(BytesIO(generated.image_bytes)) as source:
                        source.save(output_path, format="PNG")
                    return
            raise GeminiResponseError("Gemini не вернул сгенерированное изображение")
        finally:
            for image in images:
                image.close()

    def _generate_image_response(self, contents: list[Any], types: Any) -> Any:
        """Use conservative configs and retry when optional fields are rejected.

        The SDK schema can expose fields that a particular Gemini model or API
        endpoint does not implement. Start with size + aspect ratio, then remove
        optional image settings one level at a time only for a 400 compatibility
        error. Safety, quota, authentication and server errors are never retried.
        """
        configs = [
            types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="3:4", image_size=self.settings.image_size
                ),
            ),
            types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="3:4"),
            ),
            types.GenerateContentConfig(response_modalities=["IMAGE"]),
        ]
        for index, config in enumerate(configs):
            try:
                return self._get_client().models.generate_content(
                    model=self.settings.gemini_image_model,
                    contents=contents,
                    config=config,
                )
            except Exception as error:
                is_last_attempt = index == len(configs) - 1
                if is_last_attempt or not self._is_unsupported_config_error(error):
                    raise
        raise AssertionError("unreachable")

    @staticmethod
    def _is_unsupported_config_error(error: Exception) -> bool:
        if getattr(error, "code", None) != 400:
            return False
        message = str(error).lower()
        markers = (
            "parameter is not supported",
            "not supported in gemini api",
            "unknown name",
            "invalid json payload",
            "image_size",
            "image size",
            "image_config",
            "imageconfig",
            "aspect_ratio",
            "aspect ratio",
        )
        return any(marker in message for marker in markers)

    def _get_client(self) -> Any:
        if not self.settings.google_api_key:
            raise GeminiConfigurationError(
                "GOOGLE_API_KEY не задан. Создайте локальный .env по примеру .env.example"
            )
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.settings.google_api_key)
        return self._client

    @staticmethod
    def validate_plan(
        plan: DeduplicationPlan,
        candidates: list[ReferenceCandidate],
        settings: Settings,
    ) -> DeduplicationPlan:
        by_id = {candidate.reference_id: candidate for candidate in candidates}
        available_tracks = {candidate.track_id for candidate in candidates}
        normalized: list[CharacterPlan] = []
        claimed_references: set[str] = set()
        claimed_tracks: set[int] = set()

        for index, character in enumerate(
            plan.characters[: settings.max_characters], start=1
        ):
            track_ids = sorted(
                (set(character.track_ids) & available_tracks) - claimed_tracks
            )
            if not track_ids:
                continue
            reference_ids: list[str] = []
            for reference_id in character.reference_ids:
                candidate = by_id.get(reference_id)
                if (
                    candidate is not None
                    and candidate.track_id in track_ids
                    and reference_id not in reference_ids
                    and reference_id not in claimed_references
                ):
                    reference_ids.append(reference_id)
                if len(reference_ids) >= settings.max_references_per_character:
                    break

            if not reference_ids:
                fallback = sorted(
                    (candidate for candidate in candidates if candidate.track_id in track_ids),
                    key=lambda candidate: candidate.quality_score,
                    reverse=True,
                )
                reference_ids = [
                    candidate.reference_id
                    for candidate in fallback[: settings.max_references_per_character]
                ]
            claimed_references.update(reference_ids)
            claimed_tracks.update(track_ids)
            normalized.append(
                character.model_copy(
                    update={
                        "character_id": f"character_{index:02d}",
                        "track_ids": track_ids,
                        "reference_ids": reference_ids,
                    }
                )
            )

        if not normalized:
            raise GeminiResponseError("После проверки не осталось ни одного персонажа")
        return DeduplicationPlan(characters=normalized)

    @staticmethod
    def _analysis_prompt(
        candidates: list[ReferenceCandidate], max_characters: int
    ) -> str:
        track_ids = sorted({candidate.track_id for candidate in candidates})
        return f"""
You analyze reference crops of characters extracted from one short video. A
character may be a real human, 2D drawing, anime figure, stylized 3D game model,
stop-motion puppet, robot, animal, fantasy creature, mascot or another non-human
form.
Tracker IDs can be fragmented after occlusion, so two tracker IDs may be the same
character. Group only visually identical characters. Never merge merely because
silhouettes, colors or clothes
look similar. The source clip is expected to contain 2 or 3 key characters. Return
at most {max_characters} unique key characters. Ignore false detections and incidental
background extras; if there are more candidates, prioritize recurring, large and
clearly visible characters. Never invent another character to reach a quota. Available
tracker IDs: {track_ids}.

For each unique character:
1. Return all matching tracker IDs.
2. Choose up to 4 complementary reference IDs. Prefer a visible head or face, sharp
   detail, complete form or outfit, varied useful angles and minimal occlusion. A
   character does not have to be shown completely in any single reference: combine
   useful visible regions across multiple frames.
3. Describe all visible identity-preserving traits appropriate to that form: face
   or head, silhouette, anatomy, limbs, tail, wings, surface/material, colors,
   markings, garments, props and accessories. State which regions are hidden.
4. Classify media_style and character_form using the schema. Describe the exact
   source visual style: linework, cel shading, textures, materials, render quality,
   proportions and level of stylization. Never classify polished 3D animation as a
   real photograph merely because it is realistically rendered.
5. Give confidence from 0 to 1.

Use only supplied tracker IDs and reference IDs. Do not identify or name anyone.
""".strip()

    @staticmethod
    def _generation_prompt(character: CharacterPlan) -> str:
        if character.media_style == "photorealistic":
            style_instruction = """
Create a photorealistic studio-style result. Preserve real-world skin, hair,
materials, anatomy, lens perspective and natural texture. Do not cartoonize.
""".strip()
        else:
            style_instruction = f"""
Preserve the ORIGINAL visual medium exactly: {character.style_description}.
Do not photorealize, live-action-adapt, beautify or redesign the character. A 2D
character must remain 2D with matching linework, palette, cel shading and drawing
proportions. A 3D character must remain 3D with matching materials, topology feel,
lighting language and render style. Preserve deliberate exaggeration and
non-realistic anatomy.
""".strip()

        if character.character_form in {"human", "humanoid"}:
            framing_instruction = """
Show the complete character from the top of the head through the feet or footwear,
standing in a natural pose. Preserve face/head, body proportions, clothing,
patterns and accessories. If hidden lower garments or footwear must be completed,
extend visible design cues conservatively without inventing distinctive details.
""".strip()
        else:
            framing_instruction = """
Show the entire character without cropping: complete head and body plus every
visible limb, paw, hoof, wheel, tail, wing, antenna or other appendage appropriate
to its original anatomy. Do not add a human body, human pose, clothes or shoes to a
non-human character unless supported by the references. Complete hidden regions
conservatively from symmetry and visible design cues.
""".strip()

        return f"""
Create one clean full-character portrait of the SAME character shown in all
reference images. Character form: {character.character_form}.

{style_instruction}

{framing_instruction}

Observed appearance: {character.appearance}
Source style: {character.style_description}

Show exactly one character on a simple neutral background rendered in the same
medium. Use a front or gentle three-quarter view. No crop, no extra characters, no
text, no added logos, no collage and no redesign. Combine identity evidence across
all references; no single reference needs to show the whole character. Preserve
every observed color, marking, material, garment and accessory. Do not invent
distinctive patterns or costume elements unsupported by the references.
""".strip()
