"""GLiDRE — zero-shot document-level relation extraction.

GLiDRE requires pre-identified entity mentions. Character offsets use an
end-exclusive convention, just like normal Python slicing.

Run from its isolated environment:
    .venv-glidre/bin/bentoml serve services.glidre:GLiDREService -p 3003
"""

from __future__ import annotations

import bentoml
from bentoml.models import HuggingFaceModel
from pydantic import BaseModel, Field

with bentoml.importing():
    from glidre import GLiDRE

image = bentoml.images.Image(python_version="3.13").requirements_file("requirements-glidre.txt")


class MentionSpan(BaseModel):
    value: str = Field(description="Exact text of this mention")
    start: int = Field(ge=0, description="Zero-based, inclusive character offset")
    end: int = Field(gt=0, description="Zero-based, exclusive character offset")


class EntityMention(BaseModel):
    id: int | str = Field(description="Stable entity identifier shared by coreferent mentions")
    type: str | None = Field(default=None, description="Optional entity type, such as PER or ORG")
    mentions: list[MentionSpan]


@bentoml.service(
    image=image,
    resources={"cpu": "4"},
    traffic={"timeout": 300},
)
class GLiDREService:
    """Document-level RE over caller-supplied, coreference-grouped mentions."""

    model_path = HuggingFaceModel("cea-list-ia/glidre_large")

    def __init__(self) -> None:
        self.model = GLiDRE.from_pretrained(self.model_path, map_location="cpu")

    @staticmethod
    def _validate_mentions(text: str, mentions: list[EntityMention]) -> list[dict]:
        serialized = []
        for entity in mentions:
            entity_data = entity.model_dump(exclude_none=True)
            for mention in entity.mentions:
                actual = text[mention.start : mention.end]
                if actual != mention.value:
                    raise ValueError(
                        f"Mention {entity.id!r} offsets [{mention.start}, {mention.end}) "
                        f"select {actual!r}, not {mention.value!r}"
                    )
            serialized.append(entity_data)
        return serialized

    @bentoml.api
    def extract_relations(
        self,
        text: str,
        relations: list[str],
        mentions: list[EntityMention],
        threshold: float = 0.3,
        multi_label: bool = False,
    ) -> dict:
        """Classify relations between supplied entities across a document."""
        mention_data = self._validate_mentions(text, mentions)
        extracted = self.model.predict_entities(
            text=text,
            labels=relations,
            mentions=mention_data,
            threshold=threshold,
            multi_label=multi_label,
        )
        return {"relations": extracted}
