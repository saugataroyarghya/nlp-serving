"""GLiNER2 — multi-task zero-shot information extraction.

Standalone, servable while we're learning it:
    uv run bentoml serve services.entities:GliNER2Service --reload -p 3000

One BentoML Service, one endpoint per native GLiNER2 task, so every capability
can be exercised independently from the Swagger UI (/docs) while testing.
Once this is confirmed working, the entry `service.py` will depend() on this
and expose only `/entities` (extract_entities) per the target architecture.
"""

from __future__ import annotations

import bentoml
from bentoml.models import HuggingFaceModel

with bentoml.importing():
    from gliner2 import AutoExtractor

image = bentoml.images.Image(python_version="3.13").requirements_file("requirements.txt")


@bentoml.service(
    image=image,
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class GliNER2Service:
    model_path = HuggingFaceModel("fastino/gliner2.5-base-v1")

    def __init__(self) -> None:
        self.model = AutoExtractor.from_pretrained(self.model_path)

    @bentoml.api
    def extract_entities(self, text: str, labels: list[str]) -> dict:
        """Zero-shot NER. labels e.g. ["person", "organization", "location"]."""
        return self.model.extract_entities(text, labels)

    @bentoml.api
    def classify_text(self, text: str, labels: dict[str, list[str]]) -> dict:
        """Zero-shot classification. labels e.g. {"sentiment": ["positive", "negative", "neutral"]}."""
        return self.model.classify_text(text, labels)

    @bentoml.api
    def extract_structured(self, text: str, extraction_schema: dict[str, list[str]]) -> dict:
        """Structured JSON extraction. extraction_schema e.g.
        {"product": ["name::str::Full product name", "price::str::Product price"]}.
        """
        return self.model.extract_json(text, extraction_schema)

    @bentoml.api
    def extract_relations(self, text: str, relations: list[str]) -> dict:
        """Relation extraction. relations e.g. ["works_for", "lives_in"]."""
        return self.model.extract_relations(text, relations)
