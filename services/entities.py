"""GLiNER2, Laya, and KeyBERT in one compatible BentoML service.

Standalone, servable while we're learning it:
    uv run bentoml serve services.entities:GliNER2Service --reload -p 3000

The three libraries share one Python environment without dependency conflicts,
so their endpoints are exposed together through one Swagger UI (/docs).
"""

from __future__ import annotations

from typing import Any

import bentoml
from bentoml.models import HuggingFaceModel

from services.swagger_tags import SwaggerTagMiddleware

with bentoml.importing():
    from gliner2 import AutoExtractor
    from keybert import KeyBERT
    import laya

image = bentoml.images.Image(python_version="3.13").requirements_file("requirements.txt")


@bentoml.service(
    image=image,
    resources={"cpu": "4"},
    traffic={"timeout": 300},
)
class GliNER2Service:
    gliner_model_path = HuggingFaceModel("fastino/gliner2.5-base-v1")
    laya_model_path = HuggingFaceModel(
        "convaiinnovations/laya",
        exclude=["multilingual/*", "typed-decisions/*", "assets/*", "eval/*"],
    )
    keyword_model_path = HuggingFaceModel(
        "sentence-transformers/all-MiniLM-L6-v2",
        exclude=["onnx/*", "openvino/*", "*.bin", "*.h5", "*.ot"],
    )

    def __init__(self) -> None:
        self.gliner_model = AutoExtractor.from_pretrained(self.gliner_model_path)
        self.laya_model = laya.load(str(self.laya_model_path))
        self.keyword_model = KeyBERT(model=str(self.keyword_model_path))

    @bentoml.api
    def extract_entities(self, text: str, labels: list[str]) -> dict:
        """Zero-shot NER. labels e.g. ["person", "organization", "location"]."""
        return self.gliner_model.extract_entities(text, labels)

    @bentoml.api
    def classify_text(self, text: str, labels: dict[str, list[str]]) -> dict:
        """Zero-shot classification. labels e.g. {"sentiment": ["positive", "negative", "neutral"]}."""
        return self.gliner_model.classify_text(text, labels)

    @bentoml.api
    def extract_structured(self, text: str, extraction_schema: dict[str, list[str]]) -> dict:
        """Structured JSON extraction. extraction_schema e.g.
        {"product": ["name::str::Full product name", "price::str::Product price"]}.
        """
        return self.gliner_model.extract_json(text, extraction_schema)

    @bentoml.api
    def extract_relations(self, text: str, relations: list[str]) -> dict:
        """Relation extraction. relations e.g. ["works_for", "lives_in"]."""
        return self.gliner_model.extract_relations(text, relations)

    @bentoml.api
    def decide(
        self,
        state: str | dict[str, Any] | list[dict[str, Any]],
        questions: dict[str, dict[str, Any]],
    ) -> dict:
        """Use Laya to answer choice, ordered-score, and yes/no questions."""
        if not questions:
            raise ValueError("questions must contain at least one question")
        raw_result = self.laya_model.predict(state, questions)
        decisions: dict[str, str | bool] = {}
        for question_id, answer in raw_result["answers"].items():
            if answer["type"] == "choice":
                decisions[question_id] = answer["choice"]
            elif answer["type"] == "score":
                selected_index = max(answer["probabilities"], key=answer["probabilities"].get)
                decisions[question_id] = answer["legend"][selected_index]
            else:
                decisions[question_id] = answer["noul"] >= 0.5
        return {"decisions": decisions}

    @bentoml.api
    def keywords(self, text: str) -> dict:
        """Use KeyBERT to extract keywords and keyphrases."""
        if not text.strip():
            raise ValueError("text must not be empty")
        extracted = self.keyword_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=10,
            use_mmr=True,
            diversity=0.5,
        )
        return {"keywords": [keyword for keyword, _score in extracted]}


GliNER2Service.add_asgi_middleware(
    SwaggerTagMiddleware,
    route_tags={
        "/extract_entities": "GLiNER",
        "/classify_text": "GLiNER",
        "/extract_structured": "GLiNER",
        "/extract_relations": "GLiNER",
        "/decide": "Laya",
        "/keywords": "KeyBERT",
    },
    tag_descriptions={
        "GLiNER": "Zero-shot entity, classification, structured, and relation extraction.",
        "Laya": "Typed non-generative decisions with probabilities and confidence.",
        "KeyBERT": "Embedding-based keyword and keyphrase extraction.",
    },
)
