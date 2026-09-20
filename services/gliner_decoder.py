"""GLiNER Decoder — open-ontology entity discovery.

Run independently so its model is not resident beside the other large models:
    uv run bentoml serve services.gliner_decoder:GLiNERDecoderService -p 3002
"""

from __future__ import annotations

import bentoml
from bentoml.models import HuggingFaceModel

with bentoml.importing():
    from gliner import GLiNER

image = bentoml.images.Image(python_version="3.13").requirements_file("requirements.txt")


@bentoml.service(
    image=image,
    resources={"cpu": "4"},
    traffic={"timeout": 300},
)
class GLiNERDecoderService:
    """Discover entity spans and generate their types without a supplied schema."""

    model_path = HuggingFaceModel("knowledgator/gliner-decoder-large-v1.0")

    def __init__(self) -> None:
        self.model = GLiNER.from_pretrained(self.model_path, map_location="cpu")

    @bentoml.api
    def discover_entities(
        self,
        text: str,
        threshold: float = 0.3,
        num_generated_labels: int = 1,
        flat_ner: bool = True,
    ) -> dict:
        """Find entity spans and generate open-ontology labels for them."""
        entities = self.model.predict_entities(
            text,
            labels=["label"],
            threshold=threshold,
            num_gen_sequences=num_generated_labels,
            flat_ner=flat_ner,
        )
        return {"entities": entities}

    @bentoml.api
    def extract_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.3,
        num_generated_labels: int = 1,
        flat_ner: bool = True,
    ) -> dict:
        """Controlled extraction while also returning decoder-generated labels."""
        entities = self.model.predict_entities(
            text,
            labels=labels,
            threshold=threshold,
            num_gen_sequences=num_generated_labels,
            flat_ner=flat_ner,
        )
        return {"entities": entities}
