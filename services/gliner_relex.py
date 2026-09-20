"""GLiNER-RelEx — joint zero-shot entity and relation extraction.

Run independently so its model is not resident beside the other large models:
    uv run bentoml serve services.gliner_relex:GLiNERRelExService -p 3001
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
class GLiNERRelExService:
    """Joint NER and RE with independent candidate and relation thresholds."""

    model_path = HuggingFaceModel("knowledgator/gliner-relex-large-v1.0")

    def __init__(self) -> None:
        self.model = GLiNER.from_pretrained(self.model_path, map_location="cpu")

    @bentoml.api
    def extract(
        self,
        text: str,
        entity_labels: list[str] | dict[str, str],
        relations: list[str],
        entity_threshold: float = 0.4,
        adjacency_threshold: float = 0.6,
        relation_threshold: float = 0.75,
        flat_ner: bool = False,
        multi_label: bool = False,
    ) -> dict:
        """Extract entities and relations together.

        ``entity_labels`` may be a list or a mapping from label to description.
        Include ``"other"`` when relation endpoints have unknown entity types.
        """
        entities, extracted_relations = self.model.inference(
            texts=[text],
            labels=entity_labels,
            relations=relations,
            threshold=entity_threshold,
            adjacency_threshold=adjacency_threshold,
            relation_threshold=relation_threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            return_relations=True,
        )
        return {
            "entities": entities[0],
            "relations": extracted_relations[0],
        }
