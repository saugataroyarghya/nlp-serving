from __future__ import annotations

import bentoml
from bentoml.models import HuggingFaceModel

with bentoml.importing():
    from transformers import pipeline

image = bentoml.images.Image(python_version="3.13").requirements_file("requirements.txt")


@bentoml.service(
    image=image,
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class TextClassifier:
    model_path = HuggingFaceModel("distilbert-base-uncased-finetuned-sst-2-english")

    def __init__(self) -> None:
        self.pipeline = pipeline("sentiment-analysis", model=self.model_path)

    @bentoml.api
    def classify(self, text: str) -> dict:
        result = self.pipeline(text)[0]
        return {"label": result["label"], "score": result["score"]}
