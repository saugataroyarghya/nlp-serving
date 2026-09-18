import pytest
from starlette.testclient import TestClient

from service import TextClassifier


@pytest.fixture(scope="session")
def client():
    with TestClient(TextClassifier.to_asgi()) as c:
        yield c


def test_classify(client):
    r = client.post("/classify", json={"text": "I love this product!"})
    assert r.status_code == 200
    assert r.json()["label"] == "POSITIVE"
