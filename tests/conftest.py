import pytest
from starlette.testclient import TestClient

from services.entities import GliNER2Service


@pytest.fixture(scope="session")
def client():
    with TestClient(GliNER2Service.to_asgi()) as test_client:
        yield test_client
