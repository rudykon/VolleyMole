from fastapi.testclient import TestClient

from volleymole_service.main import app


def test_health() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "processing": "local"}
