from fastapi.testclient import TestClient
from tastematch.api import app

def test_health_endpoint():
    assert TestClient(app).get("/health").json() == {"status":"ok"}
