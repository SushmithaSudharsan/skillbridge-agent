"""HTTP layer: routes, validation, status codes, security headers. Needs fastapi + httpx
(installed by `pip install -e ".[dev,web]"`); skipped where they aren't available."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from test_analysis import JD  # noqa: E402
from test_web_service import SETTINGS, service  # noqa: E402

CODE = {"X-Access-Code": "open-sesame"}


@pytest.fixture
def client():
    return TestClient(create_app(SETTINGS, service()))


def test_health_is_public_and_has_security_headers(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert "default-src 'self'" in r.headers["content-security-policy"]


def test_everything_else_needs_the_access_code(client):
    assert client.get("/api/samples").status_code == 403
    assert client.get("/api/samples", headers=CODE).status_code == 200


def test_analyze_validates_input(client):
    assert client.post("/api/analyze", json={"sample_id": "aiml_fresher", "jd_text": "too short"},
                       headers=CODE).status_code == 422
    assert client.post("/api/analyze", json={"sample_id": "../x", "jd_text": JD}, headers=CODE).status_code == 422


def test_full_flow(client):
    r = client.post("/api/analyze", json={"sample_id": "aiml_fresher", "jd_text": JD, "years": 0}, headers=CODE)
    assert r.status_code == 200
    sid = r.json()["session_id"]
    pdf = client.get(f"/api/sessions/{sid}/report.pdf", headers=CODE)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert client.post(f"/api/sessions/{sid}/end").json() == {"ended": True}
    assert client.get(f"/api/sessions/{sid}/report.pdf", headers=CODE).status_code == 404


def test_bad_session_id_is_rejected(client):
    assert client.get("/api/sessions/not-an-id/report.pdf", headers=CODE).status_code == 422


def test_docs_are_off_by_default(client):
    assert client.get("/docs").status_code == 404
