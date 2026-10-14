"""The web page and the endpoints it uses. Documents here are FICTIONAL."""

from pathlib import Path

from tests.test_api import ADMIN, USER, _client


def test_page_and_assets_are_served_with_security_headers(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert '<script src="/static/app.js" defer>' in page.text
        policy = page.headers["content-security-policy"]
        assert "script-src 'self'" in policy and "connect-src 'self'" in policy
        assert "'unsafe-inline'" not in policy
        assert page.headers["x-content-type-options"] == "nosniff"
        for asset, kind in (("/static/app.js", "javascript"), ("/static/app.css", "css")):
            response = client.get(asset)
            assert response.status_code == 200 and kind in response.headers["content-type"]


def test_page_is_public_but_its_data_follows_the_keys(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/").status_code == 200  # the page itself holds no data
        assert client.get("/v1/corpus").status_code == 401
        corpus = client.get("/v1/corpus", headers={"X-API-Key": USER}).json()
        assert corpus == {"documents": 1, "issuers": ["BAM"], "first_year": None, "last_year": None}


def test_public_queries_open_questions_but_not_ingestion(tmp_path: Path) -> None:
    with _client(tmp_path, public_queries=True) as client:
        assert client.get("/v1/corpus").status_code == 200
        answer = client.post("/v1/query", json={"question": "Quel est le ratio de liquidité ?"})
        assert answer.status_code == 200 and answer.json()["status"] == "answered"
        upload = client.post("/v1/ingest", data={"title": "x"})
        assert upload.status_code == 401  # admin key still required
        wrong_key = client.get("/v1/corpus", headers={"X-API-Key": "not-a-key-FICTIONAL"})
        assert wrong_key.status_code == 401  # a bad key is refused, not treated as public
        assert client.get("/v1/corpus", headers={"X-API-Key": ADMIN}).status_code == 200


def test_sources_carry_the_full_excerpt(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        body = client.post(
            "/v1/query",
            json={"question": "Quel est le ratio de liquidité ?"},
            headers={"X-API-Key": USER},
        ).json()
        source = body["sources"][0]
        assert "140 %" in source["excerpt"] and len(source["excerpt"]) >= len(source["snippet"])


def test_feedback_is_counted(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        headers = {"X-API-Key": USER}
        sent = client.post(
            "/v1/feedback", json={"request_id": "abc123", "rating": "up"}, headers=headers
        )
        assert sent.status_code == 204
        assert 'reglens_feedback_total{rating="up"}' in client.get("/metrics").text
        bad = client.post(
            "/v1/feedback", json={"request_id": "<x>", "rating": "up"}, headers=headers
        )
        assert bad.status_code == 422
