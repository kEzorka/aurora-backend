"""Demo-клиент: одна поставка и один origin с JSON API."""

from fastapi.testclient import TestClient


def test_root_serves_the_demo(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="point-panel"' in response.text
    assert 'id="animation-panel"' in response.text


def test_demo_alias_serves_the_same_document(client: TestClient) -> None:
    assert client.get("/demo").content == client.get("/").content


def test_frontend_assets_are_local_and_available(client: TestClient) -> None:
    html = client.get("/").text

    assert "https://" not in html
    for path, content_type in (
        ("/assets/styles.css", "text/css"),
        ("/assets/app.js", "text/javascript"),
        ("/assets/gif.js", "text/javascript"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)


def test_demo_uses_both_forecast_api_shapes(client: TestClient) -> None:
    script = client.get("/assets/app.js").text

    assert 'request("/v1/meta/coverage"' in script
    assert 'request("/v1/forecast/point"' in script
    assert 'request("/v1/forecast/grid"' in script
    assert "AuroraGif.encode" in script
