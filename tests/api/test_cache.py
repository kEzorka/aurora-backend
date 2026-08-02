"""`Cache-Control`, `ETag` и `304` — приёмка BACKLOG 5.7.

Критерий приёмки дословно: «повторный запрос прогноза отдаёт `304`». Проверяется
он с обеих сторон: что повтор того же запроса действительно отдаёт `304` без
тела, и что запрос, у которого ответ **другой**, `304` не получает ни при каком
совпадении заголовков. Второе важнее: `304` на изменившийся прогноз — это
клиент, который сутки показывает вчерашнюю погоду и уверен, что она сегодняшняя.
"""

from fastapi.testclient import TestClient

COVERAGE = "/v1/meta/coverage"
GRID = "/v1/forecast/grid"
POINT = "/v1/forecast/point"

MOSCOW = {"lat": 55.75, "lon": 37.62}
#: Те же окно и срок, что в `test_forecast_grid`: узлы фикстуры, а не океан.
BOX = {"bbox": "-18,-90,54,45", "var": "t2m", "time": "2026-08-01T06:00:00Z"}


def test_a_forecast_answer_carries_both_a_ttl_and_a_tag(client: TestClient) -> None:
    """Заголовки отвечают на разные вопросы: `max-age` — сколько можно не
    спрашивать, `ETag` — что ответить, когда спросили (docs/API_CONTRACT.md §5.5)."""
    headers = client.get(POINT, params=MOSCOW).headers

    assert headers["Cache-Control"].startswith("public, max-age=")
    assert headers["ETag"].startswith('"') and headers["ETag"].endswith('"')
    # Сильный, не `W/`: равенство здесь побайтовое, и обещать меньше незачем.
    assert not headers["ETag"].startswith("W/")


def test_the_same_request_twice_gets_the_same_tag(client: TestClient) -> None:
    """ETag, меняющийся сам по себе, отменяет весь механизм: клиент никогда не
    попадёт в `304` и будет качать карту заново каждый раз."""
    first = client.get(POINT, params=MOSCOW).headers["ETag"]
    second = client.get(POINT, params=MOSCOW).headers["ETag"]

    assert first == second


def test_a_repeated_forecast_request_gets_304_without_a_body(client: TestClient) -> None:
    """Сам критерий приёмки. Карта на 1440×721 весит мегабайты, и повтор ради
    тех же байтов — это трафик, которого можно не тратить."""
    tag = client.get(GRID, params=BOX).headers["ETag"]

    again = client.get(GRID, params=BOX, headers={"If-None-Match": tag})

    assert again.status_code == 304
    assert again.content == b""
    # Заголовки кэша обязаны приехать и с `304`: по ним клиент продлевает
    # жизнь своей копии, иначе вернётся с тем же вопросом через секунду.
    assert again.headers["ETag"] == tag
    assert again.headers["Cache-Control"].startswith("public, max-age=")


def test_coverage_answers_304_as_well(client: TestClient) -> None:
    """Покрытие фронтенд зовёт при каждом старте, и меняется оно четыре раза
    в сутки."""
    tag = client.get(COVERAGE).headers["ETag"]

    assert client.get(COVERAGE, headers={"If-None-Match": tag}).status_code == 304


def test_another_question_never_gets_304(client: TestClient) -> None:
    """ETag от одного `init_time` отдал бы `304` на запрос, которого клиент не
    делал: тело зависит ещё и от запроса — от единиц, шага, набора переменных."""
    celsius = client.get(POINT, params=MOSCOW)
    kelvin = client.get(POINT, params={**MOSCOW, "units": "si"})

    assert celsius.headers["ETag"] != kelvin.headers["ETag"]

    answered = client.get(
        POINT,
        params={**MOSCOW, "units": "si"},
        headers={"If-None-Match": celsius.headers["ETag"]},
    )

    assert answered.status_code == 200
    # И это именно другой ответ, а не другой заголовок при том же теле.
    assert answered.json()["units"]["t2m"] == "K"


def test_a_different_place_gets_a_different_tag(client: TestClient) -> None:
    """Узел сетки в ответе свой у каждой точки, и ответы за разные точки —
    разные ответы, как бы близко они ни лежали."""
    here = client.get(POINT, params=MOSCOW).headers["ETag"]
    there = client.get(POINT, params={"lat": -33.9, "lon": 151.2}).headers["ETag"]

    assert here != there


def test_a_list_of_tags_and_a_star_are_both_understood(client: TestClient) -> None:
    """Клиент вправе перечислить несколько своих копий или прислать `*`
    (RFC 9110 §13.1.2). Непонятый заголовок — это `200` вместо `304`, то есть
    механизм, который молча не работает."""
    tag = client.get(POINT, params=MOSCOW).headers["ETag"]
    stale = '"' + "0" * 32 + '"'

    listed = client.get(POINT, params=MOSCOW, headers={"If-None-Match": f"{stale}, {tag}"})
    weak = client.get(POINT, params=MOSCOW, headers={"If-None-Match": f"W/{tag}"})
    star = client.get(POINT, params=MOSCOW, headers={"If-None-Match": "*"})
    unknown = client.get(POINT, params=MOSCOW, headers={"If-None-Match": stale})

    assert [listed.status_code, weak.status_code, star.status_code] == [304, 304, 304]
    assert unknown.status_code == 200


def test_an_error_is_not_cached_by_a_tag(client: TestClient) -> None:
    """Отказ кэшировать нечего: `400` завтра может стать `200` — например,
    потому что запрошенный срок наконец попал в горизонт прогона."""
    refused = client.get(POINT, params={**MOSCOW, "vars": "temperature"})

    assert refused.status_code == 400
    assert "ETag" not in refused.headers
