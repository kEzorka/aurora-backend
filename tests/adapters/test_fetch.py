"""Загрузка по диапазонам и ретраи (BACKLOG 1.7).

Сети здесь нет: транспорт подставляется вызываемым объектом, паузы — списком,
`sleep` — записывающей заглушкой. Тест, которому нужна сеть, — это мониторинг,
а не тест (`docs/TESTING.md`), и проверять на живом ECMWF, что после 503 будет
повтор, значит ждать 503.
"""

from pathlib import Path

import pytest

from adapters.errors import AdapterError
from adapters.fetch import Response, checksum_tree, fetch_ranges, fetch_to_file
from adapters.index import Range

DELAYS = (1.0, 2.0, 4.0)


class Server:
    """Транспорт-заглушка: отдаёт заготовленные ответы по очереди и помнит,
    о чём его просили."""

    def __init__(self, *answers: Response) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> Response:
        self.asked.append((url, headers["Range"]))
        return self.answers.pop(0)


class Clock:
    """`sleep`, который не спит, а записывает, сколько его просили спать."""

    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def _ok(body: bytes) -> Response:
    return Response(206, body)


def test_each_range_becomes_its_own_request_in_order() -> None:
    """Порядок сохраняется: сообщения GRIB в файле лежат по возрастанию
    смещения, и переставленные — это другой файл."""
    server = Server(_ok(b"A" * 100), _ok(b"B" * 50))

    bodies = fetch_ranges(
        "https://example/f.grib2", [Range(0, 99), Range(200, 249)], transport=server, sleep=Clock()
    )

    assert bodies == (b"A" * 100, b"B" * 50)
    assert server.asked == [
        ("https://example/f.grib2", "bytes=0-99"),
        ("https://example/f.grib2", "bytes=200-249"),
    ]


def test_a_full_file_instead_of_a_range_is_refused() -> None:
    """`200 OK` на `Range`-запрос означает, что диапазон проигнорирован: за
    сообщением в 200 КБ приехало 200 МБ, и молча."""
    server = Server(Response(200, b"the whole 200 MB file"))

    with pytest.raises(AdapterError, match="status"):
        fetch_ranges("https://example/f.grib2", [Range(0, 99)], transport=server, sleep=Clock())


def test_a_truncated_body_is_refused() -> None:
    """Прокси и CDN режут тело, и обрезанное сообщение cfgrib читает без
    ошибки: до маркера `7777` в последних точках будет мусор."""
    server = Server(_ok(b"A" * 40))

    with pytest.raises(AdapterError, match="length"):
        fetch_ranges("https://example/f.grib2", [Range(0, 99)], transport=server, sleep=Clock())


def test_an_open_ended_range_has_no_length_to_check() -> None:
    """У последнего сообщения `.idx` правой границы нет, и сравнивать длину
    не с чем. Пустое тело всё равно отвергается: файл не бывает пустым."""
    assert fetch_ranges(
        "https://example/f.grib2",
        [Range(1700, None)],
        transport=Server(_ok(b"tail")),
        sleep=Clock(),
    ) == (b"tail",)

    with pytest.raises(AdapterError, match="body"):
        fetch_ranges(
            "https://example/f.grib2",
            [Range(1700, None)],
            transport=Server(_ok(b"")),
            sleep=Clock(),
        )


def test_a_temporary_failure_is_retried_with_the_given_pauses() -> None:
    """Паузы приходят параметром, а не берутся из расписания: адаптеру
    запрещено знать про `pipeline` (`tests/test_boundaries.py`)."""
    server = Server(Response(503, b""), Response(503, b""), _ok(b"A" * 10))
    clock = Clock()

    bodies = fetch_ranges(
        "https://example/f.grib2", [Range(0, 9)], transport=server, delays=DELAYS, sleep=clock
    )

    assert bodies == (b"A" * 10,)
    assert clock.slept == [1.0, 2.0]


def test_a_permanent_failure_is_not_retried() -> None:
    """`404` не станет `200` от пятой попытки, а ждать он будет те же семь
    минут, что и настоящий отказ источника."""
    server = Server(Response(404, b"not found"))
    clock = Clock()

    with pytest.raises(AdapterError, match="status"):
        fetch_ranges(
            "https://example/f.grib2", [Range(0, 9)], transport=server, delays=DELAYS, sleep=clock
        )

    assert server.asked == [("https://example/f.grib2", "bytes=0-9")]
    assert clock.slept == []


def test_a_dropped_connection_is_retried_too() -> None:
    """Файл на 200 МБ висит на канале минутами, и оборванное соединение
    приезжает чаще, чем 503."""

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, url: str, headers: dict[str, str]) -> Response:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionResetError("peer hung up")
            return _ok(b"A" * 10)

    clock = Clock()
    bodies = fetch_ranges(
        "https://example/f.grib2", [Range(0, 9)], transport=Flaky(), delays=DELAYS, sleep=clock
    )

    assert bodies == (b"A" * 10,)
    assert clock.slept == [1.0]


def test_the_attempt_ceiling_ends_in_a_refusal_that_names_the_range() -> None:
    """Потолок попыток есть (docs/PIPELINE.md §3.5), и упереться в него —
    это отказ, а не бесконечное ожидание. Что именно не приехало, обязано быть
    в тексте: дальше это станет отметкой о пропуске."""
    server = Server(*[Response(503, b"") for _ in range(4)])
    clock = Clock()

    with pytest.raises(AdapterError, match="bytes=0-9"):
        fetch_ranges(
            "https://example/f.grib2", [Range(0, 9)], transport=server, delays=DELAYS, sleep=clock
        )

    assert len(server.asked) == 4  # три паузы — четыре попытки
    assert clock.slept == list(DELAYS)


def test_bodies_are_glued_into_one_grib_file(tmp_path: Path) -> None:
    """Сообщения GRIB конкатенируются как есть: файл из двух сообщений — это
    их байты подряд, ничего между ними не нужно."""
    server = Server(_ok(b"GRIB-one"), _ok(b"GRIB-two"))
    path = tmp_path / "slice" / "input.grib2"

    written = fetch_to_file(
        "https://example/f.grib2",
        [Range(0, 7), Range(8, 15)],
        path,
        transport=server,
        sleep=Clock(),
    )

    assert written == path
    assert path.read_bytes() == b"GRIB-oneGRIB-two"
    # Через `.tmp` и `replace`: половина файла на месте целого читается cfgrib
    # без ошибки, просто сообщений в ней меньше, чем ждали.
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_a_failed_download_leaves_no_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "input.grib2"

    with pytest.raises(AdapterError):
        fetch_to_file(
            "https://example/f.grib2",
            [Range(0, 9)],
            path,
            transport=Server(Response(404, b"")),
            sleep=Clock(),
        )

    assert not path.exists()


def test_tree_checksum_includes_relative_names_and_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "c").mkdir(parents=True)
        (root / "c" / "0").write_bytes(b"same bytes")

    assert checksum_tree(first, block=2) == checksum_tree(second, block=100)
    (second / "c" / "0").rename(second / "c" / "1")
    assert checksum_tree(first) != checksum_tree(second)


def test_tree_checksum_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        checksum_tree(tmp_path)
