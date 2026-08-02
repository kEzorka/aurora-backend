"""Загрузка по байтовым диапазонам с ретраями (BACKLOG 1.7).

Что здесь есть: один `Range`-запрос на диапазон, повтор с растущей паузой,
потолок попыток и склейка ответов в файл. Чего нет: знания о том, какие поля
нужны, — это `adapters.index`, — и знания о расписании цикла, потому что
расписание живёт в `pipeline` и адаптеру запрещено о нём знать
(`tests/test_boundaries.py`). Паузы поэтому приходят параметром, а не берутся
из `pipeline.schedule.delays`, хотя туда они и попадут.

Транспорт — тоже параметр. Тест, которому нужна сеть, — это мониторинг, а не
тест (`docs/TESTING.md`), и подстановка вызываемого объекта здесь не приём
ради тестируемости: тот же шов нужен кэшу (эпик 3), который встанет перед
источником и будет отвечать из своего файла вместо HTTP.

Что проверяется по существу:

* сервер обязан ответить `206 Partial Content`. `200 OK` на `Range`-запрос
  означает, что диапазон проигнорирован и приехал файл целиком, — за
  сообщением в 200 КБ приедет 200 МБ, и молча;
* длина ответа обязана совпасть с запрошенной. Прокси и CDN режут тело, и
  обрезанное сообщение GRIB cfgrib читает без ошибки, просто с мусором в
  последних точках;
* повторяется только то, что имеет смысл повторять. `404` не станет `200` от
  пятой попытки, а `503` — вполне.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from adapters.errors import AdapterError
from adapters.index import Range

#: Код, которым сервер отвечает на `Range`. Единственный приемлемый: `200`
#: означает, что диапазон проигнорирован и приехал файл целиком.
PARTIAL: Final = 206

#: Коды, после которых имеет смысл повторить. Всё остальное повтором не
#: лечится: `404` не станет `200` от пятой попытки, а `416` — это неверный
#: диапазон, то есть ошибка на нашей стороне.
RETRIABLE_STATUS: Final = (408, 425, 429, 500, 502, 503, 504)

#: Паузы по умолчанию: четыре повтора, от 15 секунд до пяти минут. Те же
#: числа, что у `pipeline.schedule.delays`, но своей копией — адаптер про
#: расписание не знает, а без умолчания каждый вызов тащил бы их за собой.
DEFAULT_DELAYS: Final = (15.0, 30.0, 60.0, 120.0)


class Response(NamedTuple):
    """Ответ транспорта: код, тело и что сервер сказал про диапазон."""

    status: int
    body: bytes
    content_range: str = ""


#: Транспорт: URL и заголовки на входе, `Response` на выходе. Подставляется
#: тестом, а в бою — обёрткой над HTTP-клиентом.
Transport = Callable[[str, dict[str, str]], Response]

#: Сколько ждать ответа. Файл прогона отдаётся минутами, но это время до
#: **первого** байта: висящее без ответа соединение дешевле оборвать и
#: повторить, чем держать до дедлайна отката.
TIMEOUT_SEC: Final = 120


def http(url: str, headers: dict[str, str], *, timeout: float = TIMEOUT_SEC) -> Response:
    """Единственное место в модуле, которое ходит в сеть.

    `HTTPError` разворачивается в обычный `Response`: 503 — это ответ сервера,
    и решать, повторять его или нет, — дело `fetch_ranges`, у которого есть
    список кодов и паузы. Исключение здесь оборвало бы ретраи на первом же 503,
    то есть ровно там, где они и нужны.
    """
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(
                status=int(response.status),
                body=bytes(response.read()),
                content_range=str(response.headers.get("Content-Range", "")),
            )
    except urllib.error.HTTPError as answered:
        return Response(status=int(answered.code), body=bytes(answered.read()))


def fetch_ranges(
    url: str,
    wanted: Sequence[Range],
    *,
    transport: Transport,
    delays: Sequence[float] = DEFAULT_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bytes, ...]:
    """Скачать каждый диапазон отдельным запросом, повторяя по паузам.

    Возвращает тела в порядке `wanted`. Порядок сохраняется, потому что дальше
    они склеиваются в файл, а сообщения GRIB в файле лежат по возрастанию
    смещения — переставленные, они и есть другой файл.
    """
    return tuple(
        _one_range(url, span, transport=transport, delays=delays, sleep=sleep) for span in wanted
    )


def fetch_to_file(
    url: str,
    wanted: Sequence[Range],
    path: str | Path,
    *,
    transport: Transport,
    delays: Sequence[float] = DEFAULT_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Скачать диапазоны и склеить их в один файл GRIB.

    Пишется через `.tmp` и `replace`, как манифест: оборванная загрузка иначе
    оставляет на месте файла его половину, а cfgrib читает половину без ошибки
    — просто сообщений в ней меньше, чем ждали.
    """
    bodies = fetch_ranges(url, wanted, transport=transport, delays=delays, sleep=sleep)
    return write_messages(path, bodies)


def write_messages(path: str | Path, bodies: Iterable[bytes]) -> Path:
    """Склеить тела в файл. Сообщения GRIB конкатенируются как есть."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as out:
        for body in bodies:
            out.write(body)
    tmp.replace(target)
    return target


def _one_range(
    url: str,
    span: Range,
    *,
    transport: Transport,
    delays: Sequence[float],
    sleep: Callable[[float], None],
) -> bytes:
    attempts = len(delays) + 1
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return _check(span, transport(url, {"Range": span.header}))
        except AdapterError as refused:
            if not _worth_repeating(refused):
                raise
            last = refused
        except OSError as broken:
            # Оборванное соединение — это то же «попробуй ещё раз», что и 503,
            # и приезжает оно чаще: файл на 200 МБ висит на канале минутами.
            last = broken
        if attempt + 1 < attempts:
            sleep(delays[attempt])
    raise AdapterError(
        f"{url} {span.header}", f"{attempts} attempts failed: {last}", "206 and data"
    )


def _check(span: Range, response: Response) -> bytes:
    if response.status != PARTIAL:
        raise AdapterError(f"status {span.header}", response.status, PARTIAL)
    expected = None if span.end is None else span.end - span.start + 1
    if expected is not None and len(response.body) != expected:
        # Обрезанное тело cfgrib прочитает без ошибки: сообщение GRIB кончается
        # маркером `7777`, а до него в последних точках будет мусор.
        raise AdapterError(f"length {span.header}", len(response.body), expected)
    if not response.body:
        raise AdapterError(f"body {span.header}", "empty", "at least one byte")
    return response.body


def _worth_repeating(refused: AdapterError) -> bool:
    return refused.field.startswith("status") and refused.got in RETRIABLE_STATUS
