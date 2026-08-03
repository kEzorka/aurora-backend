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

import hashlib
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

#: Код на запрос без `Range`. Так качается индекс: он маленький и нужен целиком.
OK: Final = 200

#: Сколько байт читать за раз при подсчёте суммы. Файл прогона — сотни
#: мегабайт, и читать его в память целиком незачем.
CHECKSUM_BLOCK: Final = 1 << 20

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


def fetch_document(
    url: str,
    *,
    transport: Transport,
    delays: Sequence[float] = DEFAULT_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Скачать файл целиком, а не диапазоном: так качается `.index`.

    Индекс — это килобайты, и диапазонами его брать нечем: смещения сообщений
    лежат в нём самом. Ретраи те же, что у диапазонов, и по той же причине:
    503 на индексе за семь минут до дедлайна отката — это пропущенный прогон,
    если не повторить.
    """
    return _repeat(
        url,
        "whole file",
        lambda: _check_document(transport(url, {})),
        delays=delays,
        sleep=sleep,
    )


def checksum(path: str | Path, *, block: int = CHECKSUM_BLOCK) -> str:
    """Сумма файла в том виде, в каком её ждёт манифест: `sha256:<hex>`.

    Считается по скачанному файлу, а не по телам ответов: проверять нужно то,
    что легло на диск, — между склейкой и записью есть файловая система, и
    оборванная запись даёт файл, которого ни один ответ сервера не содержал.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as data:
        while chunk := data.read(block):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def checksum_tree(path: str | Path, *, block: int = CHECKSUM_BLOCK) -> str:
    """Сумма каталога source-native Zarr: относительные имена плюс байты.

    Одних байтов недостаточно: два чанка, переставленные местами, имеют тот
    же конкатенированный поток, но описывают другое поле. Метаданные и имена
    входят в digest в детерминированном порядке.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"{root}: expected a directory")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"{root}: directory has no files")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with candidate.open("rb") as data:
            while chunk := data.read(block):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


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
    return _repeat(
        url,
        span.header,
        lambda: _check(span, transport(url, {"Range": span.header})),
        delays=delays,
        sleep=sleep,
    )


def _repeat(
    url: str,
    what: str,
    call: Callable[[], bytes],
    *,
    delays: Sequence[float],
    sleep: Callable[[float], None],
) -> bytes:
    """Повторить `call`, пока имеет смысл, и отказать, назвав, что не приехало."""
    attempts = len(delays) + 1
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
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
    raise AdapterError(f"{url} {what}", f"{attempts} attempts failed: {last}", "data")


def _check_document(response: Response) -> bytes:
    if response.status != OK:
        raise AdapterError("status whole file", response.status, OK)
    if not response.body:
        # Пустой индекс — это не «полей нет», а страница с ошибкой нулевой
        # длины: отбор по нему дал бы «поле не найдено» вместо отказа сети.
        raise AdapterError("body whole file", "empty", "at least one byte")
    return response.body


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
