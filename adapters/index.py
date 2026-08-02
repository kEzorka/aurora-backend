"""Индексы GRIB: какое сообщение где лежит в файле.

Файл прогона ECMWF — это 200+ МБ, из которых нужны считанные поля, и рядом с
ним лежит `.index`, а у GFS — `.idx`. Оба говорят одно и то же: сообщение
номер N начинается со смещения M. Скачать по ним можно ровно нужное
(`Range`), и разница между «18 полей» и «весь файл» здесь двузначная.

Форматы разные, и разница не косметическая:

* ECMWF `.index` — строки JSON, в каждой есть `_offset` и `_length`. Длина
  известна прямо, порядок строк ни на что не влияет.
* GFS `.idx` — строки через `:`, и длины в них **нет**: сообщение кончается
  там, где начинается следующее. У последнего сообщения конца нет вовсе — до
  конца файла, и это `Range: bytes=N-` без правой границы, а не «длина ноль».

Отбор здесь идёт в словаре поставщика (`2t` у ECMWF, `TMP` у GFS), а не в
каноне: канон появляется после разбора GRIB (`adapters.ecmwf.RENAMES`,
`adapters.gfs.RENAMES`), а до загрузки файла разбирать нечего. Перевод имён —
дело того, кто составляет запрос.

Качать этот модуль не умеет: сеть — в `adapters.fetch`. Здесь только
арифметика над текстом, и потому она проверяется без сети.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, NamedTuple

from adapters.errors import AdapterError

#: Разделитель полей в `.idx` у GFS и порядок первых пяти из них.
IDX_SEPARATOR: Final = ":"
IDX_NUMBER: Final = 0
IDX_OFFSET: Final = 1
IDX_PARAM: Final = 3
IDX_LEVEL: Final = 4

#: Ключи `.index` у ECMWF: смещение, длина и то, чем отбирают сообщение.
ECMWF_OFFSET: Final = "_offset"
ECMWF_LENGTH: Final = "_length"
ECMWF_PARAM: Final = "param"
ECMWF_LEVEL: Final = "levelist"

#: Насколько далеко друг от друга могут лежать сообщения, чтобы их всё же
#: качали одним запросом. Ноль — склеивать только соседние впритык.
#:
#: Величина не бесплатная в обе стороны: лишний запрос — это RTT и новый шанс
#: на 5xx, лишние байты — это трафик. Умолчание в мегабайт выбрано по тому,
#: что дешевле на 200-мегабайтном файле, где нужных сообщений десятки.
DEFAULT_GAP: Final = 1 << 20


class Message(NamedTuple):
    """Одно сообщение GRIB в файле: где начинается, сколько занимает и что это.

    `length is None` — «до конца файла». Так кончается последнее сообщение в
    `.idx` у GFS, где длины нет ни у кого, и ноль вместо `None` дал бы пустой
    `Range` на единственное сообщение, которое почти всегда и нужно.

    `level` — пустая строка у приземных полей. Пустая, а не `None`, чтобы отбор
    был обычным сравнением пар, а не разбором двух случаев.
    """

    offset: int
    length: int | None
    param: str
    level: str = ""

    @property
    def end(self) -> int | None:
        """Первый байт за сообщением или `None`, если оно тянется до конца файла."""
        return None if self.length is None else self.offset + self.length


class Range(NamedTuple):
    """Диапазон байтов для одного запроса. `end` — включительно, как в HTTP.

    Включительно потому, что таким его понимает сервер: `bytes=0-99` — это сто
    байт. Держать внутри полуинтервал и вычитать единицу на выходе значит
    завести место, где эту единицу однажды забудут, — и сообщение приедет с
    приклеенным первым байтом следующего, то есть обрезанным для cfgrib.
    """

    start: int
    end: int | None

    @property
    def header(self) -> str:
        """Значение заголовка `Range`."""
        return f"bytes={self.start}-" if self.end is None else f"bytes={self.start}-{self.end}"


def parse_ecmwf(text: str) -> tuple[Message, ...]:
    """Разобрать `.index` ECMWF: одна строка JSON — одно сообщение."""
    messages = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as bad:
            raise AdapterError(f"index line {number}", line[:80], "JSON") from bad
        if ECMWF_OFFSET not in row or ECMWF_LENGTH not in row:
            raise AdapterError(
                f"index line {number}", sorted(row), f"{ECMWF_OFFSET}, {ECMWF_LENGTH}"
            )
        messages.append(
            Message(
                offset=int(row[ECMWF_OFFSET]),
                length=int(row[ECMWF_LENGTH]),
                param=str(row.get(ECMWF_PARAM, "")),
                level=str(row.get(ECMWF_LEVEL, "")),
            )
        )
    if not messages:
        raise AdapterError("index", "empty", "at least one message")
    return tuple(messages)


def parse_gfs(text: str) -> tuple[Message, ...]:
    """Разобрать `.idx` GFS. Длину даёт следующая строка, последнюю — файл.

    Строки сортируются по смещению перед вычитанием: разность соседей — это
    длина только в порядке файла, а `.idx` его лишь обычно повторяет. Один
    переставленный номер иначе даёт отрицательную длину, и `Range` с ней
    сервер удовлетворит куском чужого сообщения.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(IDX_SEPARATOR)
        if len(fields) <= IDX_LEVEL:
            raise AdapterError(f"idx line {number}", line[:80], "num:offset:date:param:level:...")
        try:
            offset = int(fields[IDX_OFFSET])
        except ValueError as bad:
            raise AdapterError(f"idx line {number}", fields[IDX_OFFSET], "offset in bytes") from bad
        rows.append((offset, fields[IDX_PARAM], fields[IDX_LEVEL]))
    if not rows:
        raise AdapterError("idx", "empty", "at least one message")

    rows.sort()
    return tuple(
        Message(
            offset=offset,
            # Последнее сообщение кончается вместе с файлом, и длины у него нет
            # ни в каком виде: `None` доедет до `Range` без правой границы.
            length=None if position + 1 == len(rows) else rows[position + 1][0] - offset,
            param=param,
            level=level,
        )
        for position, (offset, param, level) in enumerate(rows)
    )


def select(
    messages: Iterable[Message], wanted: Sequence[tuple[str, str]] | Mapping[str, str]
) -> tuple[Message, ...]:
    """Оставить сообщения, которые просили: пары «параметр, уровень».

    Отсутствие запрошенного — отказ, а не пустой список. Поле, которого в
    индексе не оказалось, — это либо опечатка в имени, либо файл, выложенный
    наполовину; и то и другое дальше превращается в срез без переменной,
    который заметят на сборке батча, а причину будут искать в другом месте.
    """
    pairs = tuple(wanted.items()) if isinstance(wanted, Mapping) else tuple(wanted)
    found = tuple(message for message in messages if (message.param, message.level) in pairs)
    missing = sorted(set(pairs) - {(message.param, message.level) for message in found})
    if missing:
        raise AdapterError("index", f"missing {missing}", "every requested field")
    return found


def ranges(messages: Iterable[Message], *, gap: int = DEFAULT_GAP) -> tuple[Range, ...]:
    """Свести сообщения в диапазоны байтов для запросов.

    Соседние склеиваются, если между ними не больше `gap` байт: лишний запрос
    стоит RTT и нового шанса на 5xx, лишние байты — трафика, и на файле в
    200 МБ с десятками нужных сообщений первое дороже.

    Сообщение «до конца файла» съедает всё, что за ним: правой границы у него
    нет, и любое продолжение диапазона уже внутри.
    """
    if gap < 0:
        raise AdapterError("gap", gap, "zero or more bytes")
    ordered = sorted(messages)
    if not ordered:
        return ()

    merged: list[Range] = []
    start, end = ordered[0].offset, ordered[0].end
    for message in ordered[1:]:
        if end is None or message.offset <= end + gap:
            end = None if end is None or message.end is None else max(end, message.end)
        else:
            merged.append(Range(start, end - 1))
            start, end = message.offset, message.end
    merged.append(Range(start, None if end is None else end - 1))
    return tuple(merged)
