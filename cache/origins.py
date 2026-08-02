"""Живые источники под `cache.proxy.Origin` (BACKLOG 1.5, 1.6, 3.2).

Их двое, и делятся они по нарезке, а не по данным: карты за срок идут из ARCO
(`ArcoOrigin`), ряды в точке — из CDS (`CdsOrigin`). Данные при этом один и тот
же ERA5; см. `adapters.era5_cds` о том, почему одним источником не обойтись.

Прослойка тонкая и нужна ровно из-за границы пакетов: адаптеру запрещено
знать про кэш (`tests/test_boundaries.py`), поэтому переводить «данных ещё
нет» из `adapters.errors.NotYetInSourceError` в `cache.proxy.NotYetError`
приходится на стороне кэша. Запрет не формальность: адаптер, знающий про
отказы кэша, начинает решать, что кэшировать, — а это второе место с той же
логикой, и расходятся такие места молча.

Заодно здесь живёт то, что источник знает про себя, а кэш — нет: нарезка по
времени (`Grid`) и версия данных за срок (`era5t` против финального ERA5).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import xarray as xr

from adapters import era5_arco, era5_cds
from adapters.errors import AdapterError, NotYetInSourceError
from cache.chunks import encode
from cache.proxy import Grid, NotYetError
from contracts import canon

#: Нарезка ARCO по времени: чанк — один час, и это его свойство, а не наш
#: выбор (docs/CACHE.md §1). `epoch` — начало ERA5.
ARCO_GRID: Final = Grid(
    epoch=datetime(canon.HISTORY_START_YEAR, 1, 1, tzinfo=UTC),
    step=timedelta(hours=1),
    span=1,
    # Двое суток карт на один запрос. Предел маленький намеренно: чанк равен
    # сроку, и ряд в точке за год превратился бы в 8760 обращений к бакету на
    # один ответ. Ряды берутся из CDS (1.6), а не отсюда.
    max_chunks=48,
)


#: Как часто переоткрывать архив. Не оптимизация, а срок жизни слепой зоны:
#: `open_zarr` читает ось времени сразу, и открытый в понедельник архив до
#: конца работы процесса уверен, что данных за вторник нет. Отказ живёт шесть
#: часов (`cache.index.NEGATIVE_TTL_S`), после чего прокси спрашивает снова —
#: и обязан спросить у архива, который знает про новые сроки.
REFRESH_AFTER: Final = timedelta(hours=1)


class ArcoOrigin:
    """ERA5 из публичного бакета — карты за прошлое.

    Архив открывается не на каждый чанк: `open_zarr` читает описание всех
    переменных за все восемьдесят лет, и платить за метаданные больше, чем за
    данные, — не дело. Но и не один раз навсегда: см. `REFRESH_AFTER`.
    """

    # Без `Final`: `cache.proxy.Origin` — Protocol, а он требует изменяемых
    # атрибутов, и константа тут ломает соответствие протоколу молча — mypy
    # заметит это не здесь, а у каждого, кто позовёт `serve`.
    name = "arco"
    dataset = "era5"

    def __init__(
        self,
        source: Any = era5_arco.ARCO_URL,
        *,
        grid: Grid = ARCO_GRID,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        refresh_after: timedelta = REFRESH_AFTER,
    ) -> None:
        self.version = era5_arco.ADAPTER_VERSION
        self.grid = grid
        self._source = source
        self._clock = clock
        self._refresh_after = refresh_after
        self._archive: xr.Dataset | None = None
        self._opened = datetime.min.replace(tzinfo=UTC)

    def source_version(self, chunk: int) -> str:
        """`era5t` или `era5-final` — до похода наружу, по номеру чанка.

        Именно до: версия входит в ключ, а ключ решает, идти ли вообще
        (`cache.proxy.Origin`).
        """
        return era5_arco.source_version(self.grid.begins(chunk), now=self._clock())

    def fetch(self, variable: str, chunk: int) -> bytes:
        """Срез за срок чанка. `NotYetError`, когда срок новее архива."""
        moment = self.grid.begins(chunk)
        try:
            sliced = era5_arco.read_slice(self._open(), variable, moment, now=self._clock())
        except NotYetInSourceError as gap:
            # Слепая зона реанализа. Наверх она идёт типом кэша, и отказ ляжет
            # в `absent` на шесть часов: без этого один человек, листающий
            # календарь, устраивает поход в бакет на каждое движение.
            raise NotYetError(str(gap)) from gap
        return encode(sliced)

    def _open(self) -> xr.Dataset:
        """Открытый архив, не старше `REFRESH_AFTER`.

        Переоткрывать на каждый отказ нельзя: человек, листающий календарь
        вперёд, попадает в слепую зону много раз подряд, и каждое движение
        стоило бы чтения описания всего архива.
        """
        now = self._clock()
        if self._archive is None or now - self._opened >= self._refresh_after:
            self._archive = era5_arco.open_archive(self._source)
            self._opened = now
        return self._archive


#: Часов в одном чанке рядов CDS — год. Число выбрано границей инвалидации, а
#: не удобством: метка `era5t` идёт на весь чанк (`adapters.era5_cds.read_series`),
#: и замена предварительных данных финальными выбрасывает чанк целиком. При
#: чанке в год под замену раз в три месяца попадает один чанк из десяти лет
#: ряда, а не весь ряд — это и есть та «резка по границе `FINAL_AFTER`», о
#: которой просит заметка в `read_series`.
#:
#: Мельче нельзя: у CDS чанк — это заявка в очередь, и ряд за сто лет при
#: месячном чанке стоил бы 1200 походов в очередь на один ответ.
CDS_SPAN_HOURS: Final = 365 * 24

#: Нарезка рядов CDS. `max_chunks` — сто лет из docs/API_CONTRACT.md §3 плюс
#: чанк на несовпадение границ: предел здесь не про объём ответа (его режет
#: потолок шагов), а про то, чтобы период задом наперёд или опечатка в годе не
#: развернулись в поход в очередь CDS на каждый год.
CDS_GRID: Final = Grid(
    epoch=datetime(canon.HISTORY_START_YEAR, 1, 1, tzinfo=UTC),
    step=timedelta(hours=1),
    span=CDS_SPAN_HOURS,
    max_chunks=102,
)

#: Разделитель имени поля и точки в ключе кэша. `cache.index.Key` места под
#: точку не имеет — и не должен: ключ описывает чанк источника, а у карт точки
#: нет вовсе. Поэтому точка едет внутри имени переменной, а не отдельным полем.
POINT_MARK: Final = "@"


def at_point(variable: str, lat: float, lon: float) -> str:
    """Имя переменной в ключе кэша для ряда в точке: `2t@55.75,37.5`.

    Точка округляется до узла сетки здесь же (`era5_cds.snap`), а не у того,
    кто зовёт: 37.61 и 37.62 — это один и тот же узел и обязаны быть одним
    ключом, иначе кэш рядов не попадает сам в себя ни разу.

    В `variable` едет **имя поля**, а не датасет: `dataset` у origin один на
    все точки, и отчёт по кэшу (`make cache-report`) остаётся списком датасетов,
    а не списком координат, которые кто-то когда-то спрашивал.
    """
    lat, lon = era5_cds.snap(lat, lon)
    # `+ 0.0` убирает минус у нуля: -0.0 и 0.0 — один узел, а в ключе это две
    # разные строки, то есть два чанка с одними и теми же данными.
    return f"{variable}{POINT_MARK}{lat + 0.0:g},{lon + 0.0:g}"


def split_point(text: str) -> tuple[str, float, float]:
    """Обратное к `at_point`: `2t@55.75,37.5` → `("2t", 55.75, 37.5)`."""
    variable, mark, point = text.partition(POINT_MARK)
    lat, comma, lon = point.partition(",")
    if not (mark and comma and variable):
        raise ValueError(f"не имя ряда в точке: {text!r}")
    try:
        return variable, float(lat), float(lon)
    except ValueError as bad:
        raise ValueError(f"не имя ряда в точке: {text!r}") from bad


class CdsOrigin:
    """ERA5 из CDS — ряды в точке за прошлое.

    Точка приезжает внутри имени переменной (`at_point`), поэтому origin один
    на весь сервис и живёт столько же, сколько приложение. Состояния у него
    нет: заявка в CDS — это поход в очередь, а не открытый архив, и хранить
    между запросами тут нечего.
    """

    name = "cds"
    dataset = "era5-series"

    def __init__(
        self,
        *,
        grid: Grid = CDS_GRID,
        retriever: era5_cds.Retriever | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.version = era5_cds.ADAPTER_VERSION
        self.grid = grid
        self._retriever = retriever
        self._clock = clock

    def source_version(self, chunk: int) -> str:
        """`era5-final`, `era5t` или `era5t-<дата>` — по концу чанка.

        По концу, а не по началу, как у `ArcoOrigin`: там чанк равен одному
        сроку и разницы нет, а здесь чанк длиной в год, и версия по его началу
        обещала бы финальный ERA5 там, где 364 дня из 365 моложе границы.

        Третья форма — для чанка, который ещё растёт: пока его конец в будущем,
        источник каждый день отдаёт на сутки больше, и ключ без даты означал бы
        вечно замороженный обрубок года. Дата в ключе делает растущий чанк
        новым объектом раз в сутки; старые уходят вытеснением. Точной
        инвалидации (`where source_version = 'era5t'`) они при этом не видны —
        и не должны: она про замену предварительных данных финальными, а не
        про то, что вчерашний обрубок больше не нужен.
        """
        now = self._clock()
        ends = self._ends(chunk)
        version = era5_arco.source_version(ends, now=now)
        return version if ends < now else f"{version}-{now:%Y%m%d}"

    def fetch(self, variable: str, chunk: int) -> bytes:
        """Ряд за период чанка. `NotYetError`, когда данных за него ещё нет."""
        name, lat, lon = split_point(variable)
        now = self._clock()
        begins, ends = self.grid.begins(chunk), self._ends(chunk)
        if begins >= now:
            raise NotYetError(f"{begins.isoformat()} ещё не наступил")
        try:
            # Конец обрезается по «сейчас»: у CDS запрос на будущие даты — это
            # отказ через очередь длиной в минуты, а не пустой ответ.
            series = era5_cds.read_series(
                name, lat, lon, begins, min(ends, now), retriever=self._retriever, now=now
            )
        except AdapterError as empty:
            # Пустой ответ на период, начавшийся только что, — это отставание
            # реанализа (~5 суток), а не сбой: срок начала чанка попал в
            # слепую зону. Разбирается это по полям отказа, а не по тексту, и
            # только этот случай: остальные отказы адаптера — настоящие.
            if empty.field == "csv" and empty.got == "no rows":
                raise NotYetError(f"{begins.isoformat()}..{ends.isoformat()}: {empty}") from empty
            raise
        return encode(series)

    def _ends(self, chunk: int) -> datetime:
        """Последний срок чанка. Именно последний, а не начало следующего:
        по нему решается версия, и час разницы здесь — это сутки в ключе."""
        return self.grid.begins(chunk + 1) - self.grid.step
