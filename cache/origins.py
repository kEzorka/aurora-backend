"""Живые источники под `cache.proxy.Origin` (BACKLOG 1.5, 3.2).

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

from adapters import era5_arco
from adapters.errors import NotYetInSourceError
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


class ArcoOrigin:
    """ERA5 из публичного бакета — карты за прошлое.

    Архив открывается один раз на объект: `open_zarr` читает описание всех
    переменных за все восемьдесят лет, и делать это на каждый чанк значит
    платить за метаданные больше, чем за сами данные.
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
    ) -> None:
        self.version = era5_arco.ADAPTER_VERSION
        self.grid = grid
        self._source = source
        self._clock = clock
        self._archive: xr.Dataset | None = None

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
        if self._archive is None:
            self._archive = era5_arco.open_archive(self._source)
        return self._archive
