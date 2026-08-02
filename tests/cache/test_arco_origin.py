"""ERA5 из бакета через кэш — вторая половина приёмки BACKLOG 1.5.

Адаптер проверен отдельно (`tests/adapters/test_era5_arco.py`); здесь
проверяется, что он доезжает до пользователя через `cache.proxy`: чанк
кладётся на диск, второй запрос наружу не идёт, слепая зона запоминается.

Архив тот же самый (`tests/fixtures/arco.py`) — иначе «у адаптера работает, а
через кэш нет» поймать негде.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from adapters.era5_arco import FINAL, PRELIMINARY
from cache.chunks import decode, encode
from cache.index import Key, absent, open_index
from cache.origins import ARCO_GRID, REFRESH_AFTER, ArcoOrigin
from cache.proxy import NotYetError, serve
from contracts import canon
from tests.fixtures.arco import STEPS, build

NOW = datetime(2020, 12, 1, tzinfo=UTC)

#: Секунды, которыми живёт индекс кэша. К календарю фикстуры отношения не
#: имеют: возраст срока считает `clock`, а срок жизни записей — это `now`.
CLOCK = 1_800_000_000.0


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("arco") / "sfc.zarr", "2m_temperature")


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


def _origin(archive: Path, *, now: datetime = NOW) -> ArcoOrigin:
    return ArcoOrigin(archive, clock=lambda: now)


def test_a_slice_reaches_the_disk_through_the_cache(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Целиком путь: бакет, адаптер, кодек, файл в кэше — и обратно тем же
    Dataset. Кэш кладёт байты, не заглядывая внутрь, поэтому разобрать их
    обязан уметь тот, кто просил (docs/CACHE.md §3.1)."""
    served = serve(
        index, _origin(archive), "2t", STEPS[1], STEPS[1], root=tmp_path / "cache", now=CLOCK
    )

    got = decode(served.paths[0].read_bytes())

    assert served.misses == 1
    assert got["2t"].dims == ("time", "lat", "lon")
    assert got["2t"].attrs["units"] == canon.UNITS["2t"]
    assert np.array_equal(got["lon"].values, canon.LON)
    assert got["2t"].sel(lon=0.0).values[0][0] == pytest.approx(1000.0)


def test_the_second_request_does_not_touch_the_bucket(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Ради этого кэш и стоит. Считаются походы наружу, а не время ответа:
    каждый из них — это чтение чанка ARCO по сети."""
    origin = _origin(archive)
    root = tmp_path / "cache"

    first = serve(index, origin, "2t", STEPS[0], STEPS[0], root=root, now=CLOCK)
    second = serve(index, origin, "2t", STEPS[0], STEPS[0], root=root, now=CLOCK + 1)

    assert (first.misses, second.misses) == (1, 0)
    assert second.hit


def test_a_period_comes_out_chunk_per_hour(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Чанк ARCO равен часу, и период раскладывается по часам — по одному
    файлу на срок. Это и есть причина маленького `max_chunks`: год в точке
    отсюда стоил бы 8760 обращений."""
    served = serve(
        index, _origin(archive), "2t", STEPS[0], STEPS[-1], root=tmp_path / "cache", now=CLOCK
    )

    assert len(served.paths) == len(STEPS)
    assert len({path for path in served.paths}) == len(STEPS)


def test_the_blind_zone_is_remembered(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Срок новее архива — не сбой, а слепая зона реанализа. Отказ адаптера
    обязан доехать сюда типом кэша: иначе `serve` увидит незнакомое исключение
    и не запомнит отказ, а пользователь, листающий календарь, будет ходить в
    бакет на каждое движение."""
    ahead = STEPS[-1] + timedelta(hours=1)
    origin = _origin(archive)

    with pytest.raises(NotYetError):
        serve(index, origin, "2t", ahead, ahead, root=tmp_path / "cache", now=CLOCK)

    refusal = absent(index, _key(origin, "2t", ahead), now=CLOCK)
    assert refusal is not None and "новее архива" in refusal.reason


def test_the_archive_is_reopened_and_the_blind_zone_shrinks(tmp_path: Path) -> None:
    """Реанализ догоняет календарь, и origin обязан это заметить сам.

    `open_zarr` читает ось времени сразу: архив, открытый один раз навсегда, до
    конца жизни процесса уверен, что данных за вчера нет. Отказ в `absent`
    живёт шесть часов, прокси спрашивает снова — и получал бы тот же отказ до
    перезапуска сервиса.
    """
    path = build(tmp_path / "grows.zarr", "2m_temperature", times=STEPS[:2])
    ahead = STEPS[2]
    moment = [NOW]
    origin = ArcoOrigin(path, clock=lambda: moment[0])

    with pytest.raises(NotYetError):
        origin.fetch("2t", ARCO_GRID.holding(ahead))

    build(path, "2m_temperature", times=STEPS)
    moment[0] = NOW + REFRESH_AFTER

    got = decode(origin.fetch("2t", ARCO_GRID.holding(ahead)))
    assert got["time"].values[0] == np.datetime64(ahead.replace(tzinfo=None), "ns")


def test_the_key_carries_the_preliminary_mark(archive: Path) -> None:
    """`era5t` входит в ключ кэша, а не только в ответ: когда ERA5T перепишут
    финальным ERA5, инвалидация ищет чанки именно по этому слову
    (docs/CACHE.md §5). Один ключ на обе версии означал бы, что после замены
    пользователь ещё сутки получает старые числа."""
    fresh = _origin(archive, now=STEPS[0] + timedelta(days=1))
    old = _origin(archive, now=STEPS[0] + timedelta(days=365))
    chunk = ARCO_GRID.holding(STEPS[0])

    assert fresh.source_version(chunk) == PRELIMINARY
    assert old.source_version(chunk) == FINAL


def test_the_codec_keeps_what_the_pipeline_reads(
    archive: Path, index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Обход через байты не должен менять ни значения, ни оси, ни провенанс:
    дальше по конвейеру срез из кэша обязан быть неотличим от среза от
    адаптера."""
    served = serve(
        index, _origin(archive), "2t", STEPS[0], STEPS[0], root=tmp_path / "cache", now=CLOCK
    )
    got = decode(served.paths[0].read_bytes())
    again = decode(encode(got))

    assert got.attrs["source"] == FINAL
    assert got.attrs["grid"] == canon.GRID_NAME
    assert got["2t"].dtype == np.float32
    assert got["time"].values[0] == np.datetime64(STEPS[0].replace(tzinfo=None), "ns")
    assert np.array_equal(again["2t"].values, got["2t"].values)
    assert again.attrs == got.attrs


def _key(origin: ArcoOrigin, variable: str, moment: datetime) -> Key:
    """Ключ, под которым чанк ищет прокси. Собирается из самого origin: ключ,
    посчитанный в тесте по своим правилам, проверял бы тест, а не кэш."""
    chunk = origin.grid.holding(moment)
    return Key(origin.dataset, variable, origin.source_version(chunk), origin.version, str(chunk))
