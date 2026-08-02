"""Негативное кэширование — приёмка BACKLOG 3.6.

Критерий дословно: «запрос на дату из „слепой зоны“ ERA5 не ходит наружу
повторно». Считаются походы к источнику, а не время ответа: слепая зона — это
про то, кому достанется очередь CDS, а не про то, кому быстро.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cache.index import (
    NEGATIVE_TTL_S,
    Key,
    absent,
    entries,
    expire_absent,
    forget_absent,
    mark_absent,
    open_index,
    total_bytes,
)
from cache.proxy import Grid, NotYetError, moment, serve

MAPS = Grid(epoch=datetime(1940, 1, 1, tzinfo=UTC), step=timedelta(hours=1), span=6, max_chunks=8)

NOW = 1_800_000_000.0

#: Дата из слепой зоны: реанализ отстаёт на ~5 суток (docs/CACHE.md §3.3).
BLIND = "2026-08-01T00:00:00Z"

#: Чанк, в который попадает эта дата, и ключ, под которым его ищет прокси.
BLIND_CHUNK = MAPS.holding(moment(BLIND))
KEY = Key("era5", "2t", "era5t", "v1", str(BLIND_CHUNK))


class BlindOrigin:
    """Источник, у которого последних суток ещё нет.

    `since` — номер чанка, начиная с которого источник отвечает отказом. Это и
    есть слепая зона: не сбой, а нормальное состояние реанализа.
    """

    name = "cds"
    dataset = "era5"
    version = "v1"
    grid = MAPS

    def __init__(self, *, since: int | None = None) -> None:
        self.calls: list[int] = []
        self.since = since

    def source_version(self, chunk: int) -> str:
        # Край периода — это всегда ERA5T: финальным ERA5 он станет через
        # месяцы. Версия не зависит от `since` намеренно — данные, доехавшие
        # после отказа, приходят под тем же ключом, по которому им отказали.
        return "era5t"

    def fetch(self, variable: str, chunk: int) -> bytes:
        self.calls.append(chunk)
        if self.since is not None and chunk >= self.since:
            raise NotYetError("нет в ERA5: реанализ отстаёт")
        return f"{variable}:{chunk}".encode()


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


def _serve(
    index: sqlite3.Connection,
    origin: BlindOrigin,
    root: Path,
    start: str,
    stop: str = "",
    *,
    now: float = NOW,
) -> None:
    serve(index, origin, "2t", moment(start), moment(stop or start), root=root, now=now)


def _blind() -> BlindOrigin:
    """Источник, слепой ровно на запрошенном чанке."""
    return BlindOrigin(since=BLIND_CHUNK)


def test_a_date_from_the_blind_zone_does_not_go_outside_twice(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Сам критерий приёмки. Без отрицательного кэша один человек, листающий
    календарь на фронтенде, превращает каждое движение в поход наружу — а
    отвечает на них очередь CDS, общая на весь сервис."""
    origin = _blind()

    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND)
    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND, now=NOW + 60)

    assert len(origin.calls) == 1


def test_the_refusal_expires_and_the_source_is_asked_again(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Отказ живёт шесть часов, а не вечно: данные приходят, и кэш, который об
    этом не узнает, — это уже не задержка реанализа, а своя собственная."""
    origin = _blind()

    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND)
    origin.since = None

    _serve(index, origin, tmp_path, BLIND, now=NOW + NEGATIVE_TTL_S + 1)

    assert len(origin.calls) == 2
    assert absent(index, KEY, now=NOW + NEGATIVE_TTL_S + 1) is None


def test_the_refusal_holds_right_up_to_its_deadline(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Граница проверяется отдельно: `>=` вместо `>` укоротил бы срок на нуль
    секунд в тесте и на шесть часов там, где отказы приходят пачкой."""
    origin = _blind()

    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND)

    assert absent(index, KEY, now=NOW + NEGATIVE_TTL_S - 1) is not None
    assert absent(index, KEY, now=NOW + NEGATIVE_TTL_S) is None


def test_a_refusal_takes_no_place_in_the_cache(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Отказ — не объект на диске. Строка в `cache_index` означает файл: у неё
    есть размер, путь и цена, по ним считает вытеснение. Отказ, записанный
    туда же, был бы исключением, про которое обязан помнить каждый читатель
    таблицы."""
    origin = _blind()

    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND)

    assert total_bytes(index) == 0
    assert list(entries(index)) == []
    assert sorted(tmp_path.rglob("*.chunk*")) == []


def test_a_refusal_for_one_chunk_does_not_block_its_neighbour(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Слепая зона — это край периода, а не период. Соседний чанк лежит в
    источнике и обязан дойти до пользователя."""
    origin = _blind()
    earlier = moment(BLIND) - timedelta(hours=6)

    serve(index, origin, "2t", earlier, earlier, root=tmp_path, now=NOW)
    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND)

    assert len(list(entries(index))) == 1
    assert absent(index, KEY, now=NOW) is not None


def test_arrived_data_clears_the_stale_refusal(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Протухшая строка никого не задержит — `absent` смотрит на срок, — но
    осталась бы в таблице навсегда: повторно про эту дату уже не спросят."""
    origin = _blind()

    with pytest.raises(NotYetError):
        _serve(index, origin, tmp_path, BLIND)
    origin.since = None
    _serve(index, origin, tmp_path, BLIND, now=NOW + NEGATIVE_TTL_S + 1)

    assert index.execute("select count(*) from absent").fetchone()[0] == 0


def test_a_repeated_refusal_prolongs_the_row_instead_of_adding_one(
    index: sqlite3.Connection,
) -> None:
    """Пользователь, который тычет в завтрашнюю дату, обновляет свой же отказ,
    а не растит таблицу."""
    mark_absent(index, KEY, reason="нет в ERA5", now=NOW)
    later = mark_absent(index, KEY, reason="нет в ERA5", now=NOW + 3600)

    stored = absent(index, KEY, now=NOW + 3600)

    assert index.execute("select count(*) from absent").fetchone()[0] == 1
    assert later.until == NOW + 3600 + NEGATIVE_TTL_S
    assert stored is not None and stored.until == later.until
    # `since` остаётся от первого отказа: по нему видно, как давно ждут данных,
    # а обновление продлевает срок, а не переписывает историю.
    assert stored.since == NOW


def test_the_reason_survives_to_the_person_on_duty(index: sqlite3.Connection) -> None:
    """«Нет в ERA5» и «origin отказал» чинятся по-разному, а на пути запроса
    разницы уже не видно."""
    mark_absent(index, KEY, reason="нет в ERA5: реанализ отстаёт", now=NOW)

    found = absent(index, KEY, now=NOW)

    assert found is not None and found.reason == "нет в ERA5: реанализ отстаёт"


def test_expiring_takes_the_stale_rows_and_leaves_the_live_ones(
    index: sqlite3.Connection,
) -> None:
    """Чистка отказов — уборка, а не решение: `absent` и без неё не отдаёт
    протухшее. Поэтому живое она трогать не имеет права."""
    stale = Key("era5", "2t", "era5t", "v1", "старый")
    mark_absent(index, stale, reason="нет в ERA5", now=NOW - NEGATIVE_TTL_S - 1)
    mark_absent(index, KEY, reason="нет в ERA5", now=NOW)

    assert expire_absent(index, now=NOW) == 1
    assert absent(index, KEY, now=NOW) is not None


def test_a_refusal_taken_off_by_hand_is_gone(index: sqlite3.Connection) -> None:
    """Дежурный, знающий, что данные доехали, не обязан ждать шесть часов."""
    mark_absent(index, KEY, reason="нет в ERA5", now=NOW)

    assert forget_absent(index, KEY) is True
    assert forget_absent(index, KEY) is False
    assert absent(index, KEY, now=NOW) is None
