"""Вытеснение — приёмка BACKLOG 3.3 и 3.4.

Критерии дословно: «при заполнении 85% чистка идёт до 70%, запиннённое остаётся
на месте» и «дорогие объекты выживают дольше дешёвых при равной давности».

Размеры здесь круглые и мелкие: проверяется решение, кого выкинуть, а не то,
что диск умеет хранить гигабайты.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cache.evict import HALF_LIFE_S, Limits, key_of, main, score, sweep
from cache.index import Entry, Key, entries, lookup, open_index, pin, record, total_bytes

NOW = 1_800_000_000.0

#: Ёмкость кэша: чистка стартует на 850, идёт до 700.
LIMITS = Limits(capacity_bytes=1000)


@pytest.fixture
def index(tmp_path: Path) -> sqlite3.Connection:
    return open_index(tmp_path / "cache.sqlite")


def _put(
    index: sqlite3.Connection,
    root: Path,
    chunk: str,
    *,
    size: int,
    age_s: float = 0.0,
    cost_ms: int = 200,
    pinned: bool = False,
) -> Key:
    """Положить чанк: файл на диск, строка в индекс."""
    key = Key("era5", "2t", "era5-final", "v1", chunk)
    path = root / f"{chunk}.chunk"
    path.write_bytes(b"x" * size)
    record(
        index,
        key,
        path=str(path),
        size=size,
        origin="arco",
        cost_ms=cost_ms,
        pinned=pinned,
        now=NOW - age_s,
    )
    return key


def test_cleaning_starts_at_the_high_mark_and_stops_at_the_low_one(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Сам критерий приёмки 3.3."""
    for number in range(9):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600)

    swept = sweep(index, limits=LIMITS, now=NOW)

    assert swept.triggered is True
    assert (swept.before, swept.after) == (900, 700)
    assert swept.enough is True
    assert total_bytes(index) == 700
    assert swept.freed == 200 and len(swept.removed) == 2


def test_below_the_high_mark_nothing_is_touched(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Две ватермарки ради этого и заведены: с одной чистка срабатывала бы на
    каждом следующем чанке — то есть в момент записи нового прогона."""
    for number in range(8):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600)

    swept = sweep(index, limits=LIMITS, now=NOW)

    assert swept.triggered is False
    assert (swept.removed, swept.freed) == ((), 0)
    assert total_bytes(index) == 800


def test_the_pinned_object_stays_in_place(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Вторая половина критерия. Запиннённое — самое старое и самое дешёвое,
    то есть первый кандидат по любой оценке: прогноз спрашивают реже популярной
    точки, и чистый LRU однажды удалил бы именно его."""
    ancient = _put(index, tmp_path, "old", size=300, age_s=30 * 86_400, cost_ms=1, pinned=True)
    for number in range(6):
        _put(index, tmp_path, str(number), size=100, age_s=number * 60)

    swept = sweep(index, limits=LIMITS, now=NOW)

    assert lookup(index, ancient) is not None
    assert (tmp_path / "old.chunk").is_file()
    assert ancient.text() not in swept.removed
    assert total_bytes(index) == 700


def test_an_expensive_object_outlives_a_cheap_one_of_the_same_age(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Критерий приёмки 3.4. Ряд CDS возвращается за минуты — очередь; чанк
    ARCO за двести миллисекунд. По одной давности первым вылетело бы дорогое, и
    следующий запрос за ним встал бы в ту же очередь."""
    dear = _put(index, tmp_path, "cds", size=100, age_s=86_400, cost_ms=120_000)
    cheap = _put(index, tmp_path, "arco", size=100, age_s=86_400, cost_ms=200)
    for number in range(7):
        _put(index, tmp_path, str(number), size=100, age_s=1.0, cost_ms=200)

    sweep(index, limits=LIMITS, now=NOW)

    assert lookup(index, dear) is not None
    assert lookup(index, cheap) is None


def test_a_stale_object_goes_before_a_fresh_one_of_the_same_price(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Давность из оценки не выпала: у запросов есть локальность, и чанк,
    нетронутый неделю, вытесняется раньше вчерашнего."""
    stale = _put(index, tmp_path, "week", size=100, age_s=7 * 86_400)
    fresh = _put(index, tmp_path, "minute", size=100, age_s=60)
    for number in range(7):
        _put(index, tmp_path, str(number), size=100, age_s=1800)

    sweep(index, limits=LIMITS, now=NOW)

    assert lookup(index, stale) is None
    assert lookup(index, fresh) is not None


def test_a_big_object_goes_before_a_small_one_of_the_same_price(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Освобождается место, а не количество строк: при равной цене и давности
    крупный чанк держать дороже (docs/CACHE.md §3.2)."""
    fat = _put(index, tmp_path, "fat", size=300, age_s=3600)
    thin = _put(index, tmp_path, "thin", size=100, age_s=3600)
    for number in range(5):
        _put(index, tmp_path, str(number), size=100, age_s=1800)

    sweep(index, limits=LIMITS, now=NOW)

    assert lookup(index, fat) is None
    assert lookup(index, thin) is not None


def test_the_file_goes_away_with_the_row(index: sqlite3.Connection, tmp_path: Path) -> None:
    """Строка без файла — это байты, которые вытеснение позже спишет впустую;
    файл без строки — место, которое никто уже не найдёт."""
    for number in range(9):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600)

    swept = sweep(index, limits=LIMITS, now=NOW)

    left = sorted(item.name for item in tmp_path.glob("*.chunk"))
    assert len(left) == 7
    assert sorted(f"{found.chunk}.chunk" for found in entries(index)) == left
    assert "era5/2t/era5-final/v1/8" in swept.removed


def test_a_file_deleted_by_hand_still_frees_its_row(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Диск чистят мимо индекса. Строка обязана уйти всё равно, иначе она вечно
    занимает место, которого не занимает, и чистка перестаёт находить байты."""
    gone = _put(index, tmp_path, "gone", size=300, age_s=30 * 86_400)
    (tmp_path / "gone.chunk").unlink()
    for number in range(6):
        _put(index, tmp_path, str(number), size=100, age_s=number * 60)

    swept = sweep(index, limits=LIMITS, now=NOW)

    assert lookup(index, gone) is None
    assert swept.freed == 300 and total_bytes(index) == 600


def test_pinning_alone_can_leave_the_cache_over_the_mark(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Отвели меньше места, чем занимает несменяемое. Чистка не обязана
    выкидывать запиннённое, но обязана сказать, что места не нашла: молча
    вернуть «готово» — значит спрятать переполнение до отказа записи."""
    for number in range(9):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600, pinned=True)

    swept = sweep(index, limits=LIMITS, now=NOW)

    assert swept.triggered is True
    assert swept.removed == ()
    assert swept.enough is False
    assert total_bytes(index) == 900


def test_a_pin_taken_off_makes_the_object_ordinary(
    index: sqlite3.Connection, tmp_path: Path
) -> None:
    """Пиннинг снимается: прогон, ушедший из указателей (`storage.rotate`),
    держать больше незачем."""
    freed = _put(index, tmp_path, "old", size=300, age_s=30 * 86_400, cost_ms=1, pinned=True)
    for number in range(6):
        _put(index, tmp_path, str(number), size=100, age_s=number * 60)
    pin(index, freed, pinned=False)

    sweep(index, limits=LIMITS, now=NOW)

    assert lookup(index, freed) is None
    assert total_bytes(index) == 600


def test_the_dry_run_deletes_nothing(
    index: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Команда, сносящая гигабайты, обязана уметь сначала показать, что именно:
    иначе первый её запуск на боевом диске и есть проверка."""
    for number in range(9):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600)
    index.close()

    code = main([str(tmp_path / "cache.sqlite"), "--capacity", "1000", "--dry-run"])

    assert code == 0
    assert capsys.readouterr().out.count("снесла бы") == 2
    assert len(sorted(tmp_path.glob("*.chunk"))) == 9


def test_the_command_cleans_and_reports(
    index: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """То же вытеснение, но руками дежурного и по расписанию."""
    for number in range(9):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600)
    index.close()

    code = main([str(tmp_path / "cache.sqlite"), "--capacity", "1000"])

    assert code == 0
    assert "занято 700 из 1000 байт" in capsys.readouterr().out
    assert len(sorted(tmp_path.glob("*.chunk"))) == 7


def test_the_command_complains_when_the_pinned_part_does_not_fit(
    index: sqlite3.Connection, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ненулевой код возврата — это то, что заметит расписание. Молчаливое
    «готово» здесь прячет переполнение до отказа записи."""
    for number in range(9):
        _put(index, tmp_path, str(number), size=100, age_s=number * 3600, pinned=True)
    index.close()

    code = main([str(tmp_path / "cache.sqlite"), "--capacity", "1000"])

    assert code == 1
    assert "не дочистили" in capsys.readouterr().out


def test_the_score_does_not_divide_by_a_zero_size(index: sqlite3.Connection) -> None:
    """Пустой файл в кэше означает сломанный origin. Делить на него нельзя, а
    вытеснять — наравне с прочими."""
    empty = Entry(
        key="era5/2t/era5-final/v1/0",
        dataset="era5",
        variable="2t",
        source_version="era5-final",
        adapter_version="v1",
        chunk="0",
        path="/dev/null",
        bytes=0,
        last_access=NOW,
        access_count=0,
        origin="arco",
        cost_ms=200,
        pinned=False,
        created_at=NOW,
    )

    assert score(empty, now=NOW) == pytest.approx(200.0)
    assert key_of(empty) == Key("era5", "2t", "era5-final", "v1", "0")


def test_the_half_life_halves_the_score(index: sqlite3.Connection) -> None:
    """Затухание — это то, что делает оценку убывающей во времени; ошибка в нём
    незаметна на любом одиночном сравнении."""
    entry = Entry(
        key="era5/2t/era5-final/v1/0",
        dataset="era5",
        variable="2t",
        source_version="era5-final",
        adapter_version="v1",
        chunk="0",
        path="/dev/null",
        bytes=99,
        last_access=NOW - HALF_LIFE_S,
        access_count=0,
        origin="arco",
        cost_ms=200,
        pinned=False,
        created_at=NOW,
    )

    assert score(entry, now=NOW) == pytest.approx(1.0)
    # Доступ в будущем — это часы, переведённые назад, а не отрицательный
    # возраст с оценкой выше любой настоящей.
    assert score(entry, now=entry.last_access - 10_000) == pytest.approx(2.0)
