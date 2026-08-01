"""Полнота таблиц имён.

В фикстурах есть только `2t`, `t` и `tp`: остальные поля из настоящих файлов
здесь не проверяются, и это известный предел (`tests/fixtures/README.md`).
Но забытая строка в таблице — отказ уже на приёме, а не «поле поехало», и
поймать её можно без единого байта данных: канон знает, что должно быть.
"""

import pytest

from adapters import ecmwf, gfs
from contracts import canon

ADAPTERS = {"gfs": gfs, "ecmwf": ecmwf}

#: Приземные поля резервного чекпоинта `aurora-0.25-finetuned`. GFS отвечает
#: только за них: он отладочный путь, и держать в нём все 18 входов Aurora 1.5
#: значило бы писать по документации NOMADS таблицу, которую нечем проверить.
FALLBACK_SURFACE = ("2t", "10u", "10v", "msl")


def test_ecmwf_produces_every_ingested_field_except_the_documented_gaps() -> None:
    """Дыры в Open Data названы поимённо, а не «чего-то не хватает».

    `lcc`/`mcc`/`hcc` берутся из потока `aifs-single` тем же адаптером, поэтому
    они обязаны быть в таблице; `ci` не публикуется вовсе и приходит из ERA5T
    другим адаптером — его в таблице быть не должно.
    """
    produced = set(ecmwf.RENAMES.values())
    required = set(canon.SURFACE_INGESTED_VARS) | set(canon.ATMOS_VARS) | {"tp"}
    missing = required - produced
    assert missing == set(ecmwf.FROM_ERA5T), sorted(missing)
    assert set(ecmwf.STREAMS) <= produced


def test_gfs_produces_the_fallback_set() -> None:
    produced = set(gfs.RENAMES.values())
    required = set(FALLBACK_SURFACE) | set(canon.ATMOS_VARS) | {"tp"}
    assert required <= produced, sorted(required - produced)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_the_table_maps_onto_the_canon_and_nowhere_else(name: str) -> None:
    """Имя, которого нет в каноне, — это опечатка, которая обнаружится только
    на записи в хранилище, где уже поздно."""
    adapter = ADAPTERS[name]
    known = set(canon.UNITS)
    assert set(adapter.RENAMES.values()) <= known, sorted(set(adapter.RENAMES.values()) - known)


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_names_are_not_mapped_twice(name: str) -> None:
    """Два имени источника на одно каноническое означают, что при склейке
    сообщений одно поле молча затрёт другое."""
    adapter = ADAPTERS[name]
    values = list(adapter.RENAMES.values())
    assert len(values) == len(set(values)), sorted({v for v in values if values.count(v) > 1})


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_scaled_variables_are_variables(name: str) -> None:
    adapter = ADAPTERS[name]
    assert set(adapter.SCALES) <= set(adapter.RENAMES.values())
