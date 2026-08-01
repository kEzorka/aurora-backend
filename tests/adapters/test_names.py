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


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_canonical_variable_can_be_produced(name: str) -> None:
    adapter = ADAPTERS[name]
    produced = set(adapter.RENAMES.values())
    required = set(canon.SURFACE_VARS) | set(canon.ATMOS_VARS) | {"tp"}
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
