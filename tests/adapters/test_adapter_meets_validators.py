"""Шов между задачей 4 и задачей 5: канон адаптера и канон валидаторов — один.

Обе половины сейчас утверждают одно и то же, но каждая своими словами:
адаптер ставит `units` из `canon.UNITS`, валидатор сверяет `units` с
`canon.UNITS`, и ни один тест не сводит их вместе. Разойдись они завтра —
например, добавь адаптер свой множитель и забудь про подпись, — обе стороны
остались бы зелёными по отдельности.

Проверок, применимых к одному сообщению, здесь меньше, чем всего:
`fields_present` требует все 69 полей, `levels_match` — все 13 уровней, а в
сообщении одно поле и один уровень. Эти две названы явно, остальные обязаны
пройти без оговорок.
"""

from pathlib import Path

import pytest

pytest.importorskip("cfgrib", reason="cfgrib тянет бинарный eccodes; см. docs/SETUP.md §4")

from collections.abc import Callable

import xarray as xr

from adapters import ecmwf, gfs
from validators.result import Check
from validators.semantics import check_semantics
from validators.structure import check_structure

GRIB = Path(__file__).resolve().parents[1] / "fixtures" / "grib"
RETRIEVED = "2026-08-01T07:41:12Z"

Reader = Callable[..., xr.Dataset]

#: Проверки целого набора, которым одного сообщения не хватает по существу.
WHOLE_DATASET_ONLY = frozenset({"fields_present", "levels_match"})


def _checks(read: Reader, name: str) -> list[Check]:
    ds = read(GRIB / name, source_url=f"test://{name}", retrieved_at=RETRIEVED)
    return check_structure(ds) + check_semantics(ds)


@pytest.mark.parametrize(
    ("read", "name"),
    [
        (gfs.read_message, "gfs_t2m.grib2"),
        (gfs.read_message, "gfs_apcp_f006.grib2"),
        (ecmwf.read_message, "ecmwf_2t_6h.grib2"),
        (ecmwf.read_message, "ecmwf_tp_6h.grib2"),
        (ecmwf.read_message, "ecmwf_t850_6h.grib2"),
    ],
)
def test_adapter_output_passes_every_check_that_applies_to_one_message(
    read: Reader, name: str
) -> None:
    failures = [c for c in _checks(read, name) if not c.passed]
    unexpected = [c.message for c in failures if c.name not in WHOLE_DATASET_ONLY]
    assert unexpected == []


def test_the_named_exceptions_really_are_the_only_ones() -> None:
    """Список исключений — не «пока не работает», а «неприменимо к сообщению».
    Если он вдруг начнёт проходить, значит проверка ослабла и её надо чинить,
    а не радоваться зелёному."""
    names = {c.name for c in _checks(gfs.read_message, "gfs_t2m.grib2") if not c.passed}
    assert names == {"fields_present"}
