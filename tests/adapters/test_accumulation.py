"""Четвёртая ловушка: осадки за шаг.

Проверяется и на настоящих парах сообщений (ECMWF `0-6`/`0-12` против GFS
`0-6`/`6-12`), и на синтетике — там, где нужно подсунуть пару, которой в
фикстурах нет: пропущенный шаг, обратный порядок, чужой прогон.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from adapters.accumulation import StepRange, interval_amount
from adapters.errors import AdapterError

GRIB = Path(__file__).resolve().parents[1] / "fixtures" / "grib"
RETRIEVED = "2026-08-01T07:41:12Z"


def _field(values: list[float], start: int, end: int) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=np.float32),
        dims="lon",
        attrs={"units": "m", "start_step": start, "end_step": end, "step_range": f"{start}-{end}"},
    )


def test_accumulation_from_the_start_of_the_run_is_differenced() -> None:
    six = _field([1.0, 4.0], 0, 6)
    twelve = _field([3.0, 4.0], 0, 12)
    amount = interval_amount(six, twelve)
    assert amount.values.tolist() == [2.0, 0.0]
    assert amount.attrs["step_range"] == "6-12"


def test_accumulation_over_the_interval_is_taken_as_is() -> None:
    """Разность здесь дала бы -1 в первой точке: два независимых интервала."""
    six = _field([1.0, 4.0], 0, 6)
    twelve = _field([0.0, 2.0], 6, 12)
    amount = interval_amount(six, twelve)
    assert amount.values.tolist() == [0.0, 2.0]
    assert amount.attrs["step_range"] == "6-12"


def test_a_gap_between_the_messages_is_a_refusal() -> None:
    """`0-6` и `12-18`: пропущенный шаг даёт правдоподобное поле не за тот
    интервал — молчать об этом нельзя."""
    with pytest.raises(AdapterError) as raised:
        interval_amount(_field([1.0], 0, 6), _field([1.0], 12, 18))
    assert raised.value.field == "step_range"
    assert "0-6 then 12-18" in str(raised.value)


def test_the_wrong_order_is_a_refusal() -> None:
    with pytest.raises(AdapterError) as raised:
        interval_amount(_field([3.0], 0, 12), _field([1.0], 0, 6))
    assert "strictly increasing" in str(raised.value)


def test_a_field_without_step_attributes_is_a_refusal() -> None:
    bare = xr.DataArray(np.zeros(2, dtype=np.float32), dims="lon")
    with pytest.raises(AdapterError) as raised:
        StepRange.of(bare)
    assert raised.value.field == "start_step"


@pytest.mark.parametrize(
    ("module", "first", "second", "expected"),
    [
        ("ecmwf", "ecmwf_tp_6h.grib2", "ecmwf_tp_12h.grib2", "difference"),
        ("gfs", "gfs_apcp_f006.grib2", "gfs_apcp_f012.grib2", "as is"),
    ],
)
def test_real_pairs_take_the_rule_from_the_header(
    module: str, first: str, second: str, expected: str
) -> None:
    """Одно и то же правило, применённое к двум источникам, даёт разные
    действия — и в обоих случаях неотрицательное поле осадков."""
    pytest.importorskip("cfgrib", reason="cfgrib тянет бинарный eccodes; см. docs/SETUP.md §4")
    from adapters import ecmwf, gfs

    adapter = {"ecmwf": ecmwf, "gfs": gfs}[module]
    six = adapter.read_message(GRIB / first, source_url="test://six", retrieved_at=RETRIEVED)
    twelve = adapter.read_message(GRIB / second, source_url="test://twelve", retrieved_at=RETRIEVED)

    amount = interval_amount(six["tp"], twelve["tp"])
    assert float(amount.min()) >= 0.0
    assert amount.attrs["step_range"] == "6-12"

    # По массивам, а не по DataArray: у шагов +6 ч и +12 ч разное `time`,
    # и xarray, выравнивая операнды, вернул бы пустое поле.
    differenced = float((twelve["tp"].values - six["tp"].values).min())
    if expected == "difference":
        assert float(amount.min()) == pytest.approx(differenced)
    else:
        assert differenced < 0.0, "фикстура перестала показывать разницу источников"
        assert float(amount.max()) == pytest.approx(float(twelve["tp"].max()))
