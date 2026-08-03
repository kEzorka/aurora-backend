"""Вход Aurora: ровно два срока, канонический порядок и официальная статика."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from contracts import canon
from pipeline.state import State, assemble_state, load_static

LAT = np.array([1.0, 0.0], dtype=np.float32)
LON = np.array([10.0, 10.25, 10.5], dtype=np.float32)
TIMES = np.asarray(["2026-08-02T18", "2026-08-03T00"], dtype="datetime64[ns]")


def _analysis() -> xr.Dataset:
    surface = {
        name: (("time", "lat", "lon"), np.zeros((2, 2, 3), dtype=np.float32))
        for name in canon.SURFACE_INGESTED_VARS
    }
    atmosphere = {
        name: (
            ("time", "level", "lat", "lon"),
            np.zeros((2, len(canon.PRESSURE_LEVELS), 2, 3), dtype=np.float32),
        )
        for name in canon.ATMOS_VARS
    }
    return xr.Dataset(
        {**surface, **atmosphere},
        coords={
            "time": TIMES,
            "level": np.asarray(canon.PRESSURE_LEVELS),
            "lat": LAT,
            "lon": LON,
        },
    )


def _static(count: int = 36) -> dict[str, object]:
    return {f"field_{index}": index for index in range(count)}


def _assemble(dataset: xr.Dataset | None = None, static: dict[str, object] | None = None) -> State:
    return assemble_state(
        _analysis() if dataset is None else dataset,
        _static() if static is None else static,
        init_time="2026-08-03T00:00:00Z",
        expected_lat=LAT,
        expected_lon=LON,
    )


def test_state_preserves_the_contract_order_and_shapes() -> None:
    state = _assemble()
    assert tuple(state.surface) == canon.SURFACE_INGESTED_VARS
    assert tuple(state.atmosphere) == canon.ATMOS_VARS
    assert state.surface["2t"].shape == (2, 2, 3)
    assert state.atmosphere["t"].shape == (2, 13, 2, 3)
    assert len(state.static) == 36


def test_missing_field_is_never_replaced_with_zero() -> None:
    with pytest.raises(ValueError, match="ci"):
        _assemble(_analysis().drop_vars("ci"))


@pytest.mark.parametrize(
    "times",
    [
        ["2026-08-03T00"],
        ["2026-08-02T19", "2026-08-03T00"],
        ["2026-08-02T18", "2026-08-03T06"],
    ],
)
def test_times_are_exactly_init_minus_six_hours_and_init(times: list[str]) -> None:
    dataset = (
        _analysis()
        .isel(time=slice(0, len(times)))
        .assign_coords(time=np.asarray(times, dtype="datetime64[ns]"))
    )
    with pytest.raises(ValueError, match="times"):
        _assemble(dataset)


def test_pressure_level_order_is_not_sorted_or_guessed() -> None:
    dataset = _analysis().assign_coords(level=np.asarray(tuple(reversed(canon.PRESSURE_LEVELS))))
    with pytest.raises(ValueError, match="level"):
        _assemble(dataset)


def test_official_static_pickle_must_have_all_36_fields() -> None:
    with pytest.raises(ValueError, match="36"):
        _assemble(static=_static(35))


def test_static_loader_rejects_non_mapping_and_accepts_a_mapping(tmp_path: Path) -> None:
    valid = tmp_path / "valid.pickle"
    invalid = tmp_path / "invalid.pickle"
    valid.write_bytes(pickle.dumps(_static()))
    invalid.write_bytes(pickle.dumps([1, 2, 3]))

    assert len(load_static(valid)) == 36
    with pytest.raises(ValueError, match="mapping"):
        load_static(invalid)
