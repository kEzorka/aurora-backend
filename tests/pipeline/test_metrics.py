"""Метрики ловят научные ошибки: веса широты, срок и baseline."""

import numpy as np
import pytest
import xarray as xr

from pipeline.metrics import (
    anomaly_correlation,
    climatology_for,
    latitude_weighted_rmse,
    markdown,
    persistence,
    score_forecast,
)

LAT = np.array([90.0, 0.0, -90.0])
LON = np.array([0.0, 90.0])


def _field(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})


def _dataset(times: list[str], values: list[float]) -> xr.Dataset:
    field = np.stack([np.full((3, 2), value) for value in values])
    return xr.Dataset(
        {"2t": (("time", "lat", "lon"), field)},
        coords={"time": np.asarray(times, dtype="datetime64[ns]"), "lat": LAT, "lon": LON},
    )


def test_rmse_of_a_uniform_error_is_the_error() -> None:
    truth = _field(np.zeros((3, 2)))
    assert latitude_weighted_rmse(truth + 2, truth) == pytest.approx(2)


def test_polar_cells_do_not_outvote_the_equator() -> None:
    truth = _field(np.zeros((3, 2)))
    forecast = truth.copy()
    forecast.loc[{"lat": 90.0}] = 1_000
    assert latitude_weighted_rmse(forecast, truth) == pytest.approx(0, abs=1e-5)


def test_shifted_coordinates_are_an_error_not_silently_dropped() -> None:
    truth = _field(np.zeros((3, 2)))
    shifted = truth.assign_coords(lon=[0.25, 90.25])
    with pytest.raises(ValueError):
        latitude_weighted_rmse(shifted, truth)


def test_acc_is_one_for_equal_anomalies_and_minus_one_for_opposites() -> None:
    climate = _field(np.zeros((3, 2)))
    anomaly = _field(np.array([[1, -1], [2, -2], [1, -1]], dtype=float))
    assert anomaly_correlation(anomaly, anomaly, climate) == pytest.approx(1)
    assert anomaly_correlation(-anomaly, anomaly, climate) == pytest.approx(-1)


def test_persistence_repeats_exactly_one_initial_state() -> None:
    initial = _dataset(["2026-01-01T00"], [7])
    times = xr.DataArray(
        np.asarray(["2026-01-01T06", "2026-01-01T12"], dtype="datetime64[ns]"),
        dims=("time",),
    )
    got = persistence(initial, times)
    assert got["2t"].shape == (2, 3, 2)
    np.testing.assert_array_equal(got["2t"], 7)


def test_climatology_averages_years_and_selects_calendar_month() -> None:
    monthly = _dataset(
        ["2024-01-01", "2025-01-01", "2024-02-01", "2025-02-01"],
        [0, 2, 10, 14],
    )
    times = xr.DataArray(
        np.asarray(["2026-01-15", "2026-02-15"], dtype="datetime64[ns]"), dims=("time",)
    )
    got = climatology_for(monthly, times)
    np.testing.assert_array_equal(got["2t"][:, 1, 0], [1, 12])


def test_score_table_compares_model_with_both_baselines() -> None:
    forecast = _dataset(["2026-01-01T06", "2026-01-01T12"], [2, 4])
    truth = _dataset(["2026-01-01T00", "2026-01-01T06", "2026-01-01T12"], [0, 3, 6])
    monthly = _dataset(["2024-01-01", "2025-01-01"], [1, 1])

    result = score_forecast(forecast, truth, monthly, "2t", init_time="2026-01-01T00:00:00Z")

    assert [row.lead_hours for row in result] == [6, 12]
    assert result[0].rmse == pytest.approx(1)
    assert result[0].persistence_rmse == pytest.approx(3)
    assert result[0].climatology_rmse == pytest.approx(2)
    assert result[0].skill_vs_persistence == pytest.approx(2 / 3)
    assert "+12h" in markdown(result, "2t")


def test_non_positive_lead_is_rejected() -> None:
    forecast = _dataset(["2026-01-01T00"], [1])
    truth = _dataset(["2026-01-01T00"], [1])
    monthly = _dataset(["2024-01-01"], [1])
    with pytest.raises(ValueError, match="positive"):
        score_forecast(forecast, truth, monthly, "2t", init_time="2026-01-01T00:00:00Z")
