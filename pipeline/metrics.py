"""Научные метрики и базовые уровни прогноза (BACKLOG 4.5–4.6).

Вес по широте обязателен: без ``cos(lat)`` одна ячейка у полюса весит столько
же, сколько у экватора, хотя представляет почти нулевую площадь. Все входы
выравниваются точным join — сдвиг срока или координаты должен быть ошибкой, а
не NaN, который потом исчезнет из среднего.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr


class Metric(NamedTuple):
    lead_hours: int
    rmse: float
    acc: float
    persistence_rmse: float
    climatology_rmse: float
    skill_vs_persistence: float


def latitude_weighted_rmse(forecast: xr.DataArray, truth: xr.DataArray) -> float:
    """RMSE с весом площади ``cos(lat)`` по всем непустым осям."""
    predicted, observed = _aligned(forecast, truth)
    weights = _weights(predicted)
    squared = (predicted.astype("float64") - observed.astype("float64")) ** 2
    valid = squared.notnull()
    expanded = weights.broadcast_like(squared)
    denominator = expanded.where(valid).sum()
    if float(denominator) <= 0:
        raise ValueError("rmse: no finite overlapping values")
    result = np.sqrt((squared * expanded).where(valid).sum() / denominator)
    return float(result)


def anomaly_correlation(
    forecast: xr.DataArray, truth: xr.DataArray, climatology: xr.DataArray
) -> float:
    """ACC аномалий относительно той же климатологии с широтными весами."""
    predicted, observed, climate = xr.align(forecast, truth, climatology, join="exact")
    _require_grid(predicted)
    forecast_anomaly = predicted.astype("float64") - climate.astype("float64")
    truth_anomaly = observed.astype("float64") - climate.astype("float64")
    weights = _weights(predicted).broadcast_like(forecast_anomaly)
    valid = forecast_anomaly.notnull() & truth_anomaly.notnull()
    numerator = (weights * forecast_anomaly * truth_anomaly).where(valid).sum()
    forecast_norm = (weights * forecast_anomaly**2).where(valid).sum()
    truth_norm = (weights * truth_anomaly**2).where(valid).sum()
    denominator = np.sqrt(forecast_norm * truth_norm)
    if float(denominator) <= 0:
        raise ValueError("acc: anomalies have zero norm or no overlap")
    return float(numerator / denominator)


def persistence(initial: xr.Dataset, times: xr.DataArray) -> xr.Dataset:
    """Повторить начальное поле на каждый срок прогноза."""
    if "time" in initial.dims:
        if initial.sizes["time"] != 1:
            raise ValueError(f"persistence: expected one initial time, got {initial.sizes['time']}")
        seed = initial.isel(time=0, drop=True)
    else:
        seed = initial
    stamps = np.asarray(times.values, dtype="datetime64[ns]")
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("persistence: forecast times must be a non-empty axis")
    return seed.expand_dims(time=stamps).assign_coords(time=stamps)


def climatology_for(monthly: xr.Dataset, times: xr.DataArray) -> xr.Dataset:
    """Многолетнее среднее каждого календарного месяца на нужные сроки."""
    if "time" not in monthly.dims or monthly.sizes["time"] < 1:
        raise ValueError("climatology: monthly dataset needs a non-empty time axis")
    stamps = np.asarray(times.values, dtype="datetime64[ns]")
    if stamps.ndim != 1 or stamps.size == 0:
        raise ValueError("climatology: forecast times must be a non-empty axis")
    by_month = monthly.groupby("time.month").mean("time", skipna=True)
    months = xr.DataArray(
        np.asarray([int(str(stamp)[5:7]) for stamp in stamps], dtype=np.int8),
        dims=("time",),
        coords={"time": stamps},
    )
    missing = sorted(set(int(value) for value in months.values) - set(by_month["month"].values))
    if missing:
        raise ValueError(f"climatology: months unavailable: {missing}")
    return by_month.sel(month=months).drop_vars("month")


def score_forecast(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    monthly: xr.Dataset,
    variable: str,
    *,
    init_time: str,
) -> tuple[Metric, ...]:
    """Одна сравнимая таблица model/persistence/climatology по всем срокам."""
    for name, dataset in (("forecast", forecast), ("truth", truth), ("monthly", monthly)):
        if variable not in dataset:
            raise ValueError(f"{name}: variable {variable!r} is absent")
    if "time" not in forecast.dims or forecast.sizes["time"] < 1:
        raise ValueError("forecast: non-empty time axis required")
    init = _moment(init_time)
    forecast_times = forecast["time"]
    verifying = truth.sel(time=forecast_times)
    seed = truth.sel(time=[np.datetime64(init.replace(tzinfo=None), "ns")])
    persisted = persistence(seed, forecast_times)
    climate = climatology_for(monthly, forecast_times)
    results = []
    for index, stamp in enumerate(np.asarray(forecast_times.values, dtype="datetime64[ns]")):
        moment = _as_datetime(stamp)
        lead = (moment - init).total_seconds() / 3600
        if lead <= 0 or not lead.is_integer():
            raise ValueError(f"forecast time {moment.isoformat()}: expected positive whole lead")
        predicted = forecast[variable].isel(time=index)
        observed = verifying[variable].isel(time=index)
        normal = climate[variable].isel(time=index)
        rmse = latitude_weighted_rmse(predicted, observed)
        baseline = latitude_weighted_rmse(persisted[variable].isel(time=index), observed)
        climate_rmse = latitude_weighted_rmse(normal, observed)
        results.append(
            Metric(
                int(lead),
                rmse,
                anomaly_correlation(predicted, observed, normal),
                baseline,
                climate_rmse,
                math.nan if baseline == 0 else 1.0 - rmse / baseline,
            )
        )
    return tuple(results)


def markdown(metrics: tuple[Metric, ...], variable: str) -> str:
    lines = [
        f"| lead | {variable} RMSE | ACC | persistence RMSE | climatology RMSE | skill |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| +{row.lead_hours}h | {row.rmse:.4g} | {row.acc:.4f} | "
        f"{row.persistence_rmse:.4g} | {row.climatology_rmse:.4g} | "
        f"{row.skill_vs_persistence:.1%} |"
        for row in metrics
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("forecast", type=Path)
    cli.add_argument("truth", type=Path)
    cli.add_argument("climatology", type=Path)
    cli.add_argument("--var", required=True)
    cli.add_argument("--init", required=True)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    datasets = [
        xr.open_zarr(path, chunks={}) for path in (args.forecast, args.truth, args.climatology)
    ]
    try:
        result = score_forecast(
            datasets[0], datasets[1], datasets[2], args.var, init_time=args.init
        )
    finally:
        for dataset in datasets:
            dataset.close()
    print(markdown(result, args.var))
    return 0


def _aligned(first: xr.DataArray, second: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    _require_grid(first)
    left, right = xr.align(first, second, join="exact")
    return left, right


def _require_grid(field: xr.DataArray) -> None:
    if "lat" not in field.dims or "lon" not in field.dims:
        raise ValueError(f"field: expected lat/lon dimensions, got {field.dims}")


def _weights(field: xr.DataArray) -> xr.DataArray:
    _require_grid(field)
    lat = field["lat"].astype("float64")
    weights = xr.DataArray(
        np.cos(np.deg2rad(lat.values)), dims=lat.dims, coords=lat.coords, name="latitude_weight"
    )
    if bool((weights < -1e-12).any()):
        raise ValueError("lat: outside -90..90")
    return weights.clip(min=0)


def _moment(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"init_time: invalid ISO 8601 {text!r}") from error
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _as_datetime(value: np.datetime64) -> datetime:
    seconds = int(value.astype("datetime64[s]").astype(np.int64))
    return datetime.fromtimestamp(seconds, tz=UTC)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
