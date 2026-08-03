"""Каноническое входное состояние до преобразования в Aurora Batch (4.1).

Модуль не импортирует ``aurora`` или ``torch``: service-контур проверяет и
публикует вход независимо от доступности GPU. Граница к конкретному классу
``Batch`` появится в inference-окружении вместе с AuroraV1p5 (4.8), а здесь
фиксируется то, что уже является нашим контрактом: два срока, порядок полей,
уровней и осей плюс официальный pickle из 36 статических полей.
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr

from contracts import canon


class State(NamedTuple):
    surface: Mapping[str, xr.DataArray]
    atmosphere: Mapping[str, xr.DataArray]
    static: Mapping[str, object]
    init_time: str


def assemble_state(
    analysis: xr.Dataset,
    static: Mapping[str, object],
    *,
    init_time: str,
    expected_lat: object = canon.LAT,
    expected_lon: object = canon.LON,
) -> State:
    """Проверить и упорядочить вход без копирования глобальных массивов."""
    required = (*canon.SURFACE_INGESTED_VARS, *canon.ATMOS_VARS)
    missing = [name for name in required if name not in analysis]
    if missing:
        raise ValueError(f"analysis: fields missing: {', '.join(missing)}")
    if len(static) != 36:
        raise ValueError(f"static: got {len(static)} fields, expected official 36-field pickle")
    _coordinates(analysis, init_time, expected_lat, expected_lon)

    surface = {
        name: _dimensions(analysis[name], ("time", "lat", "lon"), name)
        for name in canon.SURFACE_INGESTED_VARS
    }
    atmosphere = {
        name: _dimensions(analysis[name], ("time", "level", "lat", "lon"), name)
        for name in canon.ATMOS_VARS
    }
    return State(surface, atmosphere, dict(static), init_time)


def load_static(path: str | Path) -> Mapping[str, object]:
    """Прочитать только доверенный официальный pickle Aurora.

    Pickle исполняет код при загрузке, поэтому пользовательский upload сюда
    передавать нельзя. Путь — операторская конфигурация inference-узла.
    """
    with Path(path).open("rb") as stream:
        loaded = pickle.load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"static: got {type(loaded).__name__}, expected mapping")
    return {str(name): value for name, value in loaded.items()}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("analysis", type=Path)
    cli.add_argument("static", type=Path)
    cli.add_argument("--init", required=True)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    with xr.open_zarr(args.analysis, chunks={}) as dataset:
        state = assemble_state(dataset, load_static(args.static), init_time=args.init)
    print(
        f"ok: init={state.init_time}, surface={len(state.surface)}, "
        f"atmosphere={len(state.atmosphere)}x{len(canon.PRESSURE_LEVELS)}, "
        f"static={len(state.static)}"
    )
    return 0


def _coordinates(
    dataset: xr.Dataset,
    init_time: str,
    expected_lat: object,
    expected_lon: object,
) -> None:
    for name in ("time", "level", "lat", "lon"):
        if name not in dataset.coords:
            raise ValueError(f"analysis: coordinate {name!r} is missing")
    times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
    if times.shape != (2,):
        raise ValueError(f"analysis: got {times.size} times, expected exactly two")
    wanted = np.datetime64(_iso(init_time), "ns")
    if times[1] != wanted or times[1] - times[0] != np.timedelta64(canon.STEP_HOURS, "h"):
        raise ValueError(f"analysis: times {times.tolist()}, expected init-6h and init {wanted!s}")
    _exact("level", dataset["level"].values, canon.PRESSURE_LEVELS)
    _exact("lat", dataset["lat"].values, expected_lat)
    _exact("lon", dataset["lon"].values, expected_lon)


def _dimensions(field: xr.DataArray, dims: tuple[str, ...], name: str) -> xr.DataArray:
    if field.dims != dims:
        raise ValueError(f"{name}: got dimensions {field.dims}, expected {dims}")
    return field


def _exact(name: str, got: object, expected: object) -> None:
    left, right = np.asarray(got), np.asarray(expected)
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"analysis: coordinate {name!r} does not match the canonical order")


def _iso(text: str) -> str:
    try:
        value = np.datetime64(text.replace("Z", ""), "ns")
    except ValueError as error:
        raise ValueError(f"init_time: invalid ISO 8601 {text!r}") from error
    return str(value)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
