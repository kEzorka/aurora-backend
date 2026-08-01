"""Раскладка B: ряд в точке за годы и раздувание чтения.

Сетка здесь меньше настоящей намеренно. Чтение ряда в точке стоит один чанк
независимо от размера сетки — чанк и есть весь ряд, — поэтому 128 × 128
показывает ту же цифру, что 721 × 1440, но не пишет на диск 4 ГБ.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from contracts import canon
from storage.layout import LAYOUT_A, LAYOUT_B, STEPS_PER_YEAR, read_amplification
from storage.write import write_layer

YEAR: tuple[int, int, int, int] = (STEPS_PER_YEAR, 1, 1, 1)
MAP: tuple[int, int, int, int] = (1, 1, canon.GRID_SHAPE[0], canon.GRID_SHAPE[1])


def test_point_series_costs_one_chunk_in_layout_b() -> None:
    """Год в точке лежит в одном чанке; лишнего — квадрат 4 × 4 вокруг точки."""
    assert read_amplification(LAYOUT_B, YEAR) == 16.0


def test_point_series_in_layout_a_reads_a_million_times_too_much() -> None:
    """Та же выборка по картам: 1460 карт целиком ради 1460 чисел
    (docs/STORAGE.md §3, «легко получить ×10⁶»)."""
    assert read_amplification(LAYOUT_A, YEAR) == pytest.approx(1.04e6, rel=0.01)


def test_a_map_costs_exactly_one_chunk_in_layout_a() -> None:
    assert read_amplification(LAYOUT_A, MAP) == 1.0


def test_a_map_is_hopeless_in_layout_b() -> None:
    """Обратная сторона: карту по раскладке рядов читать нельзя — она поднимет
    год целиком. Поэтому раскладки две, а не одна «получше»."""
    # Чуть больше 1460: 721 широта на чанки по 4 нацело не делится.
    assert read_amplification(LAYOUT_B, MAP) == pytest.approx(STEPS_PER_YEAR, rel=0.01)


@pytest.mark.slow
def test_point_series_reads_in_under_300_ms(tmp_path: Path) -> None:
    """Приёмка 2.2: ряд в точке за год — меньше 300 мс."""
    ny = nx = 128
    field = (
        np.random.default_rng(0).normal(288.0, 10.0, (STEPS_PER_YEAR, ny, nx)).astype(np.float32)
    )
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), field)},
        coords={
            "time": np.arange(
                np.datetime64("2025-01-01T00"),
                np.datetime64("2025-01-01T00") + np.timedelta64(6 * STEPS_PER_YEAR, "h"),
                np.timedelta64(6, "h"),
            ).astype("datetime64[ns]"),
            "lat": np.linspace(90.0, -90.0, ny),
            "lon": np.linspace(-180.0, 179.75, nx),
        },
    )
    ds["2t"].attrs["units"] = "K"
    layer = canon.Layer("one", ("2t",), (), canon.STEP_HOURS, STEPS_PER_YEAR)
    path = write_layer(ds, tmp_path / "history", layer, layout=LAYOUT_B)

    opened = xr.open_zarr(path, chunks={})
    opened["2t"].isel(lat=0, lon=0).load()  # прогрев кэша страниц
    started = time.perf_counter()
    got = opened["2t"].isel(lat=77, lon=13).load()
    elapsed = time.perf_counter() - started
    assert got.shape == (STEPS_PER_YEAR,)
    np.testing.assert_allclose(got.values, field[:, 77, 13])
    assert elapsed < 0.3, f"ряд читался {elapsed * 1000:.0f} мс"
