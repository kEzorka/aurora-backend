"""Запись слоя в Zarr v3: чанки, шарды, кодек, отказ писать поверх.

Сетка в большинстве тестов маленькая: проверяется форма записи, а не её объём.
Настоящая сетка нужна ровно там, где проверяется время чтения карты.
"""

import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from contracts import canon
from storage.layout import LAYOUT_A, LAYOUT_B, Chunking, encoding_for
from storage.write import write_layer

#: Слой на одну переменную: форму записи он показывает так же, как настоящий,
#: но стоит килобайты вместо гигабайтов.
TINY_LAYER = canon.Layer("tiny", ("2t", "10u"), (), canon.STEP_HOURS, 2)


def _tiny(times: int = 2, extra: tuple[str, ...] = ()) -> xr.Dataset:
    names = (*TINY_LAYER.surface_vars, *extra)
    ny, nx = 6, 8
    data = {
        name: (("time", "lat", "lon"), np.full((times, ny, nx), 288.0, dtype=np.float32))
        for name in names
    }
    ds = xr.Dataset(
        data,
        coords={
            "time": np.array(
                [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(times)],
                dtype="datetime64[ns]",
            ),
            "lat": np.linspace(90.0, -90.0, ny),
            "lon": np.linspace(-180.0, 179.75, nx),
        },
        attrs={"init_time": "2026-08-01T00:00:00Z", "kind": "forecast"},
    )
    for name in names:
        ds[name].attrs["units"] = "K"
        ds[name].attrs["_FillValue"] = np.float32(np.nan)
    return ds


def test_written_layer_reads_back_with_values_and_attributes(tmp_path: Path) -> None:
    path = write_layer(_tiny(), tmp_path / "coarse", TINY_LAYER)
    back = xr.open_zarr(path)
    assert set(back.data_vars) == set(TINY_LAYER.surface_vars)
    assert back["2t"].attrs["units"] == "K"
    assert back.attrs["init_time"] == "2026-08-01T00:00:00Z"
    np.testing.assert_allclose(back["2t"].values, 288.0)


def test_fields_outside_the_layer_are_not_written(tmp_path: Path) -> None:
    """`tp` у адаптера есть, в слое его нет — в файл он попасть не должен."""
    path = write_layer(_tiny(extra=("tp",)), tmp_path / "coarse", TINY_LAYER)
    assert "tp" not in xr.open_zarr(path).data_vars


def test_writing_over_an_existing_layer_is_refused(tmp_path: Path) -> None:
    """Срез, наполовину перезаписанный, читатель получит как целый
    (docs/STORAGE.md §5): публикация идёт в новый ключ, а не поверх."""
    target = tmp_path / "coarse"
    write_layer(_tiny(), target, TINY_LAYER)
    with pytest.raises(FileExistsError, match="coarse"):
        write_layer(_tiny(), target, TINY_LAYER)


def test_chunks_and_shards_land_on_disk_as_declared(tmp_path: Path) -> None:
    layout = Chunking(name="tiny", chunk=(1, 1, 3, 4), shard=(2, 1, 6, 8))
    path = write_layer(_tiny(), tmp_path / "coarse", TINY_LAYER, layout=layout)
    array = zarr.open_array(str(path / "2t"), mode="r")
    assert array.chunks == (1, 3, 4)
    assert array.shards == (2, 6, 8)


def test_codec_is_zstd_3_with_shuffle(tmp_path: Path) -> None:
    """docs/STORAGE.md §4: `zstd` уровня 3 с shuffle — не дефолт библиотеки."""
    path = write_layer(_tiny(), tmp_path / "coarse", TINY_LAYER)
    array = zarr.open_array(str(path / "2t"), mode="r")
    codec = array.compressors[0]
    assert isinstance(codec, BloscCodec)
    assert (str(codec.cname), codec.clevel, str(codec.shuffle)) == ("zstd", 3, "shuffle")


def test_chunk_is_clipped_to_the_array(tmp_path: Path) -> None:
    """Чанк раскладки A — вся карта 721 × 1440. На меньшем массиве он равен
    массиву: Zarr чанк больше массива не принимает."""
    encoding = encoding_for(_tiny(), LAYOUT_A)
    assert encoding["2t"]["chunks"] == (1, 6, 8)


def test_shard_stays_a_multiple_of_the_chunk_after_clipping(tmp_path: Path) -> None:
    """Обрезка по массиву не должна ломать делимость: шард, не кратный чанку,
    Zarr отвергает — и падает это на записи, посреди прогона."""
    for layout in (LAYOUT_A, LAYOUT_B):
        encoding = encoding_for(_tiny(times=3), layout)["2t"]
        shards = cast(tuple[int, ...], encoding["shards"])
        chunks = cast(tuple[int, ...], encoding["chunks"])
        for shard, chunk in zip(shards, chunks, strict=True):
            assert shard % chunk == 0, layout.name


def test_a_lazy_dataset_read_from_zarr_can_be_written_again(tmp_path: Path) -> None:
    """Набор, открытый из Zarr, приходит с чужими чанками. Шард, накрывающий
    два dask-чанка, xarray писать отказывается — так свёртка прошлого прогона
    падала бы на каждой публикации."""
    source = write_layer(_tiny(times=3), tmp_path / "source", TINY_LAYER)
    with xr.open_zarr(source) as lazy:
        copy = write_layer(lazy, tmp_path / "copy", TINY_LAYER)
    np.testing.assert_allclose(xr.open_zarr(copy)["2t"].values, 288.0)


def test_level_axis_is_chunked_one_level_at_a_time() -> None:
    """У поля на уровнях давления ось `level` есть, и чанк по ней всегда 1:
    запрос спрашивает уровень, а не все тринадцать."""
    ds = xr.Dataset(
        {"t": (("time", "level", "lat", "lon"), np.zeros((2, 13, 6, 8), dtype=np.float32))},
        coords={"level": np.array(canon.PRESSURE_LEVELS, dtype="int32")},
    )
    assert encoding_for(ds, LAYOUT_A)["t"]["chunks"] == (1, 1, 6, 8)


@pytest.mark.slow
def test_a_map_reads_in_under_300_ms(tmp_path: Path) -> None:
    """Приёмка 2.1: карта на момент с прогретого диска — меньше 300 мс."""
    ny, nx = canon.GRID_SHAPE
    steps = 8
    field = np.random.default_rng(0).normal(288.0, 10.0, (steps, ny, nx)).astype(np.float32)
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), field)},
        coords={
            "time": np.array(
                [np.datetime64("2026-08-01T00") + np.timedelta64(6 * i, "h") for i in range(steps)],
                dtype="datetime64[ns]",
            ),
            "lat": canon.LAT,
            "lon": canon.LON,
        },
    )
    ds["2t"].attrs["units"] = "K"
    layer = canon.Layer("one", ("2t",), (), canon.STEP_HOURS, steps)
    path = write_layer(ds, tmp_path / "coarse", layer, layout=LAYOUT_A)

    opened = xr.open_zarr(path, chunks={})
    opened["2t"].isel(time=0).load()  # прогрев кэша страниц
    started = time.perf_counter()
    got = opened["2t"].isel(time=5).load()
    elapsed = time.perf_counter() - started
    assert got.shape == (ny, nx)
    assert elapsed < 0.3, f"карта читалась {elapsed * 1000:.0f} мс"
