"""Чтение опубликованного прогона: видимость, выбор слоя, ряд в точке.

Проверяется то, что решает хранилище: какой прогон читателю обещан, какой
слой отвечает на запрос и что лежит в узле сетки. Имена и единицы ответа —
не сюда, это `tests/api`.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from contracts import canon
from storage import read
from storage.manifest import read_manifest, write_manifest
from storage.read import (
    OutOfCoverageError,
    TooManyPointsError,
    TooManyStepsError,
    UnsupportedError,
    choose_layer,
    grid_window,
    point_series,
    published_run,
)
from storage.write import write_layer

#: Первый срок фикстуры.
NOON = "2026-08-01T00:00:00Z"


def test_a_run_without_the_published_flag_is_not_readable(published: Path) -> None:
    """Указателя мало. Каталог с `published: false` — это прогон, оборванный
    посреди публикации, и отдавать его читателю нельзя (docs/STORAGE.md §5)."""
    assert published_run(published) is not None
    run = published_run(published)
    assert run is not None
    manifest = {**read_manifest(run / "manifest.json"), "published": False}
    write_manifest(run / "manifest.json", manifest)

    assert published_run(published) is None


def test_an_empty_store_has_no_run(tmp_path: Path) -> None:
    assert published_run(tmp_path) is None


def test_six_hourly_step_goes_to_the_coarse_layer() -> None:
    assert choose_layer(canon.STEP_HOURS, ("2t", "msl")) == "coarse"


def test_hourly_step_goes_to_the_hourly_layer() -> None:
    assert choose_layer(canon.FINE_STEP_HOURS, canon.HOURLY_VARS) == "hourly"


def test_hourly_step_for_a_variable_outside_the_eight_is_refused() -> None:
    """Молчаливое округление до шести часов даёт пользователю не тот ряд, о
    котором он просил, и он об этом не узнает (docs/API_CONTRACT.md §2)."""
    with pytest.raises(UnsupportedError, match="step_hours=1"):
        choose_layer(canon.FINE_STEP_HOURS, ("2t", "sp"))


def test_an_unknown_variable_is_refused_by_name() -> None:
    with pytest.raises(UnsupportedError, match="temperature"):
        choose_layer(canon.STEP_HOURS, ("temperature",))


def test_only_one_and_six_hours_are_steps() -> None:
    with pytest.raises(UnsupportedError, match="step_hours"):
        choose_layer(3, ("2t",))


def test_a_point_is_read_from_the_series_layout(published: Path) -> None:
    """Ряд в точке идёт в `points`, а не в `coarse`: карта на срок отдаёт
    4.15 МБ ради 4 байт, и на сорока сроках это 300 мс (docs/STORAGE.md §3)."""
    run = published_run(published)
    assert run is not None

    assert read.point_layer(run, "coarse", ("2t", "msl")).name == "points"


def test_a_variable_outside_the_eight_falls_back_to_the_maps(published: Path) -> None:
    """В `points` восемь переменных. `sp` шестичасовой есть только в `coarse`,
    и ответить «такой переменной нет» вместо медленного ответа было бы враньём."""
    run = published_run(published)
    assert run is not None

    assert read.point_layer(run, "coarse", ("2t", "sp")).name == "coarse"


def test_a_run_without_the_point_layer_is_still_readable(tmp_path: Path) -> None:
    """Прогон, разложенный до появления `points`, читается из карт."""
    (tmp_path / "coarse").mkdir()

    assert read.point_layer(tmp_path, "coarse", ("2t",)).name == "coarse"


def test_the_hourly_layer_has_no_second_copy(published: Path) -> None:
    """Часовой слой лежит рядами и в одном экземпляре: точку из него читают,
    карту — нет (docs/API_CONTRACT.md §2, округление `time` до шести часов)."""
    run = published_run(published)
    assert run is not None

    assert read.point_layer(run, "hourly", ("2t",)).name == "hourly"


def test_point_lands_on_a_grid_node(published: Path) -> None:
    """Пользователь просит свой двор, а получает узел сетки 0.25° — и обязан
    видеть, какой именно (docs/API_CONTRACT.md §2)."""
    run = published_run(published)
    assert run is not None
    point = point_series(run / "coarse", ("2t",), 55.75, 37.62)

    assert (point.lat, point.lon) == (54.0, 45.0)
    assert point.times[0] == "2026-08-01T00:00:00Z"
    # На диске float32: 288.15 туда не влезает, и сравнивать с ним точно —
    # проверять не хранилище, а двоичную дробь.
    assert point.values["2t"] == pytest.approx((288.15, 288.15, 288.15), abs=1e-4)
    assert point.init_time == "2026-08-01T00:00:00Z"


def test_a_window_outside_the_run_is_not_an_empty_series(published: Path) -> None:
    """Пустой ряд читатель принял бы за «погоды не будет». Дата вне покрытия —
    отдельный ответ (docs/API_CONTRACT.md §4, `404`)."""
    run = published_run(published)
    assert run is not None
    with pytest.raises(OutOfCoverageError):
        point_series(run / "coarse", ("2t",), 55.0, 37.0, start="2026-09-01T00:00:00Z")


def test_too_many_steps_is_refused_before_the_read(published: Path) -> None:
    run = published_run(published)
    assert run is not None
    with pytest.raises(TooManyStepsError):
        point_series(run / "coarse", ("2t",), 55.0, 37.0, max_steps=2)


def test_too_many_points_is_refused_before_the_read(published: Path) -> None:
    """Потолок точек проверяется до `load()`: смысл потолка в том, чтобы не
    поднимать с диска то, что всё равно не отдашь."""
    run = published_run(published)
    assert run is not None
    with pytest.raises(TooManyPointsError) as refused:
        grid_window(run / "coarse", ("2t",), (-90.0, -180.0, 90.0, 135.0), NOON, max_points=10)

    assert refused.value.requested == 6 * 8
    stride = refused.value.suggested_stride
    fits = grid_window(
        run / "coarse", ("2t",), (-90.0, -180.0, 90.0, 135.0), NOON, stride=stride, max_points=10
    )
    assert fits.shape[0] * fits.shape[1] <= 10


def test_the_suggested_stride_survives_a_window_one_row_tall(published: Path) -> None:
    """Формула `sqrt(точек / потолок)` здесь врёт: прореженный размер — это
    округление вверх, и на окне 1 × 8 с потолком 3 она даёт 2, при котором
    точек остаётся 4. Подсказка, которая не работает, хуже отсутствующей."""
    run = published_run(published)
    assert run is not None
    with pytest.raises(TooManyPointsError) as refused:
        grid_window(run / "coarse", ("2t",), (50.0, -180.0, 60.0, 135.0), NOON, max_points=3)

    assert refused.value.suggested_stride == 3
    fits = grid_window(
        run / "coarse", ("2t",), (50.0, -180.0, 60.0, 135.0), NOON, stride=3, max_points=3
    )
    assert fits.shape == (1, 3)


def test_the_window_is_cut_by_time_and_not_by_string_order(published: Path) -> None:
    run = published_run(published)
    assert run is not None
    point = point_series(
        run / "coarse",
        ("2t",),
        55.0,
        37.0,
        start="2026-08-01T06:00:00Z",
        end="2026-08-01T06:00:00Z",
    )
    assert point.times == ("2026-08-01T06:00:00Z",)


def test_gaps_come_back_as_nulls_not_as_nan(tmp_path: Path) -> None:
    """`NaN` — невалидный JSON, а `-9999` читатель примет за температуру
    (docs/API_CONTRACT.md §1)."""
    layer = canon.Layer("coarse", ("2t",), (), canon.STEP_HOURS, 2)
    field = np.array([[[288.0]], [[np.nan]]], dtype=np.float32)
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), field)},
        coords={
            "time": np.array(["2026-08-01T00", "2026-08-01T06"], dtype="datetime64[ns]"),
            "lat": [55.75],
            "lon": [37.5],
        },
        attrs={"init_time": "2026-08-01T00:00:00Z"},
    )
    path = write_layer(ds, tmp_path / "coarse", layer)

    assert point_series(path, ("2t",), 55.75, 37.5).values["2t"] == (288.0, None)


def test_a_field_on_pressure_levels_is_not_a_series(tmp_path: Path) -> None:
    """Тринадцать уровней — это тринадцать рядов, а уровня в запросе точки
    контракт не предусматривает."""
    layer = canon.Layer("coarse", (), ("t",), canon.STEP_HOURS, 1)
    ds = xr.Dataset(
        {"t": (("time", "level", "lat", "lon"), np.full((1, 2, 1, 1), 250.0, dtype=np.float32))},
        coords={
            "time": np.array(["2026-08-01T00"], dtype="datetime64[ns]"),
            "level": np.array([500, 850], dtype="int32"),
            "lat": [55.75],
            "lon": [37.5],
        },
    )
    path = write_layer(ds, tmp_path / "coarse", layer)

    with pytest.raises(UnsupportedError, match="уровн"):
        point_series(path, ("t",), 55.75, 37.5)


def test_coverage_is_keyed_by_step_and_skips_the_copy(published: Path) -> None:
    """Покрытие перечисляет слои, которые обслуживают запросы, и знает их по
    шагу. `points` — вторая копия шестичасовых полей в другой раскладке
    (docs/STORAGE.md §3), и отдельным покрытием она не является."""
    run = published_run(published)
    assert run is not None
    spans = read.coverage(run)

    assert [span.step_hours for span in spans] == [canon.FINE_STEP_HOURS, canon.STEP_HOURS]
    hourly = spans[0]
    assert (hourly.first, hourly.last, hourly.steps) == (
        "2026-08-01T00:00:00Z",
        "2026-08-01T03:00:00Z",
        4,
    )
    assert hourly.init_time == "2026-08-01T00:00:00Z"
    assert set(hourly.names) == set(canon.HOURLY_VARS)


def test_coverage_leaves_out_the_fields_on_pressure_levels(tmp_path: Path) -> None:
    """На настоящей сетке в слое 65 полей на уровнях из 91. Ни точка, ни сетка
    `level` не принимают и отвечают на них `400`, а имя в покрытии — обещание,
    что его можно подставить в `vars`: `t` дал бы вечно нерабочую кнопку."""
    layer = canon.Layer("coarse", ("2t",), ("t",), canon.STEP_HOURS, 1)
    ds = xr.Dataset(
        {
            "2t": (("time", "lat", "lon"), np.full((1, 1, 1), 288.0, dtype=np.float32)),
            "t": (("time", "level", "lat", "lon"), np.full((1, 2, 1, 1), 250.0, dtype=np.float32)),
        },
        coords={
            "time": np.array(["2026-08-01T00"], dtype="datetime64[ns]"),
            "level": np.array([500, 850], dtype="int32"),
            "lat": [55.75],
            "lon": [37.5],
        },
        attrs={"init_time": "2026-08-01T00:00:00Z"},
    )
    write_layer(ds, tmp_path / "coarse", layer)

    assert read.coverage(tmp_path)[0].names == ("2t",)


def test_coverage_of_a_run_without_layers_is_empty(tmp_path: Path) -> None:
    """Не исключение: «прогона нет» и «прогон пуст» — это один ответ читателю,
    и разбирать его на два в хранилище незачем."""
    assert read.coverage(tmp_path) == ()


def test_disk_usage_is_the_filesystem(published: Path) -> None:
    free, total = read.disk_usage(published)

    assert 0 < free <= total


def test_layer_span_is_the_first_and_the_last_step(published: Path) -> None:
    run = published_run(published)
    assert run is not None
    assert read.layer_span(run / "hourly") == ("2026-08-01T00:00:00Z", "2026-08-01T03:00:00Z")
