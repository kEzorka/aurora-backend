"""Фикстуры — тоже контракт.

Тесты адаптеров будут утверждать «GFS отдаёт долготу 0..360» и «у GFS осадки
накоплены за интервал, а у ECMWF от начала прогона». Если фикстуру однажды
перекачать другой командой или другим шагом, эти утверждения станут проверять
не то, что написано, а молча продолжат проходить. Здесь проверяется, что файлы
в `tests/fixtures/grib/` действительно обладают теми свойствами, ради которых
они выбраны — до всякого адаптера и независимо от него.
"""

from pathlib import Path

import pytest

pytest.importorskip("cfgrib", reason="cfgrib тянет бинарный eccodes; см. docs/SETUP.md §4")

import xarray as xr

GRIB = Path(__file__).resolve().parent / "grib"


def _open(name: str) -> xr.Dataset:
    # indexpath="" — не писать .idx рядом с фикстурой: файл в репозитории,
    # и тест не имеет права оставлять после себя изменения в дереве.
    return xr.open_dataset(GRIB / name, engine="cfgrib", backend_kwargs={"indexpath": ""})


def _step_hours(ds: xr.Dataset) -> int:
    return int(ds["step"].values / 3_600_000_000_000)


def test_all_fixtures_together_stay_a_few_megabytes() -> None:
    total = sum(path.stat().st_size for path in GRIB.glob("*.grib2"))
    assert total < 20 * 1024**2, f"{total / 1024**2:.1f} МБ"


def test_gfs_longitude_runs_zero_to_360() -> None:
    """Ловушка 2 из docs/DOMAIN.md §6. Без такой фикстуры её нечем проверить."""
    ds = _open("gfs_t2m.grib2")
    assert float(ds["longitude"][0]) == 0.0
    assert float(ds["longitude"][-1]) == 359.75


def test_ecmwf_longitude_runs_minus180_to_180() -> None:
    ds = _open("ecmwf_2t_6h.grib2")
    assert float(ds["longitude"][0]) == -180.0
    assert float(ds["longitude"][-1]) == 179.75


def test_both_sources_share_the_aurora_grid() -> None:
    """Иначе разница между ними была бы не в осях, а в разрешении."""
    for name in ("gfs_t2m.grib2", "ecmwf_2t_6h.grib2"):
        ds = _open(name)
        assert (ds.sizes["latitude"], ds.sizes["longitude"]) == (721, 1440), name
        assert float(ds["latitude"][0]) == 90.0, name
        assert float(ds["latitude"][-1]) == -90.0, name


def test_temperature_fixtures_are_in_kelvin_not_celsius() -> None:
    for name, var in (("gfs_t2m.grib2", "t2m"), ("ecmwf_2t_6h.grib2", "t2m")):
        ds = _open(name)
        assert ds[var].attrs["units"] == "K", name
        assert float(ds[var].min()) > 150.0, name


def test_ecmwf_precipitation_accumulates_from_the_start_of_the_run() -> None:
    """0-6 и 0-12: шаг получается разностью, и она неотрицательна всюду."""
    six, twelve = _open("ecmwf_tp_6h.grib2"), _open("ecmwf_tp_12h.grib2")
    assert _step_hours(six) == 6
    assert _step_hours(twelve) == 12
    assert float((twelve["tp"] - six["tp"]).min()) >= 0.0


def test_gfs_precipitation_accumulates_over_the_interval_only() -> None:
    """6-12, а не 0-12: разность здесь дала бы отрицательные осадки.

    Это и есть смысл пары фикстур. Правило «осадки накоплены, вычитай соседние
    шаги» верно для ECMWF и портит поле GFS; из одного источника такое не видно.
    """
    six, twelve = _open("gfs_apcp_f006.grib2"), _open("gfs_apcp_f012.grib2")
    assert _step_hours(six) == 6
    assert _step_hours(twelve) == 12
    assert float((twelve["tp"] - six["tp"]).min()) < 0.0


def test_precipitation_units_differ_between_the_two_sources() -> None:
    """Метры против kg m-2: множитель 1000, ловушка 3 из docs/DOMAIN.md §6."""
    assert _open("ecmwf_tp_6h.grib2")["tp"].attrs["units"] == "m"
    assert _open("gfs_apcp_f006.grib2")["tp"].attrs["units"] == "kg m**-2"


def test_a_truncated_fixture_does_not_open_as_half_a_field(tmp_path: Path) -> None:
    """Битый файл нужен тестам отказа. Он делается из настоящего, а не хранится:
    обрезка воспроизводима, а лишний мегабайт в репозитории — нет."""
    whole = (GRIB / "gfs_t2m.grib2").read_bytes()
    broken = tmp_path / "truncated.grib2"
    broken.write_bytes(whole[: len(whole) // 2])
    assert broken.read_bytes().startswith(b"GRIB")

    with pytest.raises(Exception):  # noqa: B017
        # Какое именно исключение — дело cfgrib и eccodes; проверяется то, что
        # половина файла не превращается в половину поля.
        xr.open_dataset(broken, engine="cfgrib", backend_kwargs={"indexpath": ""}).load()
