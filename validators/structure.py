"""Уровень «структура»: форма данных, безотносительно их смысла.

docs/DATA_CONTRACT.md §4, уровень «структура»: размерности (721, 1440),
монотонность осей, отсутствие дубликатов и пропусков по времени, наличие
всех полей набора (90 на срок у прогноза Aurora 1.5, docs/DOMAIN.md §5).

Здесь ловится вторая ловушка docs/DOMAIN.md §6: сетка 0..360 вместо
-180..180. Она не даёт ни исключения, ни NaN — просто Европа оказывается
на месте Тихого океана, и это видно только глазами на карте. Поэтому
диапазон долготы проверяется явно, а не «на глаз».
"""

import numpy as np
import xarray as xr

from contracts import canon
from validators.result import Check, fail, ok


def check_structure(
    ds: xr.Dataset,
    *,
    expect_static: bool = False,
    layer: canon.Layer | None = None,
) -> list[Check]:
    """Все структурные проверки разом. Возвращает и пройденные, и упавшие.

    `layer` — слой хранилища (`canon.LAYERS`): он задаёт и набор полей, и шаг.
    Наборов после Aurora 1.5 стало три, и по умолчанию проверяется `coarse`:
    его записывают чаще и ошибаются в нём дороже. Часовой слой без явного
    `layer` был бы отвергнут как неполный — у него 8 полей вместо 90.
    """
    checked = canon.LAYERS["coarse"] if layer is None else layer
    return [
        _grid_shape(ds),
        _lat_descending(ds),
        _lon_range(ds),
        _lon_ascending(ds),
        _levels_match(ds),
        _fields_present(ds, expect_static=expect_static, layer=checked),
        _time_unique(ds),
        _time_regular(ds, step_hours=checked.step_hours),
    ]


def _grid_shape(ds: xr.Dataset) -> Check:
    got = (int(ds.sizes.get("lat", 0)), int(ds.sizes.get("lon", 0)))
    if got != canon.GRID_SHAPE:
        return fail("grid_shape", "structure", "(lat, lon)", got, canon.GRID_SHAPE)
    return ok("grid_shape", "structure", f"{got[0]}x{got[1]} {canon.GRID_NAME}")


def _lat_descending(ds: xr.Dataset) -> Check:
    """+90 -> -90. Перевёрнутая широта переворачивает карту, не ломая её."""
    if "lat" not in ds.coords:
        return fail("lat_descending", "structure", "lat", "missing", "coordinate present")
    lat = np.asarray(ds["lat"].values, dtype=float)
    if lat.size > 1 and not np.all(np.diff(lat) < 0):
        return fail(
            "lat_descending",
            "structure",
            "lat",
            (float(lat[0]), float(lat[-1])),
            "strictly decreasing from +90 to -90",
        )
    return ok("lat_descending", "structure")


def _lon_range(ds: xr.Dataset) -> Check:
    if "lon" not in ds.coords:
        return fail("lon_range", "structure", "lon", "missing", "coordinate present")
    lon = np.asarray(ds["lon"].values, dtype=float)
    got = (float(lon[0]), float(lon[-1]))
    expected = (float(canon.LON[0]), float(canon.LON[-1]))
    if got != expected:
        return fail("lon_range", "structure", "lon", got, expected)
    return ok("lon_range", "structure")


def _lon_ascending(ds: xr.Dataset) -> Check:
    if "lon" not in ds.coords:
        return fail("lon_ascending", "structure", "lon", "missing", "coordinate present")
    lon = np.asarray(ds["lon"].values, dtype=float)
    if lon.size > 1 and not np.all(np.diff(lon) > 0):
        return fail(
            "lon_ascending",
            "structure",
            "lon",
            (float(lon[0]), float(lon[-1])),
            "strictly increasing",
        )
    return ok("lon_ascending", "structure")


def _levels_match(ds: xr.Dataset) -> Check:
    """Порядок уровней — часть контракта: Aurora получает срезы по индексу."""
    has_atmos = any(str(v) in canon.ATMOS_VARS for v in ds.data_vars)
    if "level" not in ds.coords:
        if not has_atmos:
            return ok("levels_match", "structure", "surface-only dataset")
        return fail("levels_match", "structure", "level", "missing", canon.PRESSURE_LEVELS)
    got = tuple(int(v) for v in np.asarray(ds["level"].values))
    if got != canon.PRESSURE_LEVELS:
        return fail("levels_match", "structure", "level", got, canon.PRESSURE_LEVELS)
    return ok("levels_match", "structure")


def _fields_present(ds: xr.Dataset, *, expect_static: bool, layer: canon.Layer) -> Check:
    required = set(layer.surface_vars) | set(layer.atmos_vars)
    if expect_static:
        required |= set(canon.STATIC_VARS)
    got = {str(v) for v in ds.data_vars}
    missing = required - got
    if missing:
        return fail("fields_present", "structure", "data_vars", sorted(got), sorted(required))
    return ok("fields_present", "structure", f"{len(got)} variables")


def _time_unique(ds: xr.Dataset) -> Check:
    if "time" not in ds.coords:
        return fail("time_unique", "structure", "time", "missing", "coordinate present")
    time = np.asarray(ds["time"].values)
    duplicates = time.size - np.unique(time).size
    if duplicates:
        return fail("time_unique", "structure", "time", f"{duplicates} duplicates", "0 duplicates")
    return ok("time_unique", "structure", f"{time.size} steps")


def _time_regular(ds: xr.Dataset, *, step_hours: int) -> Check:
    """Пропуск по времени — это молча укороченный прогноз, а не ошибка чтения.

    Шагов после Aurora 1.5 два (6 ч и 1 ч), но в одном наборе — ровно один:
    смесь шагов означает, что часовой слой и шестичасовой склеились при записи,
    и любая сумма по времени после этого врёт. Проверяется шаг того слоя, в
    который пишут: набор с шагом 1 ч, объявленный шестичасовым, — это не
    «другой слой», а перепутанные сроки.
    """
    if "time" not in ds.coords:
        return fail("time_regular", "structure", "time", "missing", "coordinate present")
    time = np.asarray(ds["time"].values)
    if time.size < 2:
        return ok("time_regular", "structure", "single step")
    allowed = (step_hours,)
    # Сравнение точное, в наносекундах. astype("timedelta64[h]") округляет вниз,
    # и шаг 6 ч 1 мин читается как ровно 6: сдвиг времени — первая ловушка
    # docs/DOMAIN.md §6, ей нельзя давать пройти через округление.
    deltas = np.unique(np.diff(time).astype("timedelta64[ns]").astype("int64"))
    expected = [np.timedelta64(h, "h").astype("timedelta64[ns]").astype("int64") for h in allowed]
    if len(deltas) != 1 or int(deltas[0]) not in expected:
        got = [f"{int(d) / 3.6e12:g}h" for d in deltas]
        return fail("time_regular", "structure", "time step", got, [f"{h}h" for h in allowed])
    return ok("time_regular", "structure", f"{int(deltas[0]) / 3.6e12:g}h step")
