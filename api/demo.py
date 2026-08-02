"""Локальный демонстрационный прогноз без сети, ключей и GPU.

Это не второй API и не mock-ветка в приложении. Команда один раз пишет
маленький синтетический прогон в обычном формате Zarr, публикует его обычным
указателем и запускает тот же :func:`api.app.create_app`. Поэтому интерфейс
демонстрирует настоящий путь чтения, единиц, округления и GIF-карт.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import uvicorn
import xarray as xr

from api.app import create_app
from contracts import canon
from storage.manifest import Input, Model, build_manifest
from storage.publish import current_run, publish_run, stage_path
from storage.write import write_layer

DEFAULT_DEMO_ROOT: Final = Path("artifacts/demo-store")
DEMO_COARSE: Final = canon.Layer(
    "coarse", canon.HOURLY_VARS, (), canon.STEP_HOURS, canon.FORECAST_STEPS
)
DEMO_HOURLY: Final = canon.Layer("hourly", canon.HOURLY_VARS, (), 1, 24)


def build_demo_store(root: str | Path = DEFAULT_DEMO_ROOT) -> Path:
    """Создать небольшой опубликованный прогноз или использовать готовый."""
    target = Path(root).resolve()
    if current_run(target) is not None:
        return target

    started = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    started -= timedelta(hours=started.hour % canon.STEP_HOURS)
    run_id = started.strftime("%Y-%m-%dT%HZ")
    staged = stage_path(target, run_id)
    write_layer(_forecast(DEMO_COARSE, started), staged / "coarse", DEMO_COARSE)
    write_layer(_forecast(DEMO_HOURLY, started), staged / "hourly", DEMO_HOURLY)
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "validation.json").write_text(
        '{"ok": true, "checks": [{"name": "demo", "ok": true}]}', encoding="utf-8"
    )
    manifest = build_manifest(
        f"forecast/{run_id}",
        inputs=(
            Input(
                "ifs-analysis",
                started.strftime("%Y-%m-%dT%H:%MZ"),
                "sha256:" + "0" * 64,
                "synthetic/demo",
                canon.HOURLY_VARS,
            ),
        ),
        model=Model("aurora-demo", "synthetic-fields", "local"),
        steps=DEMO_COARSE.steps,
        timings_sec={"ingest": 0, "normalize": 0, "inference": 0, "write": 0},
    )
    publish_run(target, run_id, manifest=manifest)
    return target


def _forecast(layer: canon.Layer, started: datetime) -> xr.Dataset:
    """Правдоподобные, но явно синтетические поля над европейским регионом."""
    lat = np.arange(72.0, 39.5, -0.5, dtype=np.float32)
    lon = np.arange(10.0, 80.5, 0.5, dtype=np.float32)
    steps = np.arange(layer.steps, dtype=np.float32)[:, None, None]
    latitudes = np.deg2rad(lat)[None, :, None]
    longitudes = np.deg2rad(lon)[None, None, :]
    phase = steps * (0.35 * layer.step_hours / canon.STEP_HOURS) + longitudes * 2
    wave = np.sin(phase) * np.cos(latitudes)
    u = 3.0 + 4.0 * wave
    v = 2.0 + 3.0 * np.cos(phase * 0.8 + latitudes)
    temperature = 280.0 + 10.0 * wave - (latitudes - 0.7) * 8.0

    values = {
        "2t": temperature,
        "10u": u,
        "10v": v,
        "msl": 101_325.0 + 950.0 * np.sin(phase * 0.45 + latitudes),
        "tp_1h": np.maximum(0.0, np.sin(phase - 1.8)) * 0.0015,
        "tcc": np.clip(0.45 + 0.45 * np.sin(phase - 1.1), 0.0, 1.0),
        "2d": temperature - 2.5,
        "i10fg": np.hypot(u, v) + 4.0,
    }
    times = np.array(
        [
            np.datetime64(started.replace(tzinfo=None), "ns")
            + np.timedelta64(index * layer.step_hours, "h")
            for index in range(layer.steps)
        ]
    )
    dataset = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.broadcast_to(values[name], (layer.steps, lat.size, lon.size)).astype(np.float32),
            )
            for name in layer.surface_vars
        },
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={"init_time": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "kind": "forecast"},
    )
    for name in layer.surface_vars:
        dataset[name].attrs["units"] = canon.UNITS[name]
    return dataset


def main() -> None:
    """Поднять локальный demo на http://127.0.0.1:8000/."""
    root = build_demo_store()
    uvicorn.run(create_app(root), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover - командная точка входа
    main()
