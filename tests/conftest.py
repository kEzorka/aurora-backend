"""Опубликованный прогон в tmp_path: то, что читатель видит через указатель.

Фикстура общая для `tests/storage` и `tests/api`: обе полосы читают одно и то
же хранилище, и расходиться в том, что в нём лежит, им нельзя.

Сетка маленькая, а значения — правдоподобные и разные по слоям: перевод
единиц проверяется числом (288.15 K = 15 °C, 101 325 Па = 1013.25 гПа,
ветер 3 и 4 = 5 м/с), а подмена слоя — тем, какие значения пришли.
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from api.app import create_app
from contracts import canon
from storage.manifest import Input, Model, build_manifest
from storage.publish import publish_run, stage_path
from storage.write import write_layer

#: Слои прогона в тесте: те же имена и шаги, что в каноне, но восемь полей
#: и в шестичасовом слое — 90 полей на настоящей сетке стоят 373 МБ на срок.
COARSE = canon.Layer("coarse", canon.HOURLY_VARS, (), canon.STEP_HOURS, 3)
HOURLY = canon.Layer("hourly", canon.HOURLY_VARS, (), canon.FINE_STEP_HOURS, 4)

INIT_TIME = "2026-08-01T00:00:00Z"

#: 288.15 K это ровно 15 °C, 101 325 Па — 1013.25 гПа, а 3 и 4 м/с дают 5.
VALUES = {"2t": 288.15, "10u": 3.0, "10v": 4.0, "msl": 101_325.0}


def _forecast(layer: canon.Layer, offset: float = 0.0) -> xr.Dataset:
    ny, nx = 6, 8
    data = {}
    for name in layer.surface_vars:
        field = np.full((layer.steps, ny, nx), VALUES.get(name, 0.5) + offset, dtype=np.float32)
        data[name] = (("time", "lat", "lon"), field)
    ds = xr.Dataset(
        data,
        coords={
            "time": np.array(
                [
                    np.datetime64("2026-08-01T00") + np.timedelta64(layer.step_hours * i, "h")
                    for i in range(layer.steps)
                ],
                dtype="datetime64[ns]",
            ),
            "lat": np.linspace(90.0, -90.0, ny),
            "lon": np.linspace(-180.0, 135.0, nx),
        },
        attrs={"init_time": INIT_TIME, "kind": "forecast"},
    )
    for name in layer.surface_vars:
        ds[name].attrs["units"] = canon.UNITS[name]
    return ds


def _manifest(run_id: str) -> dict[str, object]:
    return build_manifest(
        f"forecast/{run_id}",
        inputs=(
            Input(
                "ifs-analysis",
                "2026-08-01T00:00Z",
                "sha256:" + "a" * 64,
                "ifs/0p25/oper",
                canon.SURFACE_INGESTED_VARS,
            ),
        ),
        model=Model("aurora", "aurora-0.25-v1.5", "9f2c1ab"),
        steps=COARSE.steps,
        timings_sec={"ingest": 1, "normalize": 1, "inference": 1, "write": 1},
        created_at=datetime(2026, 8, 1, 5, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def published(tmp_path: Path) -> Path:
    """Хранилище с одним опубликованным прогоном. Возвращает корень."""
    run_id = "2026-08-01T00Z"
    staged = stage_path(tmp_path, run_id)
    write_layer(_forecast(COARSE), staged / "coarse", COARSE)
    # Часовой слой отличается на градус: по значению видно, из какого слоя
    # пришёл ответ, — тест на step_hours не зависит от совпадения сроков.
    write_layer(_forecast(HOURLY, offset=1.0), staged / "hourly", HOURLY)
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "validation.json").write_text('{"ok": true, "checks": []}', encoding="utf-8")
    publish_run(tmp_path, run_id, manifest=_manifest(run_id))
    return tmp_path


@pytest.fixture
def client(published: Path) -> TestClient:
    return TestClient(create_app(published))
