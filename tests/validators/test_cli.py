"""Код возврата — это интерфейс: `make validate` должен ронять конвейер."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from validators.cli import layer_from_path, main


def _tiny_broken_slice(path: Path) -> Path:
    """Мини-срез не в каноне: сетка не 721x1440, температура — нули."""
    ds = xr.Dataset(
        {"2t": (("time", "lat", "lon"), np.zeros((1, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2026-08-01T00"], dtype="datetime64[ns]"),
            "lat": np.array([1.0, 0.0]),
            "lon": np.array([0.0, 0.25]),
        },
    )
    ds["2t"].attrs["units"] = "K"
    ds.to_zarr(path)
    return path


def test_rejected_slice_exits_nonzero_and_leaves_a_report(tmp_path: Path) -> None:
    target = _tiny_broken_slice(tmp_path / "forecast.zarr")
    assert main([str(target)]) == 1
    payload = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert any(c["level"] == "structure" for c in payload["checks"] if not c["passed"])


def test_report_destination_can_be_overridden(tmp_path: Path) -> None:
    target = _tiny_broken_slice(tmp_path / "forecast.zarr")
    out = tmp_path / "reports" / "validation.json"
    assert main([str(target), "--out", str(out)]) == 1
    assert out.exists()


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("artifacts/2026-08-01T00Z/forecast/current/hourly", "hourly"),
        ("artifacts/2026-08-01T00Z/forecast/current/coarse", "coarse"),
        ("artifacts/2026-08-01T00Z/analysis", "analysis"),
        # Ничего не названо слоём — основной слой прогноза.
        ("artifacts/2026-08-01T00Z/forecast.zarr", "coarse"),
    ],
)
def test_layer_is_taken_from_the_path(path: str, expected: str) -> None:
    assert layer_from_path(Path(path)) == expected


def test_missing_artifacts_directory_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--latest"]) == 2


def test_no_arguments_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
