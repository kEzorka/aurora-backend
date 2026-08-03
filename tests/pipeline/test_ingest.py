"""Analysis ingest orchestration without network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from adapters.plan import Request
from contracts import canon
from pipeline.ingest import Collected, collect, combine, main, publish_analysis
from storage.manifest import Input
from validators import RejectedError
from validators.result import ValidationReport, fail, ok

PREVIOUS = "2026-07-31T18:00:00Z"
INIT = "2026-08-01T00:00:00Z"


def _part(request: Request) -> xr.Dataset:
    data = {
        name: xr.DataArray(
            np.ones((1, 2, 3), dtype=np.float32),
            dims=("time", "lat", "lon"),
            attrs={"units": canon.UNITS[name]},
        )
        for name in request.names
    }
    return xr.Dataset(
        data,
        coords={
            "time": np.array([request.valid_time.rstrip("Z")], dtype="datetime64[ns]"),
            "lat": [90.0, -90.0],
            "lon": [-180.0, -179.75, -179.5],
        },
        attrs={
            "source": request.source,
            "kind": "analysis",
            "adapter_version": "test",
            "init_time": request.valid_time,
        },
    )


def test_collect_keeps_source_boundaries_and_checksums(tmp_path: Path) -> None:
    paths: list[Path] = []

    def loader(request: Request, base: Path) -> tuple[xr.Dataset, str]:
        paths.append(base)
        return _part(request), "sha256:" + str(len(paths)) * 64

    result = collect(tmp_path, INIT, loader=loader, names=("2t", "ci"))

    assert set(result.dataset.data_vars) == {"2t", "ci"}
    assert result.dataset["ci"].attrs["valid_time"] == "2026-07-27T00:00:00Z"
    assert [entry.source for entry in result.inputs] == ["ifs-analysis", "era5t"]
    assert [entry.checksum for entry in result.inputs] == [
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
    ]
    assert len(paths) == len(set(paths)) == 2


def test_combine_requires_previous_then_current() -> None:
    request_previous = Request("ifs-analysis", "ifs/0p25/oper", PREVIOUS, ("2t",))
    request_current = Request("ifs-analysis", "ifs/0p25/oper", INIT, ("2t",))
    entry = Input("ifs-analysis", INIT, "sha256:" + "a" * 64, "ifs/0p25/oper", ("2t",))

    combined = combine(
        Collected(_part(request_previous), (entry,)),
        Collected(_part(request_current), (entry,)),
        init_time=INIT,
    )

    assert combined.dataset["time"].values.tolist() == [
        np.datetime64("2026-07-31T18:00:00", "ns").tolist(),
        np.datetime64("2026-08-01T00:00:00", "ns").tolist(),
    ]
    assert combined.dataset.attrs["init_time"] == INIT
    assert len(combined.inputs) == 2

    with pytest.raises(ValueError, match="analysis times"):
        combine(
            Collected(_part(request_current), (entry,)),
            Collected(_part(request_previous), (entry,)),
            init_time=INIT,
        )


def test_publish_is_versioned_atomic_and_idempotent(tmp_path: Path) -> None:
    layer = canon.Layer("analysis-test", ("2t",), (), canon.STEP_HOURS, 2)
    dataset = xr.Dataset(
        {
            "2t": (
                ("time", "lat", "lon"),
                np.full((2, 2, 3), 280.0, dtype=np.float32),
            )
        },
        coords={
            "time": np.array([PREVIOUS.rstrip("Z"), INIT.rstrip("Z")], dtype="datetime64[ns]"),
            "lat": [90.0, -90.0],
            "lon": [-180.0, -179.75, -179.5],
        },
        attrs={"init_time": INIT, "kind": "analysis"},
    )
    dataset["2t"].attrs["units"] = "K"
    inputs = (Input("ifs-analysis", INIT, "sha256:" + "a" * 64, "ifs/0p25/oper", ("2t",)),)
    report = ValidationReport((ok("fixture", "structure"),))

    first = publish_analysis(tmp_path, INIT, dataset, inputs, report, layer=layer)
    second = publish_analysis(tmp_path, INIT, dataset, inputs, report, layer=layer)

    assert first == second
    assert first == (tmp_path / "analysis" / "recent").resolve()
    assert (tmp_path / "analysis" / "recent").is_symlink()
    assert xr.open_zarr(first, chunks=None).sizes == {"time": 2, "lat": 2, "lon": 3}
    assert (first.parent / "validation.json").is_file()
    assert '"source": "ifs-analysis"' in (first.parent / "inputs.json").read_text()
    assert len(list((tmp_path / "analysis" / "runs").iterdir())) == 1


def test_interrupted_staging_is_preserved_for_inspection(tmp_path: Path) -> None:
    staging = tmp_path / "scratch" / "analysis-2026-08-01T00Z"
    staging.mkdir(parents=True)
    (staging / "partial-zarr").write_text("do not discard", encoding="utf-8")
    report = ValidationReport((ok("fixture", "structure"),))
    layer = canon.Layer("tiny", ("2t",), (), canon.STEP_HOURS, 2)
    dataset = xr.Dataset({"2t": (("time",), np.array([1.0, 2.0], dtype=np.float32))})

    with pytest.raises(FileExistsError, match="not overwritten"):
        publish_analysis(tmp_path, INIT, dataset, (), report, layer=layer)
    assert (staging / "partial-zarr").read_text(encoding="utf-8") == "do not discard"


def test_rejected_analysis_is_never_published(tmp_path: Path) -> None:
    report = ValidationReport((fail("units", "semantics", "2t", "degC", "K"),))
    dataset = xr.Dataset({"2t": (("time",), np.array([1.0, 2.0], dtype=np.float32))})
    layer = canon.Layer("tiny", ("2t",), (), canon.STEP_HOURS, 2)

    with pytest.raises(RejectedError, match="2t"):
        publish_analysis(tmp_path, INIT, dataset, (), report, layer=layer)
    assert not (tmp_path / "analysis").exists()


def test_era5_cli_requires_a_period() -> None:
    with pytest.raises(SystemExit):
        main(["--kind", "era5"])
