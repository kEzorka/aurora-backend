"""Operational analysis ingest and ERA5 materialization entry point.

Analysis ingest keeps source payloads immutable below ``raw/analysis``, builds
the two canonical times required by Aurora, validates them, writes a versioned
Zarr and only then atomically moves ``analysis/recent``.  Network transports
remain in adapters; this module owns orchestration and storage publication.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import xarray as xr

from adapters import ecmwf, era5_arco, fetch, opendata
from adapters import plan as input_plan
from contracts import canon
from pipeline import monthly
from pipeline.inputs import manifest_inputs
from pipeline.schedule import TIME_FORMAT, cycle, delays, latest_init_time
from storage.layout import layer_path
from storage.manifest import Input
from storage.write import write_layer
from validators import ValidationReport, raise_if_rejected, validate, write_report

ROOT_ENV: Final = "AURORA_ROOT"
DEFAULT_ROOT: Final = "/data/aurora"
INPUTS_NAME: Final = "inputs.json"
VALIDATION_NAME: Final = "validation.json"
ANALYSIS_NAMES: Final = (*canon.SURFACE_INGESTED_VARS, *canon.ATMOS_VARS)

Loader = Callable[[input_plan.Request, Path], tuple[xr.Dataset, str]]


class Collected(NamedTuple):
    dataset: xr.Dataset
    inputs: tuple[Input, ...]


def collect(
    raw_root: str | Path,
    valid_time: str,
    *,
    loader: Loader,
    names: Sequence[str] = ANALYSIS_NAMES,
) -> Collected:
    """Download/read one canonical time and retain every input checksum."""
    requests = input_plan.plan(valid_time, names)
    parts: list[tuple[input_plan.Request, xr.Dataset]] = []
    checksums: dict[input_plan.Request, str] = {}
    directory = Path(raw_root) / _run_id(valid_time)
    for position, request in enumerate(requests):
        base = directory / f"{position:02d}-{_safe(request.source)}-{_safe(request.stream)}"
        dataset, digest = loader(request, base)
        parts.append((request, dataset))
        checksums[request] = digest
    return Collected(
        input_plan.assemble(parts, valid_time=valid_time, names=names),
        manifest_inputs(requests, checksums),
    )


def combine(first: Collected, second: Collected, *, init_time: str) -> Collected:
    """Two adjacent canonical times in chronological contract order."""
    expected = (_previous(init_time), init_time)
    got = tuple(
        str(np.asarray(part.dataset["time"].values, dtype="datetime64[s]")[0]) + "Z"
        for part in (first, second)
    )
    if got != expected:
        raise ValueError(f"analysis times: got {got}, expected {expected}")
    dataset = xr.concat(
        (first.dataset, second.dataset),
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="equals",
        combine_attrs="override",
    )
    dataset.attrs = {**second.dataset.attrs, "init_time": init_time, "kind": "analysis"}
    return Collected(dataset, (*first.inputs, *second.inputs))


def ingest_analysis(
    root: str | Path,
    init_time: str,
    *,
    loader: Loader | None = None,
) -> Path:
    """Build, validate and atomically publish ``analysis/recent``."""
    cycle(init_time)
    store = Path(root).expanduser()
    run_id = _run_id(init_time)
    final = store / "analysis" / "runs" / run_id
    data = final / "data"
    if data.is_dir():
        _point(layer_path(store, "analysis"), data)
        return data

    active_loader = loader or operational_loader
    raw = store / "raw" / "analysis"
    previous = collect(raw, _previous(init_time), loader=active_loader)
    current = collect(raw, init_time, loader=active_loader)
    collected = combine(previous, current, init_time=init_time)
    report = validate(collected.dataset, layer="analysis")

    staging = store / "scratch" / f"analysis-{run_id}"
    write_report(report, staging / VALIDATION_NAME)
    raise_if_rejected(report)
    publish_analysis(
        store,
        init_time,
        collected.dataset,
        collected.inputs,
        report,
    )
    return data


def publish_analysis(
    root: str | Path,
    init_time: str,
    dataset: xr.Dataset,
    inputs: Sequence[Input],
    report: ValidationReport,
    *,
    layer: canon.Layer = canon.LAYERS["analysis"],
) -> Path:
    """Versioned write followed by an atomic ``analysis/recent`` pointer."""
    raise_if_rejected(report)
    store = Path(root)
    run_id = _run_id(init_time)
    final = store / "analysis" / "runs" / run_id
    data = final / "data"
    if data.is_dir():
        _point(layer_path(store, "analysis"), data)
        return data
    staging = store / "scratch" / f"analysis-{run_id}"
    if staging.exists():
        # A validation report from the current attempt is allowed; any Zarr
        # payload means a prior write was interrupted and requires inspection.
        occupied = [path for path in staging.iterdir() if path.name != VALIDATION_NAME]
        if occupied:
            raise FileExistsError(f"{staging}: interrupted analysis staging is not overwritten")
    staging.mkdir(parents=True, exist_ok=True)
    write_layer(dataset, staging / "data", layer)
    write_report(report, staging / VALIDATION_NAME)
    _write_inputs(staging / INPUTS_NAME, inputs)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    _point(layer_path(store, "analysis"), data)
    return data


def operational_loader(request: input_plan.Request, base: Path) -> tuple[xr.Dataset, str]:
    """Live Open Data/ARCO loader; imports optional backends only on use."""
    retrieved = datetime.now(UTC).strftime(TIME_FORMAT)
    if request.source == ecmwf.SOURCE:
        raw = base.with_suffix(".grib2")
        if not raw.is_file():
            opendata.download(
                request,
                raw,
                transport=fetch.http,
                gap=0,
                delays=delays(),
            )
        dataset = ecmwf.read_messages(
            raw,
            source_url=opendata.data_url(request),
            retrieved_at=retrieved,
        ).load()
        return dataset, fetch.checksum(raw)
    if request.source == "era5t" and request.names == ("ci",):
        raw = base.with_suffix(".zarr")
        if not raw.is_dir():
            _snapshot_era5t(request, raw)
        with xr.open_zarr(raw, chunks={}) as snapshot:
            moment = _moment(request.valid_time)
            dataset = era5_arco.read_slice(
                snapshot,
                "ci",
                moment,
                source_url=f"{era5_arco.ARCO_URL}#sea_ice_cover@{request.valid_time}",
                retrieved_at=retrieved,
                # The plan explicitly requests the preliminary five-day-old
                # value even when replayed years later.
                now=moment + timedelta(days=input_plan.ERA5T_LAG_DAYS),
            ).load()
        return dataset, fetch.checksum_tree(raw)
    raise ValueError(f"unsupported analysis request: {request}")


def _snapshot_era5t(request: input_plan.Request, target: Path) -> None:
    archive = era5_arco.open_archive(chunks={})
    try:
        source_name = era5_arco.SOURCE_NAMES["ci"]
        stamp = np.datetime64(_moment(request.valid_time).replace(tzinfo=None), "ns")
        raw = archive[[source_name]].sel(time=[stamp]).load().drop_encoding()
        if raw.sizes.get("time") != 1:
            raise ValueError(f"ERA5T {request.valid_time}: expected exactly one time")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".tmp")
        if staging.exists():
            raise FileExistsError(f"{staging}: interrupted ERA5T snapshot is not overwritten")
        raw.to_zarr(staging, mode="w-", zarr_format=3, consolidated=True)
        os.replace(staging, target)
    finally:
        archive.close()


def _write_inputs(path: Path, inputs: Sequence[Input]) -> Path:
    payload = [{**entry._asdict(), "fields": list(entry.fields)} for entry in inputs]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def _point(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"{link}: analysis/recent must be a symlink")
    temporary = link.with_name(link.name + ".tmp")
    if temporary.is_symlink() or temporary.is_file():
        temporary.unlink()
    elif temporary.exists():
        raise RuntimeError(f"{temporary}: temporary pointer is a directory")
    os.symlink(os.path.relpath(target, link.parent), temporary)
    os.replace(temporary, link)


def _previous(init_time: str) -> str:
    return (_moment(init_time) - timedelta(hours=canon.STEP_HOURS)).strftime(TIME_FORMAT)


def _run_id(valid_time: str) -> str:
    return _moment(valid_time).strftime("%Y-%m-%dT%HZ")


def _safe(value: str) -> str:
    return value.replace("/", "-").replace(" ", "-")


def _moment(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, TIME_FORMAT)
    except ValueError as error:
        raise ValueError(f"time: got {value!r}, expected {TIME_FORMAT}") from error
    return parsed.replace(tzinfo=UTC)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--kind", required=True, choices=("analysis", "era5"))
    cli.add_argument("--root", type=Path, default=None)
    cli.add_argument("--init", default=None)
    cli.add_argument("--from", dest="start", default=None)
    cli.add_argument("--to", dest="stop", default=None)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = (args.root or Path(os.environ.get(ROOT_ENV, DEFAULT_ROOT))).expanduser()
    if args.kind == "analysis":
        init_time = args.init or latest_init_time(datetime.now(UTC))
        print(ingest_analysis(root, init_time))
        return 0
    if args.start is None or args.stop is None:
        parser().error("--kind era5 requires --from and --to")
    print(monthly.build(root, _iso_moment(args.start), _iso_moment(args.stop)))
    return 0


def _iso_moment(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
