"""The published forecast, read through the `latest` pointer.

`publish.RunPublisher` writes a run and then renames a symlink over `latest`.
This is the other half: a reader that notices when that symlink moved and opens
the new run, without a restart and without a lock.

The check is a `readlink` on every access — a stat, tens of microseconds, against
the 11 ms a point query costs — and it is what makes the three-hourly cycle
invisible to callers. Requests in flight keep the store objects they already
hold, because Python keeps the old objects alive until the last reference goes;
the deleted files stay readable through their open descriptors. That is the whole
reason the producer renames a directory into place instead of writing into one
that is being read.

The run's two layouts are the same shape as the history stores and carry the same
coordinate names, so they are read by `MapStore` and `SeriesStore` unchanged. A
forecast is not a different kind of thing to read — it is the same grid, later.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from .map_store import MapStore
from .series_store import SeriesStore


class ForecastRun:
    """Whatever `<root>/latest` currently points at."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._target: str | None = None
        self._target_key: tuple | None = None
        self._series: SeriesStore | None = None
        self._maps: MapStore | None = None
        self._manifest: dict = {}

    # ------------------------------------------------------------------ state

    @property
    def available(self) -> bool:
        return (self.root / "latest").exists()

    def _refresh(self) -> None:
        link = self.root / "latest"
        if not link.exists():
            raise FileNotFoundError(f"no published run under {self.root}")
        target = os.readlink(link)
        run = self.root / target
        # Keyed on the directory's identity, not only on the symlink's text.
        # `produce.py --force` rebuilds the same run name in place, which removes
        # the directory this reader has open and creates a new one under the same
        # path; a reader comparing names alone would keep chunk handles into the
        # deleted tree and 500 on the next read.
        st = run.stat()
        key = (target, st.st_ino, st.st_mtime_ns)
        if key == self._target_key and self._series is not None:
            return
        # Bind all three at once. A half-swapped reader that answered /v1/point
        # from the new run and /v1/map from the old one would produce two
        # different forecasts for one timestamp and say nothing about it.
        series = SeriesStore(run / "series.zarr")
        maps = MapStore(run / "maps.zarr")
        manifest_file = run / "manifest.json"
        self._manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        self._series, self._maps = series, maps
        self._target, self._target_key = target, key

    @property
    def series(self) -> SeriesStore:
        self._refresh()
        return self._series  # type: ignore[return-value]

    @property
    def maps(self) -> MapStore:
        self._refresh()
        return self._maps  # type: ignore[return-value]

    @property
    def manifest(self) -> dict:
        self._refresh()
        return self._manifest

    # ------------------------------------------------------------------ facts

    @property
    def run_name(self) -> str:
        self._refresh()
        return str(self._target)

    @property
    def init_time(self) -> dt.datetime | None:
        raw = self.manifest.get("init_time")
        return dt.datetime.fromisoformat(raw) if raw else None

    @property
    def coverage(self) -> tuple[dt.datetime, dt.datetime] | None:
        return self.series.coverage

    def describe(self) -> dict:
        """What `/v1/meta` reports about this layer."""
        if not self.available:
            return {"name": "forecast", "available": False,
                    "reason": f"no published run under {self.root}; "
                              "run scripts/produce.py"}
        c = self.coverage
        m = self.manifest
        return {
            "name": "forecast",
            "available": True,
            "run": self.run_name,
            "init_time": m.get("init_time"),
            "model": m.get("model"),
            "source": m.get("source"),
            "precision": m.get("precision"),
            "variables": sorted(self.series.variables),
            # Reported per layer because it genuinely differs: this one is 720
            # rows ending at -89.75 while the history layers are 721 ending at
            # -90.0, and a caller near the pole has to be able to see that.
            "grid": m.get("grid"),
            "from": c[0].isoformat() + "Z" if c else None,
            "to": c[1].isoformat() + "Z" if c else None,
            "steps": m.get("steps"),
            "lead_hours": m.get("lead_hours"),
            "published_at": m.get("published_at"),
            "layout": "two copies: map-major for /v1/map, time-major for /v1/point",
        }
