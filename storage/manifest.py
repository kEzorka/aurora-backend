"""Манифест артефакта: чем он собран, из чего и опубликован ли.

Схема — `docs/DATA_CONTRACT.md` §3, поле в поле. Манифест собирается здесь, а
не в конвейере, потому что читает его тоже хранилище: по `published` API
решает, отдавать срез или прошлый прогон.

Контрольные суммы входов приходят готовыми: считает их тот, кто скачал файл
(`adapters`), а `storage` про адаптеры не знает и знать не должен
(`tests/test_boundaries.py`). Здесь проверяется только форма.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NamedTuple

from contracts import canon

#: Этапы, по которым конвейер отчитывается временем. Набор фиксирован: по нему
#: считаются метрики прогона (BACKLOG 6.2), и пропущенный этап там неотличим
#: от мгновенного.
TIMING_KEYS: Final = ("ingest", "normalize", "inference", "write")

MANIFEST_NAME: Final = "manifest.json"
VALIDATION_NAME: Final = "validation.json"

#: Формат времени в манифесте — ISO 8601 в UTC, секундная точность.
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


class Input(NamedTuple):
    """Один вход прогона: откуда, на какой срок, с какой контрольной суммой."""

    source: str
    valid_time: str
    checksum: str


class Model(NamedTuple):
    """Чем считали: имя, чекпоинт и ревизия кода."""

    name: str
    checkpoint: str
    revision: str


def build_manifest(
    artifact: str,
    *,
    inputs: Sequence[Input],
    model: Model,
    steps: int,
    timings_sec: Mapping[str, int],
    created_at: datetime | None = None,
    validation: str = VALIDATION_NAME,
) -> dict[str, Any]:
    """Собрать манифест. `published` всегда `False`: его ставит публикация."""
    if not inputs:
        raise ValueError("inputs: манифест без входов не говорит, из чего собран артефакт")
    for entry in inputs:
        if entry.source not in canon.SOURCES:
            raise ValueError(
                f"source: got {entry.source!r}, expected one of {', '.join(canon.SOURCES)}"
            )
        if not entry.checksum.startswith("sha256:"):
            raise ValueError(f"checksum: got {entry.checksum!r}, expected 'sha256:<hex>'")
    if steps <= 0:
        raise ValueError(f"steps: got {steps}, expected a positive number")
    if set(timings_sec) != set(TIMING_KEYS):
        raise ValueError(f"timings_sec: got {sorted(timings_sec)}, expected {sorted(TIMING_KEYS)}")

    moment = created_at or datetime.now(UTC)
    return {
        "artifact": artifact,
        "created_at": moment.astimezone(UTC).strftime(TIME_FORMAT),
        "inputs": [entry._asdict() for entry in inputs],
        "model": model._asdict(),
        "steps": steps,
        "timings_sec": {key: int(timings_sec[key]) for key in TIMING_KEYS},
        "validation": validation,
        "published": False,
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Записать манифест через `.tmp` и `replace`: половина манифеста хуже, чем его
    отсутствие. Уже поднятый флаг здесь отвергается — публикует `mark_published`."""
    if manifest.get("published"):
        raise ValueError("published: флаг ставится после записи данных, а не вместе с ней")
    return _dump(Path(path), manifest)


def read_manifest(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def mark_published(path: str | Path) -> dict[str, Any]:
    """Поднять `published` — последнее действие над артефактом (docs/STORAGE.md §5).

    До этого момента срез на диске лежит целиком, но читателю не обещан:
    прерванный прогон оставляет манифест с `false`, а не полуправду.
    """
    path = Path(path)
    manifest = {**read_manifest(path), "published": True}
    _dump(path, manifest)
    return manifest


def _dump(path: Path, manifest: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path
