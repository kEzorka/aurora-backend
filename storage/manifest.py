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

#: Отметка о пропуске: прогон, которого не будет (docs/PIPELINE.md §3.5).
#:
#: Отдельный документ, а не манифест с пустыми полями. У пропущенного прогона
#: нет ни артефакта, ни входов, ни шагов, ни таймингов, и провести его через
#: `build_manifest` можно было бы только сняв с манифеста все проверки формы —
#: то есть разрешив нулевые входы и нулевые шаги настоящим прогонам тоже.
SKIP_NAME: Final = "skip.json"

#: Почему прогона не будет. Набор закрыт: «не получилось» в отметке о пропуске
#: отвечает ровно на тот вопрос, который и так был ясен, а разные причины
#: требуют разного — источник молчит, значит ждать, а валидатор отверг срез,
#: значит смотреть в отчёт.
SKIP_REASONS: Final = (
    # Ни ECMWF, ни GFS не отдали данные до дедлайна отката.
    "no_input",
    # Данные приехали, но не прошли валидаторы: публиковать нечего.
    "invalid_input",
    # Инференс не уложился в дедлайн публикации либо не дошёл до конца.
    "no_forecast",
)

#: Формат времени в манифесте — ISO 8601 в UTC, секундная точность.
TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


class Input(NamedTuple):
    """Один вход прогона: откуда, на какой срок, с какой контрольной суммой.

    `source` — словарь провенанса (`canon.SOURCES`), `stream` — адрес у
    поставщика. Разделены потому, что `ifs/0p25/oper` и `aifs-single/0p25/oper`
    дают один и тот же `ifs-analysis`, а облачность приходит только из второго
    (ADDENDUM-01 §1, BACKLOG 1.10).

    `fields` перечисляет, что именно приехало этой загрузкой. Без него вход на
    срок пятидневной давности — это просто строка с чужой датой, и понять, что
    устарела ровно сплочённость льда, а не весь срез, нельзя.
    """

    source: str
    valid_time: str
    checksum: str
    stream: str
    fields: tuple[str, ...]


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
        if not entry.stream:
            raise ValueError("stream: вход без потока не говорит, откуда взялось поле")
        unknown = [name for name in entry.fields if name not in canon.UNITS]
        if not entry.fields or unknown:
            raise ValueError(
                f"fields: got {list(entry.fields)}, expected known names from the canon"
            )
    if steps <= 0:
        raise ValueError(f"steps: got {steps}, expected a positive number")
    if set(timings_sec) != set(TIMING_KEYS):
        raise ValueError(f"timings_sec: got {sorted(timings_sec)}, expected {sorted(TIMING_KEYS)}")

    moment = created_at or datetime.now(UTC)
    return {
        "artifact": artifact,
        "created_at": moment.astimezone(UTC).strftime(TIME_FORMAT),
        # `fields` — кортеж, а кортежа в JSON нет: без явного `list` манифест на
        # диске и манифест в памяти перестают сравниваться, и тесты на
        # круговой обход начинают ловить не ошибку записи, а тип.
        "inputs": [{**entry._asdict(), "fields": list(entry.fields)} for entry in inputs],
        "model": model._asdict(),
        "steps": steps,
        "timings_sec": {key: int(timings_sec[key]) for key in TIMING_KEYS},
        "validation": validation,
        "published": False,
    }


def build_skip(
    artifact: str,
    *,
    init_time: str,
    reason: str,
    waited_for: Sequence[str],
    deadline: str,
    attempts: int,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Собрать отметку о пропуске прогона.

    «Пропущенный прогон — нормальная ситуация, она должна быть видна, а не
    замаскирована» (docs/PIPELINE.md §3.5). Видна она ровно тогда, когда
    отвечает на три вопроса: какого прогона нет, почему и до каких пор ждали.

    `waited_for` — потоки, которых не дождались, обычными строками. Хранилище
    про адаптеры не знает (`tests/test_boundaries.py`), и подставляет их сюда
    конвейер; пустой список означал бы «прогон пропущен, источники ни при чём»,
    что для `no_input` неправда.

    `attempts` — сколько раз попробовали. Ноль здесь честен и важен: он
    отличает «источник молчал полтора часа» от «не пробовали вовсе, потому что
    воркер лежал».
    """
    if reason not in SKIP_REASONS:
        raise ValueError(f"reason: got {reason!r}, expected one of {', '.join(SKIP_REASONS)}")
    if reason == "no_input" and not waited_for:
        raise ValueError("waited_for: пропуск из-за источника обязан назвать источник")
    if attempts < 0:
        raise ValueError(f"attempts: got {attempts}, expected zero or more")

    moment = created_at or datetime.now(UTC)
    return {
        "artifact": artifact,
        "created_at": moment.astimezone(UTC).strftime(TIME_FORMAT),
        "init_time": init_time,
        "reason": reason,
        "waited_for": list(waited_for),
        "deadline": deadline,
        "attempts": attempts,
        # Не `published`: у пропуска нечего публиковать. Ключ назван иначе
        # намеренно — совпади он с манифестом, отметка о пропуске прошла бы
        # через `read_manifest` и `published_run` как прогон с `published: false`,
        # то есть как оборванная публикация, а это другая история.
        "skipped": True,
    }


def write_skip(path: str | Path, skip: Mapping[str, Any]) -> Path:
    """Записать отметку о пропуске. Та же атомарность, что у манифеста."""
    if not skip.get("skipped"):
        raise ValueError("skipped: отметка о пропуске без флага пропуска — это не она")
    return _dump(Path(path), skip)


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
