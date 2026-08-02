"""Прогон как каталог: имя, артефакт и отметка о пропуске.

Второе место, где расписание (`pipeline.schedule`) встречается с хранилищем
(`storage`), — по той же причине, что и `pipeline.inputs`: адаптеру запрещено
знать про хранилище, а хранилищу про расписание, и написать это больше негде
(`tests/test_boundaries.py`).

По существу здесь одна вещь: `skip.json` кто-то обязан записать. Схема отметки
и журнал прогонов появились раньше писателя, и до него критерий «пропуск
прогона виден, а не замаскирован» выполнялся только на бумаге: пропущенный
прогон не писал ничего, потому что писать было некому.

Имя каталога прогона тоже отсюда. Строится оно из срока, а не приходит
строкой: `runs/2026-08-01T00Z` и `forecast/2026-08-01T00Z` должны совпадать до
символа, иначе отметка о пропуске ложится в каталог, который никто больше не
называет.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pipeline.schedule import MISSED, TIME_FORMAT, Cycle, cycle, phase
from storage.manifest import MANIFEST_NAME, SKIP_NAME, build_skip, write_skip
from storage.publish import run_path

#: Имя каталога прогона: срок с точностью до часа. Минуты и секунды в имени
#: были бы всегда нулевыми — прогоны идут каждые шесть часов ровно.
RUN_ID_FORMAT = "%Y-%m-%dT%HZ"

#: Артефакт прогона в манифесте и в отметке о пропуске (docs/DATA_CONTRACT.md §3).
ARTIFACT_PREFIX = "forecast/"


def run_id(init_time: str) -> str:
    """Имя каталога прогона по его сроку: `2026-08-01T00:00:00Z` → `2026-08-01T00Z`.

    Срок проверяется расписанием, а не форматной строкой: `03Z` разбирается
    без ошибок и даёт каталог `runs/2026-08-01T03Z`, которого не ждёт ни один
    другой компонент. Имя каталога — ровно то место, где неверный час
    остаётся навсегда.
    """
    return _parse(cycle(init_time).init_time).strftime(RUN_ID_FORMAT)


def artifact(init_time: str) -> str:
    """Имя артефакта прогона: `forecast/2026-08-01T00Z`."""
    return ARTIFACT_PREFIX + run_id(init_time)


def skip_if_missed(
    root: str | Path,
    run: Cycle,
    *,
    now: datetime,
    reason: str,
    waited_for: Sequence[str] = (),
    attempts: int = 0,
) -> Path | None:
    """Записать отметку о пропуске, если прогон уже не состоится.

    Отметка ставится только в фазе `missed`, то есть после дедлайна публикации:
    до него прогон ещё можно спасти откатом на GFS, и пропуск, записанный
    раньше, — это ложная тревога, которую дежурный увидит вместо настоящей.

    `None` возвращается в двух случаях, и оба нормальные: срок публикации ещё
    не вышел, либо в каталоге лежит манифест — прогон состоялся или оборвался
    на середине, и то и другое видно журналом, а «пропущен» про них неправда.

    Повторный вызов первую отметку не переписывает: у прогона в фазе `missed`
    попыток больше не будет, и второе `created_at` сдвинуло бы время отказа
    вперёд — ровно то число, по которому потом считают, сколько сервис молчал.
    """
    if phase(run, now) != MISSED:
        return None
    directory = run_path(root, run_id(run.init_time))
    if (directory / MANIFEST_NAME).is_file():
        return None
    mark = directory / SKIP_NAME
    if mark.is_file():
        return mark
    return write_skip(
        mark,
        build_skip(
            artifact(run.init_time),
            init_time=run.init_time,
            reason=reason,
            waited_for=waited_for,
            deadline=run.publish_by.strftime(TIME_FORMAT),
            attempts=attempts,
            created_at=now,
        ),
    )


def _parse(init_time: str) -> datetime:
    """Срок, уже проверенный `cycle`. Время в UTC — как у соседа в
    `pipeline.schedule`: наивное отсюда однажды уедет в сравнение и уедет молча."""
    return datetime.strptime(init_time, TIME_FORMAT).replace(tzinfo=UTC)
