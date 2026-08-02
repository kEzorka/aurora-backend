"""Атомарная публикация прогона.

Прогон собирается в `scratch/<run_id>`, переезжает переименованием в
`runs/<run_id>` и становится виден только тогда, когда на него переставлен
указатель `forecast/current`. Читатель ходит через указатель и потому видит
либо прошлый прогон целиком, либо новый целиком — прерваться посередине
публикация может, а показать половину нет (docs/STORAGE.md §5).

Порядок действий:

1. проверить разложенное — слои на месте, отчёт валидатора рядом;
2. записать манифест с `published: false`;
3. `os.replace` каталога: scratch и runs лежат на одной файловой системе,
   поэтому переезд атомарен и не стоит второй копии 17.4 ГБ;
4. свернуть прошлый прогон в восемь переменных (`forecast/previous`);
5. поднять `published` — последняя запись внутрь артефакта;
6. переставить указатели.

Прошлый прогон сворачивается, а не переподписывается целиком: 15.0 ГБ
шестичасового слоя в двух экземплярах — это 30 ГБ при ядре в 40
(docs/STORAGE.md §2). Удаление позапрошлых прогонов сюда не входит: это
ротация и квота scratch, задача 2.6.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import xarray as xr

from contracts import canon
from storage.manifest import MANIFEST_NAME, VALIDATION_NAME, mark_published, write_manifest
from storage.write import write_layer

#: Слои, без которых прогон не прогон: карты на 10 суток и часовой слой.
REQUIRED_LAYERS: Final = ("coarse", "hourly")

#: Сюда конвейер складывает прогон до публикации, отсюда он переименовывается.
SCRATCH_DIR: Final = "scratch"
RUNS_DIR: Final = "runs"

CURRENT_LINK: Final = "forecast/current"
PREVIOUS_LINK: Final = "forecast/previous"

#: Имя свёрнутого слоя внутри прогона, на который смотрит `forecast/previous`.
PREVIOUS_LAYER: Final = "previous"


def stage_path(root: str | Path, run_id: str) -> Path:
    """Куда конвейер пишет прогон до публикации."""
    return Path(root) / SCRATCH_DIR / run_id


def run_path(root: str | Path, run_id: str) -> Path:
    """Где прогон лежит после публикации. Ключ — идентификатор прогона."""
    return Path(root) / RUNS_DIR / run_id


def current_run(root: str | Path) -> Path | None:
    """Прогон, который видит читатель, или `None`, если публикаций не было."""
    return _target(Path(root) / CURRENT_LINK)


def previous_run(root: str | Path) -> Path | None:
    """Свёрнутый прошлый прогон или `None`."""
    return _target(Path(root) / PREVIOUS_LINK)


def publish_run(root: str | Path, run_id: str, *, manifest: Mapping[str, Any]) -> Path:
    """Опубликовать разложенный прогон и вернуть путь, по которому он лёг."""
    root = Path(root)
    staged = stage_path(root, run_id)
    final = run_path(root, run_id)

    if not staged.is_dir():
        raise FileNotFoundError(f"{staged}: прогон не разложен")
    missing = [name for name in REQUIRED_LAYERS if not (staged / name).is_dir()]
    if missing:
        raise ValueError(f"{run_id}: слоёв нет: {', '.join(missing)}")
    # Имя берётся из манифеста, а не из константы: манифест ссылается на отчёт
    # по имени, и ссылка в никуда — тот же непроверенный срез, только молча.
    report = str(manifest.get("validation", VALIDATION_NAME))
    if not (staged / report).is_file():
        raise ValueError(f"{run_id}: нет {report}, срез не проверен")
    if final.exists():
        raise FileExistsError(f"{final}: прогон {run_id} уже опубликован")
    _check_pointer(root / CURRENT_LINK)
    _check_pointer(root / PREVIOUS_LINK)

    write_manifest(staged / MANIFEST_NAME, manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)

    previous = current_run(root)
    reduced = _reduce_previous(previous) if previous is not None else None

    mark_published(final / MANIFEST_NAME)

    _point(root / CURRENT_LINK, final)
    if reduced is not None:
        _point(root / PREVIOUS_LINK, reduced)
    return final


def _reduce_previous(run: Path) -> Path:
    """Свернуть прогон до восьми шестичасовых переменных (docs/STORAGE.md §2)."""
    target = run / PREVIOUS_LAYER
    if target.exists():
        return target
    with xr.open_zarr(run / "coarse") as coarse:
        return write_layer(coarse, target, canon.LAYERS[PREVIOUS_LAYER])


def _target(link: Path) -> Path | None:
    return link.resolve() if link.is_symlink() else None


def _check_pointer(link: Path) -> None:
    """Указатель обязан быть ссылкой. Каталог на его месте — след ручного
    вмешательства: `replace` его не заменит, и публикация встанет посередине."""
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"{link}: указатель занят каталогом, а не ссылкой")


def _point(link: Path, target: Path) -> None:
    """Переставить указатель одним `replace`: между `unlink` и `symlink` есть
    момент, когда указателя нет вовсе, и читатель в этот момент получает 404.

    Путь относительный: хранилище переезжает вместе с диском.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(link.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.path.relpath(target, link.parent), tmp)
    os.replace(tmp, link)
