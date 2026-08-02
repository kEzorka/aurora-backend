"""Ротация прогонов и чистка scratch (BACKLOG 2.6).

Публикация ничего не удаляет — намеренно (`storage.publish`). Значит удаляет
это место, и весь вопрос в том, что именно пережить обязано.

Инвариант один: на диске нужны ровно два набора данных — текущий прогон
целиком и свёртка `previous` в прошлом. Всё остальное в `runs/` — слои, на
которые не смотрит ни один указатель, а читатель ходит только через указатели
(docs/STORAGE.md §5). Прогон — это 17.4 ГБ каждые шесть часов; без ротации
диск кончается за неделю.

Каталог прогона при этом **не** удаляется: из него уходят слои, а `manifest.json`,
`validation.json` и отметка о пропуске остаются. Это не бережливость, а
единственный способ сохранить журнал: `storage.read.run_log` перечисляет
прогоны по каталогам, и снесённый каталог — это прогон, которого как будто не
было. Пропуск, стёртый ротацией, ровно тот случай, ради которого журнал и
писали. Стоит это килобайты на прогон против гигабайтов, которые уходят.

Scratch чистится по возрасту, а не по указателям: там нет ничего, на что
кто-то смотрит, — это место перекладки, и всё, что старше суток, осталось от
упавшей публикации (docs/STORAGE.md §6). Возраст берётся у каталога, а не у
файлов внутри: перекладка идёт минутами, и сутки её ни при каком раскладе не
застанут посередине.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from storage.manifest import MANIFEST_NAME, SKIP_NAME, VALIDATION_NAME, read_manifest
from storage.publish import (
    PREVIOUS_LAYER,
    RUNS_DIR,
    SCRATCH_DIR,
    current_run,
    previous_run,
)

#: Что остаётся в каталоге прогона после ротации. Всё это — документы, а не
#: данные: журнал прогонов читает их и без слоёв.
KEPT_FILES: Final = (MANIFEST_NAME, VALIDATION_NAME, SKIP_NAME)

#: Резерв под перекладку из docs/STORAGE.md §3. Не «оставить свободным», а
#: число, с которым сверяются: превышение — это повод чистить, а не считать.
SCRATCH_QUOTA_BYTES: Final = 30 * (1 << 30)

#: Возраст, после которого каталог в scratch — мусор упавшей публикации.
#: Перекладка идёт минутами, и сутки её посередине не застанут.
SCRATCH_MAX_AGE: Final = timedelta(hours=24)

#: Сколько прогон имеет право лежать в `runs/` без флага `published` и без
#: указателя на себя. Между переездом и переставленным указателем публикация
#: пересобирает производные слои, и это минуты, в течение которых прогон
#: выглядит ровно как мусор (`storage.publish`). Ротация, запущенная в эту
#: минуту, снесла бы слои из-под живой публикации. Шесть часов — период между
#: прогонами: то, что не доехало за него, не доедет.
PUBLISHING_GRACE: Final = timedelta(hours=6)


def rotate(
    root: str | Path,
    *,
    keep: Iterable[str] = (),
    now: datetime | None = None,
    grace: timedelta = PUBLISHING_GRACE,
) -> tuple[Path, ...]:
    """Снять слои, на которые не смотрит ни один указатель.

    Текущий прогон не трогается вовсе. У прошлого остаётся свёртка `previous`
    — то, на что смотрит второй указатель, — а карты и ряды уходят: 15.0 ГБ
    шестичасового слоя в двух экземплярах при ядре в 40 ГБ (docs/STORAGE.md §2).
    У всех остальных прогонов уходят все слои.

    `keep` — идентификаторы прогонов, которые трогать нельзя. Через него
    архив прогонов (BACKLOG 4.4) спасает то, что отложил: указателя на архив
    нет, а по инварианту ротации всё без указателя — мусор.

    Прогон, публикация которого идёт прямо сейчас, тоже без указателя: он уже
    переехал в `runs/`, но производные слои ещё собираются, и `published` в
    манифесте пока `false` (`storage.publish`). Такой прогон не трогается,
    пока не выйдет `grace`.

    Возвращает удалённое, по порядку. Пустой ответ означает, что удалять было
    нечего, а не что ротация не сработала.
    """
    doomed_paths = _walk(Path(root), keep=set(keep), now=now or datetime.now(UTC), grace=grace)
    for path in doomed_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return doomed_paths


def sweep_scratch(
    root: str | Path,
    *,
    now: datetime | None = None,
    older_than: timedelta = SCRATCH_MAX_AGE,
) -> tuple[Path, ...]:
    """Снести из scratch всё, что старше суток.

    Указателей на scratch нет ни у кого: это место перекладки, и каталог,
    переживший сутки, остался от публикации, которая не дошла до переезда.
    Оставленный, он занимает те же 17.4 ГБ, что и прогон, — и занимает их в
    момент, когда приходит следующий.
    """
    scratch = Path(root) / SCRATCH_DIR
    if not scratch.is_dir():
        return ()
    moment = now or datetime.now(UTC)
    removed: list[Path] = []
    for staged in sorted(scratch.iterdir()):
        if not staged.is_dir():
            continue
        age = moment - datetime.fromtimestamp(staged.stat().st_mtime, UTC)
        if age >= older_than:
            shutil.rmtree(staged)
            removed.append(staged)
    return tuple(removed)


def scratch_bytes(root: str | Path) -> int:
    """Сколько занято в scratch. Считается обходом, а не `du`: квота проверяется
    перед публикацией, а лишний процесс на этом пути — лишний способ упасть."""
    scratch = Path(root) / SCRATCH_DIR
    if not scratch.is_dir():
        return 0
    return sum(item.stat().st_size for item in scratch.rglob("*") if item.is_file())


def over_quota(root: str | Path, *, quota: int = SCRATCH_QUOTA_BYTES) -> bool:
    """Перебрал ли scratch резерв.

    Отдельной функцией, а не проверкой внутри публикации: решать, что делать с
    переполненным scratch, — дело конвейера. Он может подождать, может почистить,
    может отметить прогон пропущенным; хранилище не знает, который час.
    """
    return scratch_bytes(root) > quota


def main(argv: list[str] | None = None) -> int:
    """`python -m storage.rotate <корень>` — чистка руками и по расписанию.

    Инструмент дежурного, а не часть публикации: публикация обязана быть
    атомарной и короткой, а удаление 17.4 ГБ — ни то ни другое. Печатается
    удалённое, потому что «диск не переполняется» проверяют, глядя на список,
    а не на свободное место постфактум.
    """
    parser = argparse.ArgumentParser(prog="storage.rotate")
    parser.add_argument("root", type=Path, help="корень хранилища")
    parser.add_argument(
        "--keep", action="append", default=[], help="идентификатор прогона, который не трогать"
    )
    parser.add_argument("--scratch", action="store_true", help="ещё и вычистить scratch")
    parser.add_argument(
        "--quota", type=int, default=SCRATCH_QUOTA_BYTES, help="резерв scratch в байтах"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="показать, что ушло бы, и ничего не удалять"
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        for path in doomed(args.root, keep=args.keep):
            print(f"снесла бы {path}")
        return 0

    for path in rotate(args.root, keep=args.keep):
        print(f"снесено {path}")
    if args.scratch:
        for path in sweep_scratch(args.root):
            print(f"scratch: снесено {path}")
    if over_quota(args.root, quota=args.quota):
        # Код возврата, а не только строка: переполненный scratch — это
        # следующая перекладка, которая упадёт на середине.
        print(f"scratch: {scratch_bytes(args.root)} байт при резерве {args.quota}")
        return 1
    return 0


def doomed(root: str | Path, *, keep: Iterable[str] = ()) -> tuple[Path, ...]:
    """Что унесла бы ротация. Тот же обход, но без удаления.

    Нужен ровно для `--dry-run`: команда, которая сносит гигабайты, обязана
    уметь сначала показать, что именно, — иначе первый её запуск на боевом
    диске и есть проверка.
    """
    return _walk(Path(root), keep=set(keep), now=datetime.now(UTC), grace=PUBLISHING_GRACE)


def _still_publishing(run: Path, now: datetime, grace: timedelta) -> bool:
    """Прогон без указателя, у которого публикация ещё может дойти до конца.

    Признак — манифест без `published`. Опубликованный прогон, с которого сняли
    указатель, — это уже прошлое, и слои ему не нужны; а вот тот, у кого
    манифест лежит с `published: false` десять минут, прямо сейчас собирает
    производные слои.
    """
    manifest = run / MANIFEST_NAME
    if not manifest.is_file():
        # Ни манифеста, ни указателя: либо одна отметка о пропуске, либо
        # каталог, из которого ротация уже всё унесла. И то и другое — не
        # публикация.
        return False
    if read_manifest(manifest).get("published"):
        return False
    return now - datetime.fromtimestamp(manifest.stat().st_mtime, UTC) < grace


def _walk(root: Path, *, keep: set[str], now: datetime, grace: timedelta) -> tuple[Path, ...]:
    """Что в `runs/` лишнее. Обход и решение — здесь, удаление — у вызвавшего:
    `--dry-run` обязан ходить по тому же коду, иначе он показывает не то."""
    runs = root / RUNS_DIR
    if not runs.is_dir():
        return ()

    current = current_run(root)
    reduced = previous_run(root)
    # Указатель смотрит на слой внутри прогона, а не на сам прогон: каталог
    # прошлого прогона — родитель этого слоя.
    previous = reduced.parent if reduced is not None else None

    doomed_paths: list[Path] = []
    for run in sorted(runs.iterdir()):
        if not run.is_dir() or run.name in keep:
            continue
        if current is not None and run.samefile(current):
            continue
        if _still_publishing(run, now, grace):
            continue
        survivor = PREVIOUS_LAYER if previous is not None and run.samefile(previous) else None
        doomed_paths.extend(
            item
            for item in sorted(run.iterdir())
            if item.name not in KEPT_FILES and item.name != survivor
        )
    return tuple(doomed_paths)


if __name__ == "__main__":
    raise SystemExit(main())
