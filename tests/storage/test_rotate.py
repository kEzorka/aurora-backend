"""Ротация прогонов и чистка scratch (BACKLOG 2.6).

Критерий приёмки — «диск не переполняется при 10 прогонах подряд», и проверяется
он не объёмом (сетка здесь 6×8), а инвариантом: после ротации на диске остаются
ровно те данные, на которые смотрят указатели. Всё, что сверх, — 17.4 ГБ каждые
шесть часов.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from storage.manifest import SKIP_NAME, build_skip, read_manifest, write_skip
from storage.publish import publish_run, run_path, stage_path
from storage.read import run_log
from storage.rotate import (
    SCRATCH_QUOTA_BYTES,
    doomed,
    over_quota,
    rotate,
    scratch_bytes,
    sweep_scratch,
)
from storage.rotate import main as rotate_main
from tests.storage.test_publish import _manifest, _stage

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _publish(root: Path, run_id: str, value: float = 288.0) -> Path:
    _stage(root, run_id, value)
    return publish_run(root, run_id, manifest=_manifest(run_id))


def test_the_current_run_survives_rotation_whole(tmp_path: Path) -> None:
    final = _publish(tmp_path, "2026-08-01T00Z")

    assert rotate(tmp_path, now=NOW) == ()
    assert (final / "coarse").is_dir()
    assert (final / "hourly").is_dir()
    assert (final / "points").is_dir()


def test_the_previous_run_keeps_only_what_the_pointer_needs(tmp_path: Path) -> None:
    """15.0 ГБ шестичасового слоя в двух экземплярах — это 30 ГБ при ядре в 40
    (docs/STORAGE.md §2). Указателю нужна свёртка, и только она."""
    old = _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)

    rotate(tmp_path, now=NOW)

    assert (old / "previous").is_dir()
    assert not (old / "coarse").exists()
    assert not (old / "hourly").exists()
    assert not (old / "points").exists()
    # Указатель после ротации по-прежнему ведёт на данные, а не в пустоту.
    assert (tmp_path / "forecast" / "previous" / "2t").exists()


def test_a_run_two_cycles_back_keeps_no_layers_at_all(tmp_path: Path) -> None:
    oldest = _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)
    _publish(tmp_path, "2026-08-01T12Z", value=292.0)

    rotate(tmp_path, now=NOW)

    assert [item.name for item in sorted(oldest.iterdir())] == ["manifest.json", "validation.json"]


def test_the_run_log_still_shows_a_rotated_run(tmp_path: Path) -> None:
    """Каталог не сносится целиком намеренно: журнал прогонов перечисляет
    каталоги, и снесённый — это прогон, которого как будто не было."""
    _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)
    _publish(tmp_path, "2026-08-01T12Z", value=292.0)

    rotate(tmp_path, now=NOW)
    log = run_log(tmp_path)

    assert [state.run_id for state in log] == [
        "2026-08-01T00Z",
        "2026-08-01T06Z",
        "2026-08-01T12Z",
    ]
    assert {state.state for state in log} == {"published"}


def test_a_skip_mark_is_not_swept_away(tmp_path: Path) -> None:
    """Пропуск, стёртый ротацией, — ровно тот случай, ради которого журнал и
    писали (docs/PIPELINE.md §3.5)."""
    missed = run_path(tmp_path, "2026-08-01T06Z")
    missed.mkdir(parents=True)
    write_skip(
        missed / SKIP_NAME,
        build_skip(
            "forecast/2026-08-01T06Z",
            init_time="2026-08-01T06:00:00Z",
            reason="no_input",
            waited_for=("ifs/0p25/oper",),
            deadline="2026-08-01T15:30:00Z",
            attempts=5,
            created_at=NOW,
        ),
    )
    _publish(tmp_path, "2026-08-01T12Z")

    rotate(tmp_path, now=NOW)

    assert (missed / SKIP_NAME).is_file()
    assert [state.state for state in run_log(tmp_path) if state.run_id == "2026-08-01T06Z"] == [
        "skipped"
    ]


def test_a_kept_run_is_not_touched(tmp_path: Path) -> None:
    """Через `keep` архив прогонов (BACKLOG 4.4) спасает то, что отложил:
    указателя на архив нет, а без указателя всё остальное — мусор."""
    archived = _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)
    _publish(tmp_path, "2026-08-01T12Z", value=292.0)

    rotate(tmp_path, keep=["2026-08-01T00Z"], now=NOW)

    assert (archived / "coarse").is_dir()


def test_a_publication_in_flight_is_not_stripped_from_under_itself(tmp_path: Path) -> None:
    """Между переездом и переставленным указателем публикация собирает
    производные слои, и это минуты, в которые прогон выглядит как мусор:
    указателя нет, `published` в манифесте `false` (`storage.publish`)."""
    _publish(tmp_path, "2026-08-01T00Z")
    # Прогон переехал в `runs/`, манифест лежит с `published: false`, слои на
    # месте, указатель ещё не переставлен — снимок ровно этого момента.
    landing = run_path(tmp_path, "2026-08-01T06Z")
    _stage(tmp_path, "2026-08-01T06Z", 290.0)
    stage_path(tmp_path, "2026-08-01T06Z").rename(landing)
    manifest = landing / "manifest.json"
    manifest.write_text('{"published": false, "inputs": []}', encoding="utf-8")
    # Возраст берётся у манифеста, и в тесте он задаётся явно: часы на машине
    # к сроку прогона отношения не имеют.
    written = (NOW - timedelta(minutes=5)).timestamp()
    os.utime(manifest, (written, written))

    assert rotate(tmp_path, now=NOW) == ()
    assert (landing / "coarse").is_dir()

    # А через шесть часов публикация уже не доедет: это мусор, и он занимает
    # те же 17.4 ГБ, что живой прогон.
    assert rotate(tmp_path, now=NOW + timedelta(hours=7)) != ()
    assert not (landing / "coarse").exists()


def test_rotation_of_an_empty_store_is_not_an_error(tmp_path: Path) -> None:
    assert rotate(tmp_path, now=NOW) == ()
    assert sweep_scratch(tmp_path, now=NOW) == ()
    assert scratch_bytes(tmp_path) == 0


def test_scratch_is_swept_by_age_not_by_pointers(tmp_path: Path) -> None:
    """Указателей на scratch нет ни у кого: каталог, переживший сутки, остался
    от публикации, которая не дошла до переезда (docs/STORAGE.md §6)."""
    fresh = _stage(tmp_path, "2026-08-01T06Z")
    stale = _stage(tmp_path, "2026-07-30T00Z")
    old = (NOW - timedelta(days=2)).timestamp()
    os.utime(stale, (old, old))

    swept = sweep_scratch(tmp_path, now=NOW)

    assert swept == (stale,)
    assert not stale.exists()
    assert fresh.is_dir()


def test_the_scratch_quota_is_a_number_to_compare_with(tmp_path: Path) -> None:
    """Резерв — не «оставить свободным», а величина, с которой сверяются перед
    публикацией (docs/STORAGE.md §6). Что делать с переполнением, решает
    конвейер: хранилище не знает, который час."""
    _stage(tmp_path, "2026-08-01T06Z")

    assert scratch_bytes(tmp_path) > 0
    assert not over_quota(tmp_path)
    assert over_quota(tmp_path, quota=0)
    assert SCRATCH_QUOTA_BYTES == 30 * (1 << 30)


def test_a_dry_run_shows_the_same_paths_it_would_delete(tmp_path: Path) -> None:
    """Команда, которая сносит гигабайты, обязана уметь сначала показать, что
    именно: иначе первый её запуск на боевом диске и есть проверка."""
    _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)

    preview = doomed(tmp_path)

    assert preview  # свёрнутому прогону слои уже не нужны
    assert all(path.exists() for path in preview)
    assert rotate(tmp_path, now=NOW) == preview


def test_the_command_line_reports_what_it_removed(tmp_path: Path) -> None:
    _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)

    assert rotate_main([str(tmp_path), "--scratch"]) == 0
    assert not (tmp_path / "runs" / "2026-08-01T00Z" / "coarse").exists()


def test_the_command_line_fails_when_scratch_is_over_quota(tmp_path: Path) -> None:
    """Код возврата важен: переполненный scratch — это следующий прогон,
    который упадёт на середине перекладки (docs/STORAGE.md §6)."""
    _stage(tmp_path, "2026-08-01T06Z")
    huge = tmp_path / "scratch" / "2026-08-01T06Z" / "big.bin"
    huge.write_bytes(b"0" * 4096)

    assert rotate_main([str(tmp_path)]) == 0  # резерв 30 ГБ, до него далеко
    assert rotate_main([str(tmp_path), "--quota", "0"]) == 1
    assert scratch_bytes(tmp_path) >= 4096


def test_rotation_leaves_the_manifest_readable(tmp_path: Path) -> None:
    """Слоёв нет, а документ остаётся документом: иначе журнал прогонов читает
    полуфайл и падает на прогоне, который просто состарился."""
    oldest = _publish(tmp_path, "2026-08-01T00Z")
    _publish(tmp_path, "2026-08-01T06Z", value=290.0)
    _publish(tmp_path, "2026-08-01T12Z", value=292.0)

    rotate(tmp_path, now=NOW)

    assert read_manifest(oldest / "manifest.json")["published"] is True
