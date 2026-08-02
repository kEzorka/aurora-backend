"""Вторая половина приёмки BACKLOG 1.10: «в манифесте видно, какое поле из
какого потока»."""

import json
from pathlib import Path

import pytest

from adapters.plan import ERA5T_STREAM, Request, plan
from pipeline.inputs import manifest_inputs
from storage.manifest import Model, build_manifest, read_manifest, write_manifest

NOON = "2026-08-01T00:00:00Z"


def _checksums() -> dict[Request, str]:
    return {request: f"sha256:{index:064d}" for index, request in enumerate(plan(NOON))}


def test_the_manifest_says_which_field_came_from_which_stream(tmp_path: Path) -> None:
    """Через полгода вопрос будет звучать так: почему облачность в этом прогоне
    другая? Ответ должен читаться из манифеста, а не восстанавливаться по коду
    той версии, которая его писала."""
    requests = plan(NOON)
    manifest = build_manifest(
        f"forecast/{NOON}",
        inputs=manifest_inputs(requests, _checksums()),
        model=Model("aurora", "aurora-0.25-v1.5", "9f2c1ab"),
        steps=40,
        timings_sec={"ingest": 480, "normalize": 190, "inference": 260, "write": 520},
    )
    write_manifest(tmp_path / "manifest.json", manifest)

    entries = read_manifest(tmp_path / "manifest.json")["inputs"]
    streams = {name: entry["stream"] for entry in entries for name in entry["fields"]}
    assert streams["hcc"] == "aifs-single/0p25/oper"
    assert streams["2t"] == "ifs/0p25/oper"
    assert streams["ci"] == ERA5T_STREAM


def test_the_five_day_old_field_is_visible_as_such(tmp_path: Path) -> None:
    """Единственное, ради чего вход разбит по полям. Вход на 27 июля в прогоне
    за 1 августа без списка полей читается как «весь срез пятидневный»."""
    inputs = manifest_inputs(plan(NOON), _checksums())
    stale = [entry for entry in inputs if entry.valid_time != NOON]

    assert [entry.fields for entry in stale] == [("ci",)]
    assert stale[0].valid_time == "2026-07-27T00:00:00Z"


def test_every_download_carries_its_own_checksum() -> None:
    """Одна сумма на весь срез не сказала бы, какой из трёх файлов приехал
    битым, а сумма только у основного потока узаконила бы облачность и лёд
    вовсе без проверки."""
    inputs = manifest_inputs(plan(NOON), _checksums())

    assert len({entry.checksum for entry in inputs}) == len(inputs)


def test_a_download_without_a_checksum_is_refused() -> None:
    requests = plan(NOON)
    checksums = {request: "sha256:" + "a" * 64 for request in requests[:-1]}

    with pytest.raises(ValueError, match=ERA5T_STREAM):
        manifest_inputs(requests, checksums)


def test_the_manifest_is_json_and_not_python(tmp_path: Path) -> None:
    """`fields` — кортеж, а кортежей в JSON нет: писать манифест `repr`-ом
    значит записать `('ci',)` и не прочитать его ничем, кроме Python."""
    manifest = build_manifest(
        f"forecast/{NOON}",
        inputs=manifest_inputs(plan(NOON), _checksums()),
        model=Model("aurora", "aurora-0.25-v1.5", "9f2c1ab"),
        steps=40,
        timings_sec={"ingest": 1, "normalize": 1, "inference": 1, "write": 1},
    )

    assert json.loads(json.dumps(manifest))["inputs"][0]["fields"][0] == "2t"
