"""Адреса Open Data, отбор по индексу и загрузка среза (BACKLOG 1.7, 1.10).

Сети нет: транспорт подставляется, как и в `test_fetch`. Индекс синтетический,
но формой — настоящий, и адреса проверяются буквой в букву: опечатка в них
даёт 404 в бою и ничего в тесте, где адрес никто не читает.
"""

import hashlib
import json
from pathlib import Path

import pytest

from adapters.errors import AdapterError
from adapters.fetch import Response, checksum
from adapters.opendata import data_url, download, index_keys, index_url
from adapters.plan import plan
from contracts import canon

MIDNIGHT = "2026-08-01T00:00:00Z"


def _index(*rows: dict[str, object]) -> str:
    return "\n".join(json.dumps(row) for row in rows)


class Server:
    """Транспорт-заглушка: индекс на `.index`, тела на `Range`."""

    def __init__(self, index: str) -> None:
        self.index = index
        self.asked: list[tuple[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> Response:
        self.asked.append((url, headers.get("Range", "")))
        if url.endswith(".index"):
            return Response(200, self.index.encode())
        start, _, end = headers["Range"].removeprefix("bytes=").partition("-")
        return Response(206, b"x" * (int(end) - int(start) + 1))


def _surface(request_names: tuple[str, ...]) -> str:
    """Индекс, где каждое приземное поле занимает по сто байт подряд."""
    return _index(
        *[
            {"param": name, "levtype": "sfc", "_offset": position * 100, "_length": 100}
            for position, name in enumerate(request_names)
        ]
    )


def test_the_file_and_its_index_are_neighbours() -> None:
    """Индекс лежит рядом с данными и отличается расширением. Опечатка здесь —
    это 404 в семь утра, когда до дедлайна отката полтора часа."""
    (request,) = plan(MIDNIGHT, ["2t"])

    assert data_url(request) == (
        "https://data.ecmwf.int/forecasts/20260801/00z/ifs/0p25/oper/"
        "20260801000000-0h-oper-fc.grib2"
    )
    assert index_url(request) == data_url(request).removesuffix(".grib2") + ".index"


def test_a_field_from_another_stream_goes_to_another_directory() -> None:
    """Облачности нет в `ifs/oper` ни в каком виде: она в `aifs-single`, и
    поток — это каталог в адресе (`adapters.ecmwf.STREAMS`)."""
    (request,) = plan(MIDNIGHT, ["lcc"])

    assert "/aifs-single/0p25/oper/" in data_url(request)


def test_the_hour_of_the_run_is_in_the_path_and_in_the_file_name() -> None:
    (request,) = plan("2026-08-01T12:00:00Z", ["2t"])

    assert "/20260801/12z/" in data_url(request)
    assert data_url(request).endswith("/20260801120000-0h-oper-fc.grib2")


def test_a_pressure_level_field_becomes_thirteen_messages() -> None:
    """В индексе `t` — это тринадцать сообщений, по одному на уровень, и
    просить его без уровня значит не найти ни одного."""
    keys = index_keys(["t"])

    assert len(keys) == len(canon.PRESSURE_LEVELS)
    assert ("t", "850") in keys
    assert ("t", "") not in keys


def test_soil_is_asked_for_by_its_provider_name_and_top_layer() -> None:
    """`sot`/`vsw` лежат четырьмя слоями под одним именем. Без `levelist=1`
    приедет слой, которого Aurora не просила, под именем верхнего."""
    assert index_keys(["stl1", "swvl1"]) == (("sot", "1"), ("vsw", "1"))


def test_the_static_geopotential_is_the_one_without_a_level() -> None:
    """В индексе он зовётся `z`, как и поле на уровнях давления. Различает их
    уровень, и совпадение имени тут ничего не значит."""
    assert index_keys(["z_surf"]) == (("z", ""),)


def test_a_surface_field_keeps_its_canonical_name() -> None:
    """Семнадцать приземных полей канона — это и есть `shortName` ECMWF."""
    assert index_keys(["2t", "msl", "10u"]) == (("2t", ""), ("msl", ""), ("10u", ""))


def test_the_index_comes_first_and_only_the_wanted_bytes_after(tmp_path: Path) -> None:
    """Файл прогона — 200+ МБ, из которых нужны считанные поля. Целиком он не
    качается никогда: сначала индекс, потом диапазоны по нему."""
    (request,) = plan(MIDNIGHT, ["2t", "msl"])
    server = Server(_surface(("2t", "msl", "sp")))

    written = download(request, tmp_path / "slice.grib2", transport=server, sleep=lambda _: None)

    assert written.stat().st_size == 200  # два сообщения по сто байт, без `sp`
    assert [url.rsplit("/", 1)[-1] for url, _ in server.asked] == [
        "20260801000000-0h-oper-fc.index",
        "20260801000000-0h-oper-fc.grib2",
    ]
    assert server.asked[1][1] == "bytes=0-199"


def test_a_field_missing_from_the_index_stops_the_download(tmp_path: Path) -> None:
    """Файл, выложенный наполовину, — обычное дело в семь утра. Срез без
    переменной заметят на сборке батча, а причину будут искать не здесь."""
    (request,) = plan(MIDNIGHT, ["2t", "msl"])
    path = tmp_path / "slice.grib2"

    with pytest.raises(AdapterError, match="missing"):
        download(request, path, transport=Server(_surface(("2t",))), sleep=lambda _: None)

    assert not path.exists()


def test_era5t_is_refused_rather_than_downloaded_empty(tmp_path: Path) -> None:
    """CDS — это очередь и заявки, другой протокол целиком, и `ci` качает
    `adapters.era5_grid`. Адрес Open Data для `ci` существует и отдаёт 404."""
    (request,) = plan(MIDNIGHT, ["ci"])

    with pytest.raises(AdapterError, match="source"):
        download(request, tmp_path / "ci.grib2", transport=Server(""), sleep=lambda _: None)


def test_an_index_that_is_a_page_of_html_is_retried_then_refused(tmp_path: Path) -> None:
    """503 на индексе — это пропущенный прогон, если не повторить."""
    (request,) = plan(MIDNIGHT, ["2t"])
    slept: list[float] = []

    class Broken:
        def __call__(self, url: str, headers: dict[str, str]) -> Response:
            return Response(503, b"")

    with pytest.raises(AdapterError, match="whole file"):
        download(
            request,
            tmp_path / "slice.grib2",
            transport=Broken(),
            delays=(1.0, 2.0),
            sleep=slept.append,
        )

    assert slept == [1.0, 2.0]


def test_the_checksum_is_the_one_the_manifest_asks_for(tmp_path: Path) -> None:
    """`sha256:<hex>` и ничего другого: `storage.manifest` отвергает остальное,
    а сумма нужна на каждую загрузку отдельно (`pipeline.inputs`)."""
    path = tmp_path / "slice.grib2"
    path.write_bytes(b"GRIB" * 1000)

    assert checksum(path) == "sha256:" + hashlib.sha256(b"GRIB" * 1000).hexdigest()
    # Считается блоками: файл прогона — сотни мегабайт, и в памяти ему делать
    # нечего. Сумма от этого меняться не должна.
    assert checksum(path, block=7) == checksum(path)
