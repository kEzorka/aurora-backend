"""Форма хранилища: где лежит слой, какими чанками и сколько это весит.

Раскладка под два шага по времени — не оптимизация, а форма (`ADDENDUM-01`,
шапка). Часовой слой это отдельный набор из восьми переменных, а не мелкий шаг
шестичасового: все 90 полей на 72 срока весят 27 ГБ, то есть полтора бюджета
прогона. Поэтому слой знает свой набор полей (`canon.LAYERS`), а запись —
свой путь и свои чанки, и оба зафиксированы здесь до первой записи.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple

import xarray as xr
from zarr.codecs import BloscCodec

from contracts import canon

#: float32 — тип прогноза (docs/STORAGE.md §4). float16 истории считается
#: отдельно: он относится к слоям, которых в бюджете прогона нет.
BYTES_PER_VALUE: Final = 4

#: 721 × 1440 = 1 038 240 точек, одно поле — 4.15 МБ.
GRID_POINTS: Final = canon.GRID_SHAPE[0] * canon.GRID_SHAPE[1]

#: Бюджет прогона (`ADDENDUM-01` §3): 15.0 ГБ шестичасового слоя плюс 2.4 ГБ
#: часового. Гигабайты десятичные — как во всех расчётах docs/STORAGE.md.
RUN_BUDGET_BYTES: Final = 18 * 10**9

#: Каталоги слоёв (docs/STORAGE.md §2). Имя каталога — интерфейс: по нему
#: `validators.cli` понимает, каким набором полей проверять срез.
LAYER_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "coarse": "forecast/current/coarse",
        "hourly": "forecast/current/hourly",
        "points": "forecast/current/points",
        "previous": "forecast/previous",
        "analysis": "analysis/recent",
    }
)

#: Сколько сроков в году при шестичасовом шаге: 365 × 4. Размер чанка
#: раскладки B по времени — ряд в точке читается ровно годами.
STEPS_PER_YEAR: Final = 1460


class Chunking(NamedTuple):
    """Чанк и шард по осям `(time, level, lat, lon)`, в элементах.

    Чанк равен тому, что читает запрос, — не «целевому размеру». Раздувание
    чтения = прочитано / нужно, и на карте, склеенной из двенадцати сроков,
    запрос одного срока распаковывает двенадцать. Целевые 20–200 МБ относятся
    к шарду: он единица файла, и там размер решает не чтение, а то, что
    файловая система умирает от миллионов мелких файлов (docs/STORAGE.md §3).
    """

    name: str
    chunk: tuple[int, int, int, int]
    shard: tuple[int, int, int, int]

    @property
    def chunk_bytes(self) -> int:
        return _product(self.chunk) * BYTES_PER_VALUE

    @property
    def shard_bytes(self) -> int:
        return _product(self.shard) * BYTES_PER_VALUE

    @property
    def chunks_per_shard(self) -> int:
        return _product(self.shard) // _product(self.chunk)


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


#: A — карты: чанк это вся карта на один момент (4.15 МБ), шард — 12 сроков
#: (50 МБ), то есть трое суток шестичасового прогноза в одном файле.
LAYOUT_A: Final = Chunking(
    name="maps",
    chunk=(1, 1, canon.GRID_SHAPE[0], canon.GRID_SHAPE[1]),
    shard=(12, 1, canon.GRID_SHAPE[0], canon.GRID_SHAPE[1]),
)

#: B — ряды: чанк это год в квадрате 4×4 точки (93 КБ), шард — 16×16 чанков
#: (24 МБ). Без шардинга год одной переменной это 181 × 360 = 65 160 чанков;
#: с ним — 255 файлов.
LAYOUT_B: Final = Chunking(
    name="series",
    chunk=(STEPS_PER_YEAR, 1, 4, 4),
    shard=(STEPS_PER_YEAR, 1, 64, 64),
)


#: Какому слою какая раскладка. Не аргумент вызова: слой и его форма — одно
#: решение, и записать `coarse` рядами (или `points` картами) значит получить
#: слой, который на свой запрос отвечает в сто раз дольше, ничем себя не выдав.
#:
#: `hourly` лежит рядами, а не картами, и второго экземпляра у него нет:
#: карту часового слоя не читает никто — `/v1/forecast/grid` округляет `time`
#: до ближайшего шестичасового срока (docs/API_CONTRACT.md §2). Ряд в точке из
#: него читают, и раскладка карт стоила бы там 80–300 мс.
_LAYOUT_BY_LAYER: Final[Mapping[str, Chunking]] = MappingProxyType(
    {
        "coarse": LAYOUT_A,
        "hourly": LAYOUT_B,
        "points": LAYOUT_B,
        "previous": LAYOUT_A,
        "analysis": LAYOUT_A,
    }
)


def layout_for(layer: str) -> Chunking:
    """Раскладка слоя по имени. Незнакомое имя — карты: неизвестный слой чаще
    всего карта, и ошибиться в эту сторону дешевле, чем разложить рядами то,
    что читают целиком."""
    return _LAYOUT_BY_LAYER.get(layer, LAYOUT_A)


#: Оси раскладки по именам канона. Порядок тот же, что в `Chunking.chunk`.
_AXIS: Final[Mapping[str, int]] = MappingProxyType({"time": 0, "level": 1, "lat": 2, "lon": 3})


def _codec() -> BloscCodec:
    """`zstd` уровня 3 с shuffle (docs/STORAGE.md §4), а не дефолт библиотеки.

    Shuffle переставляет байты одного разряда рядом: у соседних значений
    метеополя старшие байты совпадают, и после перестановки они сжимаются
    как повторы. На float32-полях это заметно больше, чем разница уровней.
    """
    return BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")


def encoding_for(ds: xr.Dataset, layout: Chunking) -> dict[str, dict[str, object]]:
    """Кодировка для `to_zarr`: чанк, шард и кодек на каждое поле.

    Чанк обрезается по массиву: чанк больше массива Zarr не примет, а слои
    бывают меньше сетки (месячные средние, тесты). Шард после обрезки
    пересчитывается через то же отношение «чанков в шарде», иначе он
    перестаёт делиться на чанк нацело — и запись падает посреди прогона.
    """
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in ds.data_vars.items():
        chunks: list[int] = []
        shards: list[int] = []
        for dim, size in zip(variable.dims, variable.shape, strict=True):
            axis = _AXIS.get(str(dim))
            if axis is None:
                chunk, per_shard = size, 1
            else:
                chunk = min(layout.chunk[axis], size)
                per_shard = layout.shard[axis] // layout.chunk[axis]
            whole = -(-size // chunk) * chunk  # округление вверх до целых чанков
            chunks.append(chunk)
            shards.append(min(chunk * per_shard, whole))
        encoding[str(name)] = {
            "chunks": tuple(chunks),
            "shards": tuple(shards),
            "compressors": [_codec()],
        }
    return encoding


def read_amplification(layout: Chunking, request: tuple[int, int, int, int]) -> float:
    """Прочитано / нужно для выборки формы `request` по осям `(time, level, lat, lon)`.

    Метрика раскладки (docs/STORAGE.md §3). Чанк распаковывается целиком, и
    выборка тянет все чанки, которых коснулась: ряд в точке по раскладке карт
    поднимает 1460 карт ради 1460 чисел — раздувание ×10⁶.
    """
    return read_values(layout, request) / _product(request)


def read_values(layout: Chunking, request: tuple[int, int, int, int]) -> int:
    """Сколько логических значений распакуют чанки для выборки ``request``.

    Отдельное целое число нужно отчёту 2.7: умножение коэффициента с плавающей
    точкой обратно на размер запроса способно дать байт меньше из-за округления.
    Нулевая ось — ошибка вызывающего, а не бесплатный запрос.
    """
    if len(request) != 4 or any(wanted <= 0 for wanted in request):
        raise ValueError(f"request: expected four positive axes, got {request}")
    touched = 1
    for axis, wanted in enumerate(request):
        chunk = layout.chunk[axis]
        touched *= -(-wanted // chunk) * chunk
    return touched


def layer_path(root: str | Path, layer: str) -> Path:
    """Путь слоя внутри хранилища. Неизвестное имя — ошибка, а не новый каталог."""
    if layer not in LAYER_PATHS:
        raise ValueError(f"layer: got {layer!r}, expected one of {sorted(LAYER_PATHS)}")
    return Path(root) / LAYER_PATHS[layer]


def layer_bytes(layer: canon.Layer) -> int:
    """Вес слоя целиком: поля × точки сетки × 4 байта."""
    return layer.fields * GRID_POINTS * BYTES_PER_VALUE


def run_bytes() -> int:
    """Вес одного прогона: шестичасовой слой плюс часовой — то, что выдала
    модель (`ADDENDUM-01` §3: 15.0 + 2.4 = 17.4 ГБ).

    `points` сюда не входит, как не входит `previous`: это не новые данные, а
    копия части `coarse` в другой раскладке. Закреплённое на диске считает
    `pinned_bytes`.
    """
    return layer_bytes(canon.LAYERS["coarse"]) + layer_bytes(canon.LAYERS["hourly"])


def pinned_bytes() -> int:
    """Сколько прогноз держит на диске: прогон, его точечная копия и свёртка
    прошлого прогона — 17.4 + 1.33 + 1.33 = 20.1 ГБ из ядра в 40 ГБ."""
    return run_bytes() + layer_bytes(canon.LAYERS["points"]) + layer_bytes(canon.LAYERS["previous"])


def select_layer(ds: xr.Dataset, layer: canon.Layer) -> xr.Dataset:
    """Ровно поля слоя: лишние отсечь, на нехватке — упасть.

    Асимметрия намеренная. Лишние поля у адаптера законны (он отдаёт и `tp`,
    и `tp_1h`), и записывать их в слой нельзя: часовой слой лежит одним
    массивом по оси переменных, девятая переменная меняет раскладку. А поле,
    которого нет, — это не «слой поменьше»: читатель увидит не пропуск, а
    прогноз, в котором этой величины не было.
    """
    wanted = (*layer.surface_vars, *layer.atmos_vars)
    missing = [name for name in wanted if name not in ds.data_vars]
    if missing:
        raise ValueError(f"layer {layer.name}: полей нет: {', '.join(missing)}")
    return ds[list(wanted)]
