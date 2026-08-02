"""План загрузки → входы манифеста.

Одно место, где план приёма (`adapters.plan`) встречается со схемой манифеста
(`storage.manifest`). Отдельным модулем оно живёт не для красоты: адаптеру
запрещено знать про хранилище, а хранилищу — про адаптеры
(`tests/test_boundaries.py`), и написать эти пять строк больше негде.

Что здесь проверяется по существу: контрольная сумма нужна на **каждую**
загрузку. Одна сумма на весь срез не сказала бы, какой из трёх файлов приехал
битым, а сумма только у основного потока молча узаконила бы облачность и лёд
без проверки вообще.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from adapters.plan import Request
from storage.manifest import Input


def manifest_inputs(
    requests: Sequence[Request], checksums: Mapping[Request, str]
) -> tuple[Input, ...]:
    """Входы манифеста в порядке плана.

    Порядок сохраняется, чтобы два одинаковых прогона давали одинаковый
    манифест: перетасованные входы читаются как разные прогоны при `diff`.
    """
    missing = [request for request in requests if request not in checksums]
    if missing:
        raise ValueError(
            "checksums: нет суммы для "
            + ", ".join(f"{request.stream}@{request.valid_time}" for request in missing)
        )
    return tuple(
        Input(
            source=request.source,
            valid_time=request.valid_time,
            checksum=checksums[request],
            stream=request.stream,
            fields=request.names,
        )
        for request in requests
    )
