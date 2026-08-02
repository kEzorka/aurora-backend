"""API читает хранилище и кэш. Всё остальное для него не существует.

docs/ARCHITECTURE.md: «Единственное направление зависимостей: API → Store/Cache».
Правило, которое ничем не проверяется, нарушается на третьей неделе.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "api": {"adapters", "pipeline", "torch", "aurora", "cfgrib", "cdsapi"},
    # Адаптер знает только про источник и канон. Куда положить прочитанное,
    # решает pipeline; знай об этом адаптер — и правило приёма поехало бы
    # вслед за форматом хранилища.
    "adapters": {"api", "storage", "cache", "pipeline", "torch", "aurora"},
    "storage": {"api", "adapters", "torch", "aurora"},
    "cache": {"api", "pipeline", "torch", "aurora"},
    "contracts": {"api", "storage", "cache", "adapters", "pipeline", "xarray", "torch"},
    "validators": {"api", "storage", "cache", "adapters", "pipeline", "torch"},
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_does_not_import_what_it_must_not(package: str) -> None:
    banned = FORBIDDEN[package]
    offences: list[str] = []
    for source in (ROOT / package).rglob("*.py"):
        for imported in sorted(_imported_roots(source) & banned):
            offences.append(f"{source.relative_to(ROOT)} imports {imported}")
    assert not offences, "\n".join(offences)


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_every_listed_package_actually_exists(package: str) -> None:
    """Иначе проверка выше проходит потому, что искать было негде."""
    assert (ROOT / package / "__init__.py").is_file()
