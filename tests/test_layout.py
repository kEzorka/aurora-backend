"""Структура репозитория и точки входа — из docs/README.md.

Проверяется тестом, а не договорённостью: дерево и цели Makefile — это контракт
между людьми, и он расползается ровно так же, как код.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "docs",
    "contracts",
    "adapters",
    "pipeline",
    "validators",
    "storage",
    "cache",
    "api",
    "tests/fixtures",
    "artifacts",
]

MAKE_TARGETS = [
    "setup",
    "ingest-analysis",
    "ingest-era5",
    "forecast",
    "validate",
    "test",
    "serve",
    "cache-report",
]


@pytest.mark.parametrize("name", REQUIRED_DIRS)
def test_required_directory_exists(name: str) -> None:
    assert (ROOT / name).is_dir(), f"{name}/ отсутствует, см. docs/README.md"


@pytest.mark.parametrize("target", MAKE_TARGETS)
def test_makefile_declares_target(target: str) -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert f"\n{target}:" in makefile, f"цель {target} не объявлена в Makefile"


def test_service_requirements_have_no_torch() -> None:
    """docs/SETUP.md §4: два разных окружения, не пытаться свести в одно."""
    text = (ROOT / "requirements" / "service.txt").read_text(encoding="utf-8")
    declared = [
        line.split("#")[0].strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any(line.startswith("torch") for line in declared), (
        "torch не место в окружении сервиса (docs/SETUP.md §4)"
    )


def test_inference_requirements_exist_separately() -> None:
    text = (ROOT / "requirements" / "inference.txt").read_text(encoding="utf-8")
    assert "microsoft-aurora" in text
