"""Результат проверки. Отказ обязан называть поле, значение и ожидание.

docs/DATA_CONTRACT.md §4: «Формулировка "validation failed" без деталей
не принимается на ревью». Поэтому конструктор отказа принимает field, got
и expected по отдельности и собирает сообщение сам — забыть их нельзя,
даже если очень хочется.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

LEVELS = ("structure", "semantics", "physics", "sanity")


@dataclass(frozen=True, slots=True)
class Check:
    """Одна проверка: что смотрели, на каком уровне, чем кончилось."""

    name: str
    level: str
    passed: bool
    message: str = ""
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level: got {self.level!r}, expected one of {LEVELS}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "passed": self.passed,
            "message": self.message,
            "details": jsonable(self.details),
        }


def ok(name: str, level: str, message: str = "", **details: Any) -> Check:
    return Check(name=name, level=level, passed=True, message=message, details=details)


def fail(name: str, level: str, field: str, got: Any, expected: Any) -> Check:
    """Единственный способ создать отказ — и он требует все три части."""
    return Check(
        name=name,
        level=level,
        passed=False,
        message=f"{field}: got {got!r}, expected {expected!r}",
        details={"field": field, "got": got, "expected": expected},
    )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks]}


def jsonable(value: Any) -> Any:
    """numpy-скаляры json не сериализует, а отказ обязан долетать до файла."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
