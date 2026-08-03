"""Проверка окружения без вывода секретов и без неявных сетевых запросов.

``make setup`` отвечает на вопрос «можно ли здесь разрабатывать и запускать
service-контур». GPU и токен CDS на ноутбуке отмечаются предупреждением, а не
маскируются зелёным. Для production-приёмки секреты делаются обязательными
флагом ``--strict-secrets``; inference-профиль всегда требует Torch, Aurora,
CUDA и HF_TOKEN.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final

OK: Final = "ok"
WARN: Final = "warn"
FAIL: Final = "fail"
ROLES: Final = ("service", "inference", "all")
SERVICE_PACKAGES: Final = ("numpy", "xarray", "zarr", "fastapi", "uvicorn")
OPTIONAL_SERVICE_PACKAGES: Final = ("cfgrib", "eccodes", "gcsfs", "cdsapi")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


VersionLookup = Callable[[str], str | None]
ModuleLookup = Callable[[str], bool]
CudaProbe = Callable[[], tuple[bool, str]]


def inspect_environment(
    role: str = "service",
    *,
    strict_secrets: bool = False,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    python: tuple[int, int] | None = None,
    version: VersionLookup | None = None,
    module: ModuleLookup | None = None,
    cuda: CudaProbe | None = None,
) -> tuple[Check, ...]:
    """Вернуть все проверки; зависимости подставляются в тестах без импорта GPU."""
    if role not in ROLES:
        raise ValueError(f"role: got {role!r}, expected one of {ROLES}")
    environ = os.environ if env is None else env
    base = Path.home() if home is None else home
    py = (sys.version_info.major, sys.version_info.minor) if python is None else python
    lookup = _version if version is None else version
    available = _module if module is None else module
    probe_cuda = _cuda if cuda is None else cuda
    checks: list[Check] = []

    supported = (3, 11) <= py < (3, 13)
    checks.append(Check("python", OK if supported else FAIL, f"{py[0]}.{py[1]} (нужно 3.11–3.12)"))

    if role in ("service", "all"):
        for package in SERVICE_PACKAGES:
            found = lookup(package)
            checks.append(
                Check(f"package:{package}", OK if found else FAIL, found or "не установлен")
            )
        zarr = lookup("zarr")
        zarr3 = zarr is not None and zarr.split(".", 1)[0] == "3"
        checks.append(Check("zarr-format", OK if zarr3 else FAIL, zarr or "не установлен"))
        for package in OPTIONAL_SERVICE_PACKAGES:
            found = lookup(package)
            checks.append(
                Check(f"optional:{package}", OK if found else WARN, found or "не установлен")
            )

        cds = _cds(base / ".cdsapirc")
        secret_status = OK if cds else FAIL if strict_secrets else WARN
        checks.append(Check("cds-credentials", secret_status, cds or "~/.cdsapirc не настроен"))
        root = Path(environ.get("AURORA_ROOT", "/data/aurora"))
        checks.append(_storage(root))

    if role in ("inference", "all"):
        for name in ("torch", "aurora"):
            present = available(name)
            checks.append(
                Check(
                    f"module:{name}",
                    OK if present else FAIL,
                    "доступен" if present else "не установлен",
                )
            )
        token = bool(environ.get("HF_TOKEN", "").strip())
        checks.append(
            Check("hf-token", OK if token else FAIL, "задан" if token else "HF_TOKEN не задан")
        )
        visible, detail = probe_cuda()
        checks.append(Check("cuda", OK if visible else FAIL, detail))

    for path in (Path(".env.example"), Path("LICENSES.md")):
        exists = path.is_file()
        checks.append(
            Check(
                f"project:{path}",
                OK if exists else FAIL,
                "на месте" if exists else "нет файла",
            )
        )
    return tuple(checks)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--role", choices=ROLES, default="service")
    cli.add_argument("--strict-secrets", action="store_true")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    checks = inspect_environment(args.role, strict_secrets=args.strict_secrets)
    for check in checks:
        print(f"[{check.status.upper():4}] {check.name}: {check.detail}")
    failed = sum(check.status == FAIL for check in checks)
    warned = sum(check.status == WARN for check in checks)
    print(f"Итого: {len(checks) - failed - warned} ok, {warned} warning, {failed} failed")
    return 1 if failed else 0


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch не установлен"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False"
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    return gib >= 32, f"{torch.cuda.get_device_name(0)}, VRAM {gib:.1f} GiB (нужно ≥32)"


def _cds(path: Path) -> str | None:
    if not path.is_file():
        return None
    names = {
        line.partition(":")[0].strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if ":" in line
    }
    return "url/key присутствуют" if {"url", "key"} <= names else None


def _storage(root: Path) -> Check:
    existing = root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    writable = existing.is_dir() and os.access(existing, os.W_OK)
    detail = f"{root} (ближайший существующий родитель {existing})"
    return Check("storage-root", OK if writable else WARN, detail)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
