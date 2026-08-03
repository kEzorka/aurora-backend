"""Setup-check различает обязательное, опциональное и секретное без утечки значений."""

from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline.setup_check import FAIL, OK, WARN, Check, inspect_environment, main


def _versions(missing: set[str] | None = None) -> Callable[[str], str | None]:
    absent = missing or set()
    return lambda name: None if name in absent else "3.0.1" if name == "zarr" else "1.0"


def _status(checks: tuple[Check, ...], name: str) -> str:
    found = next(check for check in checks if check.name == name)
    return found.status


def test_service_core_is_green_without_importing_optional_packages(tmp_path: Path) -> None:
    (tmp_path / ".cdsapirc").write_text("url: secret-url\nkey: secret-key\n", encoding="utf-8")
    checks = inspect_environment(
        home=tmp_path,
        env={"AURORA_ROOT": str(tmp_path)},
        python=(3, 12),
        version=_versions({"cfgrib", "eccodes", "gcsfs", "cdsapi"}),
    )

    assert _status(checks, "python") == OK
    assert _status(checks, "zarr-format") == OK
    assert _status(checks, "optional:cfgrib") == WARN
    assert _status(checks, "cds-credentials") == OK


def test_missing_core_or_wrong_python_fails(tmp_path: Path) -> None:
    checks = inspect_environment(
        home=tmp_path,
        env={"AURORA_ROOT": str(tmp_path)},
        python=(3, 13),
        version=_versions({"fastapi"}),
    )

    assert _status(checks, "python") == FAIL
    assert _status(checks, "package:fastapi") == FAIL


def test_secrets_warn_on_laptop_and_fail_in_strict_mode(tmp_path: Path) -> None:
    relaxed = inspect_environment(
        home=tmp_path, env={"AURORA_ROOT": str(tmp_path)}, version=_versions()
    )
    strict = inspect_environment(
        home=tmp_path,
        env={"AURORA_ROOT": str(tmp_path)},
        version=_versions(),
        strict_secrets=True,
    )

    assert _status(relaxed, "cds-credentials") == WARN
    assert _status(strict, "cds-credentials") == FAIL


def test_inference_requires_model_token_and_real_cuda(tmp_path: Path) -> None:
    checks = inspect_environment(
        "inference",
        home=tmp_path,
        env={},
        module=lambda name: name == "torch",
        cuda=lambda: (False, "CUDA unavailable"),
    )

    assert _status(checks, "module:torch") == OK
    assert _status(checks, "module:aurora") == FAIL
    assert _status(checks, "hf-token") == FAIL
    assert _status(checks, "cuda") == FAIL


def test_cli_never_prints_credential_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "do-not-print-this-token"
    (tmp_path / ".cdsapirc").write_text(f"url: hidden\nkey: {secret}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert main(["--role", "service"]) == 0
    assert secret not in capsys.readouterr().out
