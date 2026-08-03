"""Cycle timing budget is derived from the schedule, not duplicated."""

import json
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

from pipeline.timing import main, markdown, summarize


def test_normal_and_fallback_windows_are_reported() -> None:
    result = summarize(
        {"timings_sec": {"ingest": 480, "normalize": 190, "inference": 260, "write": 520}}
    )

    assert result.total_seconds == 1450
    assert result.normal_window_seconds == 9000
    assert result.fallback_window_seconds == 3600
    assert result.remaining_seconds == 7550
    assert result.within_budget
    assert "within budget" in markdown(result)


def test_overrun_is_visible() -> None:
    result = summarize(
        {"timings_sec": {"ingest": 8000, "normalize": 500, "inference": 500, "write": 1}}
    )

    assert result.total_seconds == 9001
    assert result.remaining_seconds == -1
    assert not result.within_budget


@pytest.mark.parametrize(
    "timings",
    [
        {"ingest": 1},
        {"ingest": 1, "normalize": 1, "inference": 1, "write": -1},
        {"ingest": 1, "normalize": 1, "inference": True, "write": 1},
    ],
)
def test_invalid_stage_records_are_rejected(timings: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="timings_sec"):
        summarize({"timings_sec": timings})


def test_cli_reads_the_manifest_and_fails_on_overrun(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "published": True,
                "timings_sec": {
                    "ingest": 9001,
                    "normalize": 0,
                    "inference": 0,
                    "write": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    assert main([str(manifest)]) == 1
    assert "over budget" in capsys.readouterr().out
