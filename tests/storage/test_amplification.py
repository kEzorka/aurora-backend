"""BACKLOG 2.7: цифры отчёта получаются из боевых форм чанков."""

import pytest

from storage.amplification import QUERIES, cost, markdown
from storage.layout import LAYOUT_A, LAYOUT_B, read_values


def test_report_covers_both_layouts_and_common_queries() -> None:
    rendered = markdown()

    assert "Maps" in rendered and "Series" in rendered
    assert all(query.name in rendered for query in QUERIES)


def test_one_map_is_exact_in_maps_and_explodes_in_series() -> None:
    query = QUERIES[0]

    assert cost(LAYOUT_A, query).amplification == 1
    assert cost(LAYOUT_B, query).amplification == pytest.approx(1466.08, rel=0.001)


def test_one_year_at_a_point_explodes_in_maps_and_costs_one_tile_in_series() -> None:
    query = QUERIES[4]

    assert cost(LAYOUT_A, query).amplification == 1_038_240
    assert cost(LAYOUT_B, query).amplification == 16


@pytest.mark.parametrize("shape", [(0, 1, 1, 1), (1, 1, 1), (-1, 1, 1, 1)])
def test_invalid_requests_are_not_reported_as_zero_cost(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="four positive"):
        read_values(LAYOUT_A, shape)  # type: ignore[arg-type]
