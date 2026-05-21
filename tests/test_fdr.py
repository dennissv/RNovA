from __future__ import annotations

import pytest

from rnova.fdr import (
    FDRComputationError,
    find_exact_fdr_threshold,
    parse_score_series,
    require_columns,
    require_equal_lengths,
)


def test_find_exact_fdr_threshold_preserves_original_walk_success() -> None:
    tagged = []
    for index in range(200):
        tag = 1 if index == 99 else 0
        tagged.append((float(200 - index), tag))

    threshold, fdr, item = find_exact_fdr_threshold(tagged, label="node")

    assert fdr == 0.01
    assert threshold == 100.0
    assert item == (100.0, 0)


def test_find_exact_fdr_threshold_fails_for_impossible_data() -> None:
    tagged = [(float(20 - index), 0) for index in range(20)]

    with pytest.raises(FDRComputationError, match="no prefix produces"):
        find_exact_fdr_threshold(tagged, label="node")


def test_find_exact_fdr_threshold_fails_for_empty_input() -> None:
    with pytest.raises(FDRComputationError, match="at least two"):
        find_exact_fdr_threshold([], label="node")


def test_require_equal_lengths_reports_mismatch() -> None:
    with pytest.raises(FDRComputationError, match="row count mismatch"):
        require_equal_lengths(2, 3, source="sample")


def test_require_columns_reports_missing() -> None:
    with pytest.raises(FDRComputationError, match="missing required columns"):
        require_columns(["scan", "score"], ("scan", "node_mass", "score"), source="target.csv")


def test_parse_score_series_rejects_bad_values() -> None:
    with pytest.raises(FDRComputationError, match="non-numeric"):
        parse_score_series("1.0;bad;2.0", source="target row 1")
