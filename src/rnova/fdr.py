from __future__ import annotations

import math
from collections.abc import Iterable

from .errors import RNovAError


class FDRComputationError(RNovAError):
    """Raised when an FDR threshold cannot be computed safely."""


def require_columns(columns: Iterable[str], required: tuple[str, ...], *, source: str) -> None:
    available = set(columns)
    missing = [column for column in required if column not in available]
    if missing:
        raise FDRComputationError(f"{source} is missing required columns: {', '.join(missing)}")


def require_equal_lengths(target_len: int, decoy_len: int, *, source: str) -> None:
    if target_len != decoy_len:
        raise FDRComputationError(
            f"{source} target/decoy row count mismatch: target={target_len}, decoy={decoy_len}"
        )


def parse_score_series(
    raw_score: object,
    *,
    source: str,
    trim_edges: bool = False,
    ignore_negative_inf: bool = False,
) -> list[float]:
    tokens = str(raw_score).split(";")
    if trim_edges:
        tokens = tokens[1:-1]

    scores: list[float] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if ignore_negative_inf and token == "-inf":
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise FDRComputationError(f"{source} contains a non-numeric score: {token!r}") from exc
        if not math.isfinite(value):
            raise FDRComputationError(f"{source} contains a non-finite score: {token!r}")
        scores.append(value)
    return scores


def find_exact_fdr_threshold(
    tagged_scores: Iterable[tuple[float, int]],
    *,
    label: str,
    target_fdr: float = 0.01,
) -> tuple[float, float, tuple[float, int]]:
    merged = sorted(tagged_scores, key=lambda item: item[0], reverse=True)
    if len(merged) < 2:
        raise FDRComputationError(
            f"{label}: need at least two tagged scores to compute the current FDR threshold"
        )

    _validate_tags(merged, label=label)

    original_mid = len(merged) // 2
    found_mid = _find_mid_with_original_walk(merged, original_mid, target_fdr)
    if found_mid is None:
        found_mid = _find_mid_exhaustively(merged, target_fdr)
    if found_mid is None:
        raise FDRComputationError(
            f"{label}: no prefix produces rounded FDR {target_fdr:.2f}; refusing to guess a threshold"
        )

    fdr = _rounded_fdr(merged, found_mid)
    threshold_item = merged[found_mid]
    return threshold_item[0], fdr, threshold_item


def _validate_tags(merged: list[tuple[float, int]], *, label: str) -> None:
    for score, tag in merged:
        if tag not in {0, 1}:
            raise FDRComputationError(f"{label}: invalid FDR tag {tag!r}; expected 0 or 1")
        if not math.isfinite(score):
            raise FDRComputationError(f"{label}: non-finite score {score!r}")


def _find_mid_with_original_walk(
    merged: list[tuple[float, int]],
    mid: int,
    target_fdr: float,
) -> int | None:
    visited: set[int] = set()
    while 0 < mid < len(merged) and mid not in visited:
        visited.add(mid)
        fdr = _rounded_fdr(merged, mid)
        if fdr == target_fdr:
            return mid
        if fdr > target_fdr:
            mid = mid // 2
        else:
            mid = mid + mid // 2
    return None


def _find_mid_exhaustively(merged: list[tuple[float, int]], target_fdr: float) -> int | None:
    for mid in range(1, len(merged)):
        if _rounded_fdr(merged, mid) == target_fdr:
            return mid
    return None


def _rounded_fdr(merged: list[tuple[float, int]], mid: int) -> float:
    prefix = merged[:mid]
    if not prefix:
        raise FDRComputationError("cannot compute FDR from an empty score prefix")
    return round(sum(tag for _, tag in prefix) / len(prefix), 2)
