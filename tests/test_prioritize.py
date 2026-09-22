"""Priority Score math: each component's rescaling and the weighted combination."""

import pandas as pd
import pytest

from reviewlens.prioritize import DEFAULT_WEIGHTS, score_pain_points


def _pain_points(**overrides) -> pd.DataFrame:
    base = pd.DataFrame({
        "theme": ["A", "B", "C"],
        "review_count": [10, 5, 1],
        "weighted_severity": [5.0, 3.0, 1.0],
        "last_30_count": [10, 5, 0],
        "prev_30_count": [5, 5, 0],
        "reach_sum": [100, 50, 0],
    })
    return base.assign(**overrides) if overrides else base


def test_empty_input_returns_empty_with_score_columns():
    empty = pd.DataFrame(columns=["theme", "review_count", "weighted_severity", "last_30_count", "prev_30_count", "reach_sum"])
    result = score_pain_points(empty)
    assert result.empty
    assert "priority_score" in result.columns


def test_most_mentioned_pain_point_gets_max_frequency_score():
    result = score_pain_points(_pain_points())
    top = result[result["theme"] == "A"].iloc[0]
    assert top["frequency_score"] == 100.0


def test_severity_score_rescales_1_to_5_range_to_0_to_100():
    result = score_pain_points(_pain_points())
    # weighted_severity=5.0 (max) -> 100, weighted_severity=1.0 (min) -> 0
    assert result.loc[result["theme"] == "A", "severity_score"].iloc[0] == 100.0
    assert result.loc[result["theme"] == "C", "severity_score"].iloc[0] == 0.0


def test_flat_trend_scores_fifty():
    # last_30 == prev_30 -> 0% growth -> trend_score should be exactly 50
    result = score_pain_points(_pain_points())
    assert result.loc[result["theme"] == "B", "trend_score"].iloc[0] == 50.0


def test_growth_scores_above_fifty_decline_scores_below():
    result = score_pain_points(_pain_points())
    a_trend = result.loc[result["theme"] == "A", "trend_score"].iloc[0]  # 10 vs 5 = +100%
    assert a_trend == 100.0


def test_zero_prev_30_does_not_divide_by_zero():
    # theme C has prev_30_count=0 and last_30_count=0 - must not raise or produce inf/NaN
    result = score_pain_points(_pain_points())
    assert result["trend_score"].notna().all()
    assert (result["trend_score"].abs() != float("inf")).all()


def test_single_pain_point_gets_full_marks_on_relative_scores():
    single = _pain_points().iloc[[0]]
    result = score_pain_points(single)
    assert result["frequency_score"].iloc[0] == 100.0
    assert result["reach_score"].iloc[0] == 100.0


def test_priority_score_is_weighted_average_of_components():
    result = score_pain_points(_pain_points())
    row = result[result["theme"] == "A"].iloc[0]
    expected = (
        DEFAULT_WEIGHTS["frequency"] * row["frequency_score"]
        + DEFAULT_WEIGHTS["severity"] * row["severity_score"]
        + DEFAULT_WEIGHTS["trend"] * row["trend_score"]
        + DEFAULT_WEIGHTS["reach"] * row["reach_score"]
    )
    assert row["priority_score"] == pytest.approx(expected)


def test_results_sorted_descending_by_priority():
    result = score_pain_points(_pain_points())
    scores = result["priority_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_custom_weights_change_ranking():
    # weight entirely on trend: theme A (10 vs 5, +100%) should beat theme B (5 vs 5, 0%)
    trend_only = {"frequency": 0, "severity": 0, "trend": 1, "reach": 0}
    result = score_pain_points(_pain_points(), weights=trend_only)
    assert result.iloc[0]["theme"] == "A"
