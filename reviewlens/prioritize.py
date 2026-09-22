"""Turn a pain point's raw signals into a 0-100 Priority Score.

Four components, each independently rescaled to 0-100, then combined by
user-adjustable weights (must sum to 1):

  frequency_score  - min-max normalized review_count across this app's pain points.
                      The most-mentioned pain point scores 100.
  severity_score    - weighted_severity (1-5, weighted by low star ratings, see
                      cluster._weighted_severity) rescaled linearly to 0-100.
  trend_score        - percent change in mentions, last 30 days vs. the previous 30:
                      pct = (last_30 - prev_30) / max(prev_30, 1) * 100, clipped to
                      [-100, 100], then mapped so -100% -> 0, 0% -> 50, +100% -> 100.
  reach_score        - min-max normalized sum of thumbs-up across this app's pain points.

  priority_score = 100 * (w_freq*freq + w_sev*sev + w_trend*trend + w_reach*reach) / 100
                  (i.e. the weighted average of the four 0-100 component scores)
"""

import pandas as pd

DEFAULT_WEIGHTS = {"frequency": 0.3, "severity": 0.3, "trend": 0.2, "reach": 0.2}


def _min_max_scale(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(100.0, index=series.index)
    return 100.0 * (series - lo) / (hi - lo)


def score_pain_points(pain_points: pd.DataFrame, weights: dict[str, float] = DEFAULT_WEIGHTS) -> pd.DataFrame:
    if pain_points.empty:
        return pain_points.assign(
            frequency_score=[], severity_score=[], trend_score=[], reach_score=[], priority_score=[]
        )

    total = max(1, pain_points["review_count"].sum())
    pp = pain_points.copy()

    pp["frequency_score"] = _min_max_scale(pp["review_count"] / total)
    pp["severity_score"] = (pp["weighted_severity"] - 1) / 4 * 100

    trend_pct = (pp["last_30_count"] - pp["prev_30_count"]) / pp["prev_30_count"].clip(lower=1) * 100
    trend_pct = trend_pct.clip(-100, 100)
    pp["trend_score"] = (trend_pct + 100) / 2

    pp["reach_score"] = _min_max_scale(pp["reach_sum"])

    weight_sum = sum(weights.values()) or 1.0
    pp["priority_score"] = (
        weights["frequency"] * pp["frequency_score"]
        + weights["severity"] * pp["severity_score"]
        + weights["trend"] * pp["trend_score"]
        + weights["reach"] * pp["reach_score"]
    ) / weight_sum

    return pp.sort_values("priority_score", ascending=False).reset_index(drop=True)
