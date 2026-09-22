"""Structured-output schema validation: valid payloads parse, invalid ones are rejected."""

import pytest
from pydantic import ValidationError

from reviewlens.schema import BatchClassification, ReviewClassification, Sentiment, Theme


def test_valid_classification_parses():
    c = ReviewClassification(review_index=0, theme="UI/UX", sentiment="positive", severity=1, reasoning="fine")
    assert c.theme == Theme.UI_UX
    assert c.sentiment == Sentiment.POSITIVE


def test_invalid_theme_rejected():
    with pytest.raises(ValidationError):
        ReviewClassification(review_index=0, theme="Not A Real Theme", sentiment="positive", severity=1, reasoning="x")


def test_invalid_sentiment_rejected():
    with pytest.raises(ValidationError):
        ReviewClassification(review_index=0, theme="UI/UX", sentiment="ecstatic", severity=1, reasoning="x")


@pytest.mark.parametrize("severity", [0, 6, -1])
def test_severity_out_of_range_rejected(severity):
    with pytest.raises(ValidationError):
        ReviewClassification(review_index=0, theme="UI/UX", sentiment="positive", severity=severity, reasoning="x")


def test_severity_boundary_values_accepted():
    for severity in (1, 5):
        c = ReviewClassification(review_index=0, theme="UI/UX", sentiment="positive", severity=severity, reasoning="x")
        assert c.severity == severity


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        ReviewClassification(review_index=0, theme="UI/UX", sentiment="positive", severity=1)  # no reasoning


def test_batch_classification_wraps_list():
    batch = BatchClassification(classifications=[
        {"review_index": 0, "theme": "Pricing", "sentiment": "negative", "severity": 4, "reasoning": "expensive"},
        {"review_index": 1, "theme": "Other", "sentiment": "neutral", "severity": 1, "reasoning": "vague"},
    ])
    assert len(batch.classifications) == 2
    assert batch.classifications[0].review_index == 0
