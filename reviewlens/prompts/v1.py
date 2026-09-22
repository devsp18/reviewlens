"""v1 - zero-shot baseline: bare theme/sentiment/severity list, no definitions or examples."""

from reviewlens.schema import SENTIMENT_VALUES, THEME_VALUES

VERSION = "v1"


def build_prompt(reviews: list[str]) -> str:
    themes = ", ".join(THEME_VALUES)
    sentiments = ", ".join(SENTIMENT_VALUES)
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(reviews))

    return f"""Classify each app review below.

Themes (pick exactly one per review): {themes}
Sentiment (pick exactly one per review): {sentiments}
Severity: an integer from 1 (no issue) to 5 (critical, blocking issue)

Return one classification per review, with review_index matching the number below
and one sentence of reasoning for each.

Reviews:
{numbered}
"""
