"""v2 - v1 plus clear per-theme definitions and hand-written few-shot examples."""

from reviewlens.schema import SENTIMENT_VALUES, THEME_DEFINITIONS

VERSION = "v2"

FEW_SHOT_EXAMPLES = [
    {
        "text": "App crashes every time I try to open my transaction history, super frustrating.",
        "theme": "Performance & Crashes",
        "sentiment": "negative",
        "severity": 4,
    },
    {
        "text": "Can't log back in after changing my phone, verification code never arrives.",
        "theme": "Login & Account",
        "sentiment": "negative",
        "severity": 5,
    },
    {
        "text": "Wish there was a dark mode, my eyes hurt using this at night.",
        "theme": "Feature Request",
        "sentiment": "neutral",
        "severity": 2,
    },
    {
        "text": "Simple, clean, does exactly what I need. Love it.",
        "theme": "UI/UX",
        "sentiment": "positive",
        "severity": 1,
    },
    {
        "text": "Got charged twice for the same order and support took a week to refund me.",
        "theme": "Payments & Billing",
        "sentiment": "negative",
        "severity": 5,
    },
]


def build_prompt(reviews: list[str]) -> str:
    theme_block = "\n".join(f"- {name}: {definition}" for name, definition in THEME_DEFINITIONS.items())
    sentiments = ", ".join(SENTIMENT_VALUES)
    examples_block = "\n".join(
        f'Review: "{ex["text"]}"\n-> theme={ex["theme"]}, sentiment={ex["sentiment"]}, severity={ex["severity"]}'
        for ex in FEW_SHOT_EXAMPLES
    )
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(reviews))

    return f"""You are classifying app store reviews for a product team. Assign exactly one
theme, one sentiment, and a severity score to each review.

Themes:
{theme_block}

Sentiment (pick exactly one): {sentiments}
Severity: an integer from 1 (no issue, purely positive) to 5 (critical, blocking issue)

Examples:
{examples_block}

Now classify these reviews. Return one classification per review, with review_index
matching the number below and one sentence of reasoning for each.

Reviews:
{numbered}
"""
