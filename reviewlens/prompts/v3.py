"""v3 - v2's definitions plus RAG-retrieved few-shot examples, per review.

Unlike v1/v2's fixed examples, each review here gets its own nearest-neighbor
examples (see reviewlens.rag), retrieved from a k-fold-held-out index so a
review is never retrieved as its own example during evaluation.
"""

from reviewlens.schema import SENTIMENT_VALUES, THEME_DEFINITIONS

VERSION = "v3"


def build_prompt(reviews: list[str], retrieved_examples: list[list[dict]] | None = None) -> str:
    """retrieved_examples[i] is a list of {text, theme, sentiment} dicts for reviews[i]."""
    retrieved_examples = retrieved_examples or [[] for _ in reviews]

    theme_block = "\n".join(f"- {name}: {definition}" for name, definition in THEME_DEFINITIONS.items())
    sentiments = ", ".join(SENTIMENT_VALUES)

    blocks = []
    for i, (text, examples) in enumerate(zip(reviews, retrieved_examples)):
        if examples:
            examples_block = "\n".join(
                f'  Similar labeled review: "{ex["text"]}" -> theme={ex["theme"]}, sentiment={ex["sentiment"]}'
                for ex in examples
            )
            blocks.append(f"{i}. {text}\n{examples_block}")
        else:
            blocks.append(f"{i}. {text}")
    numbered = "\n".join(blocks)

    return f"""You are classifying app store reviews for a product team. Assign exactly one
theme, one sentiment, and a severity score to each review. Each review below may
be preceded by similar labeled examples retrieved to guide your judgment - use
them as reference, but classify based on the review's own content.

Themes:
{theme_block}

Sentiment (pick exactly one): {sentiments}
Severity: an integer from 1 (no issue, purely positive) to 5 (critical, blocking issue)

Now classify these reviews. Return one classification per review, with review_index
matching the number below and one sentence of reasoning for each.

Reviews:
{numbered}
"""
