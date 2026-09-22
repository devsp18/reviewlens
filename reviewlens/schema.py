"""Shared classification taxonomy and Pydantic schema for Gemini structured output."""

from enum import Enum

from pydantic import BaseModel, Field


class Theme(str, Enum):
    PERFORMANCE_CRASHES = "Performance & Crashes"
    LOGIN_ACCOUNT = "Login & Account"
    PAYMENTS_BILLING = "Payments & Billing"
    UI_UX = "UI/UX"
    FEATURE_REQUEST = "Feature Request"
    CUSTOMER_SUPPORT = "Customer Support"
    PRICING = "Pricing"
    NOTIFICATIONS = "Notifications"
    DATA_PRIVACY = "Data & Privacy"
    OTHER = "Other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


THEME_VALUES = [t.value for t in Theme]
SENTIMENT_VALUES = [s.value for s in Sentiment]

THEME_DEFINITIONS: dict[str, str] = {
    "Performance & Crashes": "App is slow, freezes, lags, drains battery, or crashes outright.",
    "Login & Account": "Trouble signing in, account lockouts, verification, password reset, session issues.",
    "Payments & Billing": "Failed transactions, incorrect charges, refunds, payment methods, transfers.",
    "UI/UX": "Layout, navigation, design, readability, ease of use, confusing flows.",
    "Feature Request": "Explicit ask for new functionality or a missing capability compared to competitors.",
    "Customer Support": "Experience contacting or waiting on support, chat bots, response quality.",
    "Pricing": "Cost, subscription fees, value for money, complaints about price increases.",
    "Notifications": "Push notifications, alerts, reminders being excessive, missing, or broken.",
    "Data & Privacy": "Data collection, permissions, security concerns, trust in how data is handled.",
    "Other": "Anything that doesn't clearly fit the themes above (e.g. general praise with no specifics).",
}


class ReviewClassification(BaseModel):
    """One review's classification, keyed by its position in the batch prompt."""

    review_index: int = Field(description="0-based index matching the numbered review in the prompt")
    theme: Theme
    sentiment: Sentiment
    severity: int = Field(ge=1, le=5, description="1 = no issue, 5 = critical/blocking issue")
    reasoning: str = Field(description="One sentence explaining the theme/sentiment/severity choice")


class BatchClassification(BaseModel):
    classifications: list[ReviewClassification]
