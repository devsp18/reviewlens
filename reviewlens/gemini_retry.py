"""Shared retry policy for Gemini API calls: retry on transient 5xx errors and 429
(rate limit - these free-tier quotas are per-minute and clear quickly), but not on
other 4xx errors (bad request, invalid model, etc.) which won't succeed on retry."""

from google.genai import errors as genai_errors
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429


def gemini_retry(extra_exception_types: tuple[type[BaseException], ...] = ()):
    """Retry decorator: transient Gemini errors, plus any extra exception types
    (e.g. a malformed-response error) that should follow the same backoff schedule."""

    def is_retryable(exc: BaseException) -> bool:
        return _is_transient(exc) or isinstance(exc, extra_exception_types)

    return retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
