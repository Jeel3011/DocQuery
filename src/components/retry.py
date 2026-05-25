"""
DocQuery — Retry with Exponential Backoff + Jitter

Decorator for retrying external API calls on transient failures.

Pattern:
  attempt 1: wait base_delay * 2^0 + jitter  (e.g. 1.0 + random(0, 0.5)s)
  attempt 2: wait base_delay * 2^1 + jitter  (e.g. 2.0 + random(0, 1.0)s)
  attempt 3: wait base_delay * 2^2 + jitter  (e.g. 4.0 + random(0, 2.0)s)
  cap: max_delay (default 30s)

Why jitter?
  Without jitter, all clients retry simultaneously after a 429, creating a
  "thundering herd" that worsens the outage. Jitter spreads retries over a
  window, reducing load on the recovering service.

Usage:
    from src.components.retry import with_retry

    @with_retry(max_retries=3, base_delay=1.0, retryable_exceptions=(openai.RateLimitError,))
    def call_openai(...):
        ...

    # Or use the pre-configured wrappers:
    result = retry_on_openai_error(my_func, arg1, kwarg=val)
"""

import time
import random
import functools
from typing import Tuple, Type, Callable, Any
from src.logger import get_logger

logger = get_logger(__name__)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
):
    """
    Decorator: retry a function with exponential backoff + jitter.

    Args:
        max_retries:             Maximum number of retries (total attempts = max_retries + 1).
        base_delay:              Base wait time in seconds (doubles each retry).
        max_delay:               Maximum wait cap in seconds.
        retryable_exceptions:    Only these exception types trigger a retry.
        non_retryable_exceptions: These exceptions bypass retries immediately (e.g. auth errors).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except non_retryable_exceptions as exc:
                    logger.error(
                        "[retry] Non-retryable error in %s: %s", func.__name__, exc
                    )
                    raise
                except retryable_exceptions as exc:
                    last_exception = exc

                    if attempt == max_retries:
                        logger.error(
                            "[retry] %s failed after %d/%d attempts: %s",
                            func.__name__, attempt + 1, max_retries + 1, exc,
                        )
                        raise

                    # Exponential backoff with full jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)
                    wait = delay + jitter

                    logger.warning(
                        "[retry] %s attempt %d/%d failed (%s: %s). Retrying in %.2fs…",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        type(exc).__name__,
                        str(exc)[:120],
                        wait,
                    )
                    time.sleep(wait)

            # Should never reach here
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def retry_on_openai_error(func: Callable, *args, **kwargs) -> Any:
    """
    Convenience wrapper: run func(*args, **kwargs) with OpenAI-appropriate retry config.

    Retries on general Exception (catches rate limits, server errors).
    Use max_retries=2 to avoid spending too long on a dead API.
    """
    try:
        import openai
        retryable = (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError, Exception)
    except ImportError:
        retryable = (Exception,)

    @with_retry(
        max_retries=2,
        base_delay=2.0,
        max_delay=20.0,
        retryable_exceptions=retryable,
    )
    def _inner():
        return func(*args, **kwargs)

    return _inner()
