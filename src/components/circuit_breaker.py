"""
DocQuery — Circuit Breaker

Protects against cascading failures when external APIs (OpenAI, Pinecone) are
unavailable or rate-limited.

Three states:
  CLOSED   — Normal operation. Requests flow through.
  OPEN     — Failure threshold exceeded. Requests fail fast (no waiting).
  HALF_OPEN — After cooldown, one probe request is allowed to test recovery.

Configuration tuned for LLM APIs:
  - Trip on: 5 failures in a 60s rolling window (OpenAI)
  - Trip on: 3 failures in a 60s rolling window (Pinecone)
  - Cooldown: 60s before HALF_OPEN probe
  - Recovery: 2 consecutive successes to close from HALF_OPEN

Why this matters:
  OpenAI had a 34-hour outage in June 2025. Without a circuit breaker, every
  query hangs for 30s (request_timeout) before failing. With the breaker OPEN,
  failures are instant — the system degrades gracefully instead of hanging.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional
from src.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # normal — requests flow through
    OPEN = "open"           # failing fast — no requests allowed
    HALF_OPEN = "half_open" # probe — one request allowed to test recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5     # failures in window to trip OPEN
    success_threshold: int = 2     # successes in HALF_OPEN to close
    timeout_duration: int = 60     # seconds to stay OPEN before HALF_OPEN
    window_size: int = 60          # rolling window for failure counting (seconds)
    request_timeout: int = 30      # per-request timeout (informational only)


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the circuit is OPEN."""
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker for external API calls.

    Usage:
        breaker = CircuitBreaker("openai")
        try:
            result = breaker.call(my_openai_function, arg1, arg2)
        except CircuitOpenError:
            # Fast-fail path — OpenAI is known to be down
            return fallback_response()
        except Exception:
            # OpenAI returned an error; breaker already recorded the failure
            raise
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
        self._failure_times: list[float] = []
        self._success_count: int = 0
        self._opened_at: Optional[float] = None

    # ── Internal state helpers ────────────────────────────────────────────────

    def _failures_in_window(self) -> int:
        """Count failures within the rolling time window (must be called with lock held)."""
        now = time.time()
        cutoff = now - self.config.window_size
        self._failure_times = [t for t in self._failure_times if t > cutoff]
        return len(self._failure_times)

    def _record_failure(self):
        with self._lock:
            self._failure_times.append(time.time())
            self._success_count = 0

            if self.state == CircuitState.HALF_OPEN:
                # Probe failed — stay OPEN and extend cooldown
                self.state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.error(
                    "Circuit '%s': OPEN (probe failed, extending cooldown for %ds)",
                    self.name, self.config.timeout_duration,
                )

            elif self._failures_in_window() >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.error(
                    "Circuit '%s': OPEN — %d failures in %ds window (threshold=%d)",
                    self.name,
                    self._failures_in_window(),
                    self.config.window_size,
                    self.config.failure_threshold,
                )

    def _record_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self._failure_times = []
                    self._success_count = 0
                    logger.info("Circuit '%s': CLOSED (recovered after %d successes)", self.name, self.config.success_threshold)

    def _is_request_allowed(self) -> bool:
        """Return True if a request should be attempted. Must NOT hold lock on entry."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                elapsed = time.time() - (self._opened_at or 0)
                if elapsed >= self.config.timeout_duration:
                    self.state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        "Circuit '%s': HALF_OPEN — testing recovery (%.0fs elapsed)",
                        self.name, elapsed,
                    )
                    return True   # allow the probe request
                return False      # still cooling down

            # HALF_OPEN — allow exactly one probe at a time
            if self.state == CircuitState.HALF_OPEN:
                return True

        return True  # safety default

    # ── Public API ────────────────────────────────────────────────────────────

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute func through the circuit breaker.

        Raises:
            CircuitOpenError: if the circuit is OPEN (fast-fail, don't wait).
            Any exception from func: if func raises, the failure is recorded.
        """
        if not self._is_request_allowed():
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN. "
                f"External service unavailable. "
                f"Retry in {self.config.timeout_duration}s."
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except CircuitOpenError:
            raise  # don't record as a new failure — it's already open
        except Exception as exc:
            self._record_failure()
            raise

    @property
    def status(self) -> dict:
        """Return a JSON-serialisable status dict for the /health endpoint."""
        with self._lock:
            failures = self._failures_in_window()
        return {
            "name": self.name,
            "state": self.state.value,
            "failures_in_window": failures,
            "failure_threshold": self.config.failure_threshold,
            "window_size_s": self.config.window_size,
        }

    def __repr__(self):
        return f"<CircuitBreaker name={self.name!r} state={self.state.value}>"


# ── Singleton breakers (shared across all requests in this process) ───────────
#
# These live for the lifetime of the FastAPI / Celery worker process.
# Thread-safe — multiple concurrent requests share the same breaker state.

_openai_breaker = CircuitBreaker(
    "openai",
    CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout_duration=60,
        window_size=60,
        request_timeout=30,
    ),
)

_pinecone_breaker = CircuitBreaker(
    "pinecone",
    CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_duration=30,
        window_size=60,
        request_timeout=10,
    ),
)


def get_openai_breaker() -> CircuitBreaker:
    """Return the singleton OpenAI circuit breaker."""
    return _openai_breaker


def get_pinecone_breaker() -> CircuitBreaker:
    """Return the singleton Pinecone circuit breaker."""
    return _pinecone_breaker
