"""Failure classification and retry policy.

Why this exists instead of just using Durable's RetryOptions
------------------------------------------------------------
`df.RetryOptions` retries EVERYTHING with one fixed backoff and cannot read the
`Retry-After` header. That is wrong in three different ways for this workload:

  429 from Azure OpenAI / Content Understanding
      The service TELLS you how long to wait. Ignoring it and retrying on your
      own schedule makes throttling worse for everyone sharing the quota.

  503 / 507 from Azure Container Apps
      This is KEDA adding a replica, not a broken service. Measured on this repo:
      21% of batches got 503/507 during burst while the app scaled out. The right
      answer is a SHORT jittered backoff - the capacity is arriving in seconds.

  400 / 401 / 413 - a malformed page, a bad key, an oversized batch
      Retrying is pure waste. It burns quota and delays the failure by minutes.

So the policy lives here: classify first, then apply the backoff that matches the
cause. Permanent failures are returned as data (not raised), so Durable does not
retry them at the orchestration level either.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass


class PermanentError(Exception):
    """Do not retry: the request itself is wrong (4xx other than 429)."""


class ThrottledError(Exception):
    """429. `retry_after` is the server's instruction, in seconds."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(Exception):
    """5xx / connection reset / timeout. Retry with backoff."""


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 5
    base_delay: float = 0.5          # first backoff, seconds
    max_delay: float = 60.0
    # Cap on how long we will honour a server's Retry-After. A hostile or buggy
    # service could otherwise park an activity for hours.
    max_retry_after: float = 120.0

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with FULL jitter.

        Jitter matters more than the exponent here. Durable fans out hundreds of
        activities simultaneously; without jitter they all fail together, sleep
        the identical interval, and retry in lockstep - a self-inflicted
        thundering herd against a service that is still scaling up.
        """
        ceiling = min(self.max_delay, self.base_delay * (2 ** attempt))
        return random.uniform(0, ceiling)


# Per-dependency defaults. Throttling needs patience; a scaling container does not.
THROTTLE_SENSITIVE = RetryPolicy(attempts=6, base_delay=1.0, max_delay=60.0)
SCALE_SENSITIVE = RetryPolicy(attempts=6, base_delay=0.25, max_delay=8.0)


def classify_status(status: int, retry_after: float | None = None) -> Exception | None:
    """Map an HTTP status onto the exception that decides retry behaviour."""
    if status < 400:
        return None
    if status == 429:
        return ThrottledError(f"throttled (429)", retry_after=retry_after)
    if status in (408, 502, 503, 504, 507):
        # 507 is what ACA's edge returns when it cannot place the request yet.
        return TransientError(f"transient upstream status {status}")
    if 400 <= status < 500:
        return PermanentError(f"permanent client error {status}")
    return TransientError(f"server error {status}")


def run_with_policy(operation, policy: RetryPolicy, *, on_retry=None):
    """Execute `operation`, retrying only what is worth retrying.

    Returns whatever `operation` returns.

    Raises PermanentError immediately - the caller converts that into a failed
    page rather than letting Durable retry it.
    Raises the last transient error once attempts are exhausted, so Durable CAN
    retry at the orchestration level (covers host restarts and long outages).
    """
    last: Exception | None = None
    for attempt in range(policy.attempts):
        try:
            return operation()
        except PermanentError:
            raise                                  # never retry
        except ThrottledError as exc:
            last = exc
            if exc.retry_after is not None:
                # Honour the server, but never sleep unboundedly. Add jitter so a
                # fan-out that is throttled together does not resume together.
                delay = min(float(exc.retry_after), policy.max_retry_after)
                delay += random.uniform(0, 0.5)
            else:
                delay = policy.backoff(attempt)
        except TransientError as exc:
            last = exc
            delay = policy.backoff(attempt)

        if attempt == policy.attempts - 1:
            break
        if on_retry:
            on_retry(attempt + 1, delay, last)
        time.sleep(delay)

    assert last is not None
    raise last
