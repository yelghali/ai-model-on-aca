"""Callback delivery to the uploading app.

Two things the first version got wrong, both fixed here.

1. It swallowed its own retry.
   The activity caught every exception and RETURNED a failure dict. Durable's
   `call_activity_with_retry` only retries on an EXCEPTION, so catching
   everything guaranteed the notification was never retried - if the app was
   down for two seconds the result was silently lost. Now transport failures and
   5xx RAISE (Durable retries), while 4xx returns as permanent (retrying a
   rejected payload is pointless).

2. It was unauthenticated.
   Anyone who learned the URL could post fake results. Callbacks are now signed
   with HMAC-SHA256 over the exact body bytes, so the app can verify the call
   really came from this pipeline.

Verification on the app side:

    import hmac, hashlib
    expected = hmac.new(secret.encode(), request.body, hashlib.sha256).hexdigest()
    hmac.compare_digest(expected, request.headers["X-Signature-256"].split("=")[1])

Compare with `compare_digest`, never `==`, to avoid a timing side channel.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import requests

from .http_policy import PermanentError, TransientError

NOTIFY_URL = os.getenv("NOTIFY_URL", "")
NOTIFY_SECRET = os.getenv("NOTIFY_SECRET", "")
NOTIFY_TIMEOUT = float(os.getenv("NOTIFY_TIMEOUT", "15"))


def is_configured() -> bool:
    return bool(NOTIFY_URL)


def sign(body: bytes) -> str | None:
    if not NOTIFY_SECRET:
        return None
    return "sha256=" + hmac.new(NOTIFY_SECRET.encode(), body, hashlib.sha256).hexdigest()


def post(payload: dict) -> dict:
    """Deliver one callback. Raises on retryable failures so Durable retries."""
    if not NOTIFY_URL:
        return {"notified": False, "reason": "NOTIFY_URL not configured"}

    # Serialise ONCE: the signature must cover the exact bytes sent, or the app
    # will recompute a different digest from re-serialised JSON.
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    headers = {"Content-Type": "application/json",
               "X-Delivery-Timestamp": str(int(time.time()))}
    sig = sign(body)
    if sig:
        headers["X-Signature-256"] = sig

    try:
        resp = requests.post(NOTIFY_URL, data=body, headers=headers, timeout=NOTIFY_TIMEOUT)
    except requests.RequestException as exc:
        # App is down or unreachable -> transient. RAISE so Durable retries.
        raise TransientError(f"notify transport: {type(exc).__name__}") from exc

    if resp.status_code >= 500:
        raise TransientError(f"notify: upstream {resp.status_code}")
    if 400 <= resp.status_code < 500:
        # The app rejected the payload. Retrying will not change its mind.
        raise PermanentError(f"notify: rejected {resp.status_code}")

    return {"notified": True, "status_code": resp.status_code, "signed": bool(sig)}
