"""Simulated Azure OpenAI and Azure AI Content Understanding, plus the YOLO client.

The simulators exist to EXERCISE the retry policy, not to bypass it. They inject
the same failure modes the real services produce:

  429 + Retry-After   Azure OpenAI under TPM pressure
  503                 transient service unavailability
  400                 a page the service refuses (permanent - must NOT be retried)

Failure rates are env-tunable so a demo can dial them up and show the workflow
surviving. Defaults are realistic-ish, not pathological.

The YOLO client is real by default: point YOLO_URL at the ACA app and it sends a
true batch (repeated `file` parts). With YOLO_URL unset it simulates, so the
whole pipeline runs locally for free.
"""
from __future__ import annotations

import hashlib
import os
import random
import time

import requests

from .http_policy import PermanentError, ThrottledError, TransientError, classify_status

# Injected failure rates (0..1)
SIM_429_RATE = float(os.getenv("SIM_429_RATE", "0.15"))
SIM_503_RATE = float(os.getenv("SIM_503_RATE", "0.05"))
SIM_400_RATE = float(os.getenv("SIM_400_RATE", "0.01"))
SIM_LATENCY_MS = float(os.getenv("SIM_LATENCY_MS", "250"))

YOLO_URL = os.getenv("YOLO_URL", "").rstrip("/")
YOLO_TIMEOUT = float(os.getenv("YOLO_TIMEOUT", "120"))


def _maybe_fail(service: str) -> None:
    """Inject a failure mode. Deliberately raises the SAME exception types the
    real HTTP path produces, so the retry policy is identical either way."""
    r = random.random()
    if r < SIM_400_RATE:
        raise PermanentError(f"{service}: unprocessable page (400)")
    if r < SIM_400_RATE + SIM_429_RATE:
        # Real services return Retry-After; the policy must honour it.
        raise ThrottledError(f"{service}: throttled (429)", retry_after=random.uniform(1, 3))
    if r < SIM_400_RATE + SIM_429_RATE + SIM_503_RATE:
        raise TransientError(f"{service}: service unavailable (503)")


def _work(scale: float = 1.0) -> None:
    time.sleep(SIM_LATENCY_MS / 1000.0 * scale * random.uniform(0.6, 1.4))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def content_understanding(page_png: bytes, page: int) -> dict:
    """Azure AI Content Understanding (simulated).

    Returns the shape the real service gives for a document analyser: markdown
    content plus extracted fields and tables.
    """
    _maybe_fail("content-understanding")
    _work(1.0)
    h = _digest(page_png)
    return {
        "service": "azure-ai-content-understanding",
        "analyzer": os.getenv("CU_ANALYZER", "prebuilt-documentAnalyzer"),
        "markdown": f"# Page {page + 1}\n\nExtracted layout and text for page {page + 1} ({h}).",
        "fields": {"page_number": page + 1, "content_hash": h},
        "tables": [],
    }


def openai_page(page_png: bytes, page: int) -> dict:
    """Azure OpenAI vision summarisation (simulated)."""
    _maybe_fail("azure-openai")
    _work(1.4)
    return {
        "service": "azure-openai",
        "model": os.getenv("OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        "summary": f"Summary of page {page + 1}.",
        "usage": {"prompt_tokens": 900, "completion_tokens": 120},
    }


def yolo_batch(figures: list[dict]) -> list[dict]:
    """Detect on a BATCH of figures.

    figures: [{page, figure, png}]
    returns: [{page, figure, count, detections}] - correlation is explicit,
             never positional, because figures-per-page varies.
    """
    if not YOLO_URL:
        _work(0.4)
        return [
            {"page": f["page"], "figure": f["figure"], "count": 1,
             "detections": [{"label": "diagram", "confidence": 0.90,
                             "box_xyxy": [10.0, 10.0, 200.0, 150.0]}]}
            for f in figures
        ]

    files = [
        ("file", (f"p{f['page']:04d}-f{f['figure']:02d}.png", f["png"], "image/png"))
        for f in figures
    ]
    try:
        resp = requests.post(f"{YOLO_URL}/predict", files=files, timeout=YOLO_TIMEOUT)
    except requests.RequestException as exc:
        # Connection reset mid-scale-out is transient, not a bad request.
        raise TransientError(f"yolo: {type(exc).__name__}") from exc

    err = classify_status(resp.status_code, _retry_after(resp))
    if err:
        raise err

    body = resp.json()
    items = body.get("images") or [{"count": body.get("count", 0),
                                    "detections": body.get("detections", [])}]
    return [
        {"page": f["page"], "figure": f["figure"],
         "count": item.get("count", 0), "detections": item.get("detections", [])}
        for f, item in zip(figures, items)
    ]


def _retry_after(resp) -> float | None:
    v = resp.headers.get("Retry-After")
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None
