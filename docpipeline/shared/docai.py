"""Mistral Document AI (`mistral-ocr-latest`) — text AND figures in one call.

Why this is a better figure source than PDF introspection
---------------------------------------------------------
The PyMuPDF path in shared/pdfsplit.py has to do two awkward things:

  * `get_images()` only sees EMBEDDED RASTERS, so vector diagrams (TikZ,
    matplotlib PDF export, Illustrator) are invisible - resnet.pdf yielded
    0 figures that way, a paper full of diagrams.
  * the vector fallback clusters raw drawing paths by proximity, which is a
    heuristic: it can merge two adjacent diagrams or split a sparse one.

Document AI sidesteps both. It is a layout model, so it returns the figure
regions it actually recognised, with bounding boxes and cropped image bytes -
no guessing from PDF internals. And it returns the page markdown in the SAME
response, so one call replaces two.

    per page:  Content Understanding + PyMuPDF extraction   (2 steps, heuristic)
    becomes:   Mistral Document AI                          (1 step, model-based)

API shape (docs.mistral.ai/capabilities/OCR/basic_ocr):

    POST /v1/ocr
    { "model": "mistral-ocr-latest",
      "document": {"type": "image_url", "image_url": "data:image/png;base64,..."},
      "include_image_base64": true }

    -> { "pages": [ { "index": int,
                      "markdown": str,          # figures appear as ![img-0.jpeg](...)
                      "images": [ { "id": "img-0.jpeg",
                                    "top_left_x": int, "top_left_y": int,
                                    "bottom_right_x": int, "bottom_right_y": int,
                                    "image_base64": "data:image/jpeg;base64,..." } ],
                      "dimensions": {...} } ],
         "usage_info": {...} }

With no MISTRAL_API_KEY the module SIMULATES the service, deriving figures from
PyMuPDF so the offline demo still produces realistic output in the real response
shape. Set the key and the same code path calls the real API.
"""
from __future__ import annotations

import base64
import os

import requests

from .http_policy import TransientError, classify_status

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
# Override for Azure AI Foundry / self-hosted deployments of the same model.
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1/ocr")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-ocr-latest")
MISTRAL_TIMEOUT = float(os.getenv("MISTRAL_TIMEOUT", "120"))


def is_configured() -> bool:
    return bool(MISTRAL_API_KEY)


def _retry_after(resp) -> float | None:
    v = resp.headers.get("Retry-After")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _decode_data_uri(value: str) -> bytes | None:
    """`image_base64` arrives as a data URI, sometimes as bare base64."""
    if not value:
        return None
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(payload)
    except Exception:
        return None


def analyze_page(page_png: bytes, page: int) -> dict:
    """Run Document AI on ONE page image.

    Returns the normalised shape the pipeline consumes:

        {"service", "model", "markdown", "figures": [
            {"figure": int, "bbox": [x0,y0,x1,y1], "width", "height",
             "kind": "docai", "png": bytes}
        ]}
    """
    if not is_configured():
        return _simulate(page_png, page)

    data_uri = "data:image/png;base64," + base64.b64encode(page_png).decode()
    body = {
        "model": MISTRAL_MODEL,
        "document": {"type": "image_url", "image_url": data_uri},
        "include_image_base64": True,
    }
    try:
        resp = requests.post(
            MISTRAL_ENDPOINT, json=body, timeout=MISTRAL_TIMEOUT,
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                     "Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        raise TransientError(f"mistral-docai: {type(exc).__name__}") from exc

    # 429 carries Retry-After; the shared policy honours it. 4xx is permanent.
    err = classify_status(resp.status_code, _retry_after(resp))
    if err:
        raise err

    payload = resp.json()
    pages = payload.get("pages") or [{}]
    p0 = pages[0]

    figures = []
    for idx, img in enumerate(p0.get("images") or []):
        png = _decode_data_uri(img.get("image_base64", ""))
        if not png:
            continue
        x0, y0 = img.get("top_left_x", 0), img.get("top_left_y", 0)
        x1, y1 = img.get("bottom_right_x", 0), img.get("bottom_right_y", 0)
        figures.append({
            "figure": idx, "bbox": [x0, y0, x1, y1],
            "width": max(0, x1 - x0), "height": max(0, y1 - y0),
            "kind": "docai", "png": png,
        })

    return {
        "service": "mistral-document-ai",
        "model": payload.get("model", MISTRAL_MODEL),
        "markdown": p0.get("markdown", ""),
        "dimensions": p0.get("dimensions"),
        "usage": payload.get("usage_info"),
        "figures": figures,
    }


def _simulate(page_png: bytes, page: int) -> dict:
    """Offline stand-in producing the same shape.

    Figures come from the PyMuPDF path so the demo still exercises real figure
    counts; the injected faults come from simulators._maybe_fail so the retry
    policy is tested identically.
    """
    from .simulators import _maybe_fail, _work

    _maybe_fail("mistral-docai")
    _work(1.2)

    return {
        "service": "mistral-document-ai (simulated)",
        "model": MISTRAL_MODEL,
        "markdown": f"# Page {page + 1}\n\nExtracted layout and text for page {page + 1}.\n\n"
                    f"![img-0.jpeg](img-0.jpeg)",
        "dimensions": None,
        "usage": {"pages_processed": 1},
        "figures": [],   # caller falls back to pdfsplit when simulating
    }
