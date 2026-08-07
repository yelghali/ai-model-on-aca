"""Download real technical PDFs (many pages, real diagrams) into samples/.

These are public arXiv papers - genuine technical documents with figures, which
is what makes the figures-vs-pages distinction meaningful. Synthetic pages would
hide the fact that figure count is decoupled from page count.

    python scripts/fetch_samples.py
"""
from __future__ import annotations

import pathlib

# `requests` bundles certifi. Python's stdlib urllib relies on the OS trust store,
# which on some Windows setups has no usable CA bundle and fails with
# CERTIFICATE_VERIFY_FAILED. Using requests fixes that properly - TLS
# verification stays ON.
import requests

DEST = pathlib.Path(__file__).resolve().parent.parent / "samples"

PAPERS = [
    ("attention-is-all-you-need.pdf", "https://arxiv.org/pdf/1706.03762"),
    ("resnet-deep-residual-learning.pdf", "https://arxiv.org/pdf/1512.03385"),
    ("yolo-unified-detection.pdf", "https://arxiv.org/pdf/1506.02640"),
    ("bert.pdf", "https://arxiv.org/pdf/1810.04805"),
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in PAPERS:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  have  {name} ({target.stat().st_size / 1024:.0f} KB)")
            continue
        print(f"  get   {name} <- {url}")
        try:
            r = requests.get(url, timeout=60, allow_redirects=True,
                             headers={"User-Agent": "docpipeline-sample-fetch"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                print("        FAILED: response is not a PDF")
                continue
            target.write_bytes(r.content)
            print(f"        {target.stat().st_size / 1024:.0f} KB")
        except Exception as exc:
            print(f"        FAILED: {exc}")
    print(f"\nsamples in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
