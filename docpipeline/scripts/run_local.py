"""Run the pipeline logic locally, without Azure.

Exercises the SAME modules the Durable activities use (pdfsplit, simulators,
http_policy) on a real PDF, with the same two-phase shape:

    PHASE 1  fan out per page   -> Content Understanding + OpenAI + figures
    PHASE 2  fan out per chunk  -> YOLO, batched
    PHASE 3  merge              -> one JSON

The point is to prove the parts that are easy to get wrong - retry
classification, (page, figure) correlation after batching, partial-failure
handling - before involving the Functions host.

    python scripts/run_local.py samples/attention-is-all-you-need.pdf
    python scripts/run_local.py samples/*.pdf --workers 8 --chunk 8
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from shared import docai  # noqa: E402
from shared.http_policy import (  # noqa: E402
    SCALE_SENSITIVE,
    THROTTLE_SENSITIVE,
    PermanentError,
    run_with_policy,
)
from shared.pdfsplit import extract_figures, page_count, render_page  # noqa: E402
from shared.simulators import content_understanding, openai_page, yolo_batch  # noqa: E402

import os  # noqa: E402

TEXT_SERVICE = os.getenv("TEXT_SERVICE", "content-understanding").lower()
STATS = collections.Counter()


def _on_retry(service):
    def hook(attempt, delay, exc):
        STATS[f"retry_{service}"] += 1
        STATS[f"retry_{type(exc).__name__}"] += 1
    return hook


def process_page(pdf: bytes, page: int, dpi: int) -> dict:
    png = render_page(pdf, page, dpi=dpi)
    out = {"page": page + 1, "status": "ok", "figures": [], "errors": []}

    # Text extraction. Document AI returns the FIGURES in the same call;
    # Content Understanding does not, so figures come from the PDF instead.
    doc_figures = []
    if TEXT_SERVICE == "mistral-docai":
        try:
            res = run_with_policy(lambda: docai.analyze_page(png, page), THROTTLE_SENSITIVE,
                                  on_retry=_on_retry("mistral_docai"))
            doc_figures = res.pop("figures", [])
            out["text"] = res
        except PermanentError as exc:
            out["status"] = "partial"
            out["errors"].append({"service": "mistral-document-ai", "error": str(exc)})
            STATS["permanent_docai"] += 1
    else:
        try:
            out["text"] = run_with_policy(
                lambda: content_understanding(png, page), THROTTLE_SENSITIVE,
                on_retry=_on_retry("content_understanding"))
        except PermanentError as exc:
            out["status"] = "partial"
            out["errors"].append({"service": "content-understanding", "error": str(exc)})
            STATS["permanent_cu"] += 1

    try:
        out["openai"] = run_with_policy(
            lambda: openai_page(png, page), THROTTLE_SENSITIVE,
            on_retry=_on_retry("openai"))
    except PermanentError as exc:
        out["status"] = "partial"
        out["errors"].append({"service": "azure-openai", "error": str(exc)})
        STATS["permanent_openai"] += 1

    for fig in (doc_figures or extract_figures(pdf, page)):
        STATS[f"figures_{fig.get('kind', 'unknown')}"] += 1
        out["figures"].append({
            "page": page + 1, "figure": fig["figure"], "kind": fig.get("kind"),
            "width": fig["width"], "height": fig["height"], "png": fig["png"],
        })
    return out


def run(path: pathlib.Path, workers: int, chunk: int, dpi: int, max_pages: int) -> dict:
    pdf = path.read_bytes()
    pages = page_count(pdf)
    if max_pages:
        pages = min(pages, max_pages)
    print(f"\n=== {path.name}  ({pages} pages, {len(pdf)/1024:.0f} KB) ===")

    t0 = time.perf_counter()

    # PHASE 1 - fan out per page
    with ThreadPoolExecutor(max_workers=workers) as ex:
        page_results = list(ex.map(lambda p: process_page(pdf, p, dpi), range(pages)))
    t_pages = time.perf_counter() - t0

    # Collect figures across ALL pages, then chunk. Figure count is decoupled
    # from page count - that is what makes YOLO batching possible here.
    figures = [f for r in page_results for f in r["figures"]]
    chunks = [figures[i:i + chunk] for i in range(0, len(figures), chunk)]
    print(f"  phase 1: {pages} pages in {t_pages:.1f}s -> {len(figures)} figures "
          f"-> {len(chunks)} YOLO batches of <= {chunk}")

    # PHASE 2 - fan out per chunk
    t1 = time.perf_counter()
    def detect(c):
        try:
            return run_with_policy(lambda: yolo_batch(c), SCALE_SENSITIVE,
                                   on_retry=_on_retry("yolo"))
        except PermanentError as exc:
            STATS["permanent_yolo"] += 1
            return [{"page": f["page"], "figure": f["figure"], "count": 0,
                     "detections": [], "error": str(exc)} for f in c]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        yolo_results = list(ex.map(detect, chunks)) if chunks else []
    t_yolo = time.perf_counter() - t1
    print(f"  phase 2: {len(chunks)} batches in {t_yolo:.1f}s")

    # PHASE 3 - merge, correlating by (page, figure) EXPLICITLY
    det = {(d["page"], d["figure"]): d for c in yolo_results for d in c}
    pages_out, failed = [], 0
    for p in sorted(page_results, key=lambda r: r["page"]):
        figs = []
        for f in p["figures"]:
            d = det.get((f["page"], f["figure"]), {})
            figs.append({"figure": f["figure"], "kind": f["kind"],
                         "width": f["width"], "height": f["height"],
                         "count": d.get("count", 0), "detections": d.get("detections", [])})
        if p["status"] != "ok":
            failed += 1
        pages_out.append({k: v for k, v in {
            "page": p["page"], "status": p["status"],
            "text": p.get("text"),
            "openai": p.get("openai"), "figures": figs,
            "errors": p["errors"] or None}.items() if v is not None})

    doc = {"file": path.name, "pages": len(pages_out),
           "figures": sum(len(p["figures"]) for p in pages_out),
           "status": "ok" if failed == 0 else "partial",
           "pages_with_errors": failed, "results": pages_out}

    out_dir = path.parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{path.stem}.json"
    out_file.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # Sanity: every figure must have received its own detection slot.
    assert doc["figures"] == len(figures), "figure lost between phases"
    print(f"  phase 3: status={doc['status']} pages_with_errors={failed} "
          f"-> {out_file.relative_to(path.parent.parent)}  ({out_file.stat().st_size/1024:.0f} KB)")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--workers", type=int, default=8, help="fan-out width")
    ap.add_argument("--chunk", type=int, default=8, help="figures per YOLO call")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    paths = [pathlib.Path(p) for pat in args.pdfs for p in sorted(glob.glob(pat))]
    if not paths:
        print("no PDFs matched - run scripts/fetch_samples.py first")
        return 1

    t0 = time.perf_counter()
    docs = [run(p, args.workers, args.chunk, args.dpi, args.max_pages) for p in paths]
    wall = time.perf_counter() - t0

    print("\n" + "=" * 66)
    print(f"{len(docs)} document(s) in {wall:.1f}s")
    print(f"  pages           : {sum(d['pages'] for d in docs)}")
    print(f"  figures         : {sum(d['figures'] for d in docs)}")
    print(f"  partial docs    : {sum(1 for d in docs if d['status'] != 'ok')}")
    print("\nRETRY / FAILURE BEHAVIOUR (injected faults)")
    for k in sorted(STATS):
        print(f"  {k:<28} {STATS[k]}")
    if not STATS:
        print("  (no faults injected - raise SIM_429_RATE to exercise retries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
