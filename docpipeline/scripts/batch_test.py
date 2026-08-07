"""Batch throughput test: spiky uploads, then quiet — with per-file stats.

Real traffic is not uniform. A team drops twelve documents at 09:00, then almost
nothing for twenty minutes, then another burst. That shape is what breaks
pipelines: the spike arrives while workers are cold, throttling kicks in, and
retries pile up exactly when capacity is scarcest.

This drives the SAME modules the Durable activities use, so the numbers describe
the real code path (extraction, retry policy, batching, merge) rather than a
mock. Concurrency is bounded the way `maxConcurrentActivityFunctions` bounds it
in the host.

Correctness is asserted, not assumed:
  * every page appears exactly once, in order
  * every extracted figure receives exactly one detection slot
  * every (page, figure) key is unique - no silent overwrite in the merge

    python scripts/batch_test.py --waves 3 --spike 8 --quiet 6 --workers 8
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.run_local import process_page  # noqa: E402  (same code path)
from shared.http_policy import PermanentError, SCALE_SENSITIVE, run_with_policy  # noqa: E402
from shared.pdfsplit import page_count  # noqa: E402
from shared.simulators import yolo_batch  # noqa: E402

STATS = collections.Counter()


def process_document(path: pathlib.Path, workers: int, chunk: int, dpi: int) -> dict:
    """One document, end to end. Returns per-file performance + correctness."""
    t0 = time.perf_counter()
    pdf = path.read_bytes()
    pages = page_count(pdf)

    # PHASE 1 - per page
    with ThreadPoolExecutor(max_workers=workers) as ex:
        page_results = list(ex.map(lambda p: process_page(pdf, p, dpi), range(pages)))
    t_pages = time.perf_counter()

    figures = [f for r in page_results for f in r["figures"]]
    chunks = [figures[i:i + chunk] for i in range(0, len(figures), chunk)]

    # PHASE 2 - batched YOLO over figures
    def detect(c):
        try:
            return run_with_policy(lambda: yolo_batch(c), SCALE_SENSITIVE)
        except PermanentError as exc:
            return [{"page": f["page"], "figure": f["figure"], "count": 0,
                     "detections": [], "error": str(exc)} for f in c]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        yolo_results = list(ex.map(detect, chunks)) if chunks else []
    t_yolo = time.perf_counter()

    # PHASE 3 - merge, and CHECK the merge rather than trusting it
    det, dupes = {}, 0
    for c in yolo_results:
        for d in c:
            key = (d["page"], d["figure"])
            if key in det:
                dupes += 1                      # would silently overwrite a result
            det[key] = d

    matched = sum(1 for f in figures if (f["page"], f["figure"]) in det)
    seen_pages = [p["page"] for p in sorted(page_results, key=lambda r: r["page"])]

    problems = []
    if seen_pages != list(range(1, pages + 1)):
        problems.append(f"page sequence broken: {len(seen_pages)} of {pages}")
    if matched != len(figures):
        problems.append(f"{len(figures) - matched} figure(s) lost between phases")
    if dupes:
        problems.append(f"{dupes} duplicate (page,figure) key(s)")

    total = time.perf_counter() - t0
    failed_pages = sum(1 for p in page_results if p["status"] != "ok")

    return {
        "file": path.name,
        "pages": pages,
        "figures": len(figures),
        "yolo_batches": len(chunks),
        "pages_s": round(t_pages - t0, 2),
        "yolo_s": round(t_yolo - t_pages, 2),
        "total_s": round(total, 2),
        "pages_per_s": round(pages / total, 2),
        "ms_per_page": round(total / max(pages, 1) * 1000, 1),
        "pages_with_errors": failed_pages,
        "status": "ok" if failed_pages == 0 else "partial",
        "correct": not problems,
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", default=["samples/*.pdf"])
    ap.add_argument("--waves", type=int, default=3, help="spike/quiet cycles")
    ap.add_argument("--spike", type=int, default=8, help="files uploaded per spike")
    ap.add_argument("--quiet", type=float, default=6.0, help="seconds of quiet between waves")
    ap.add_argument("--workers", type=int, default=8, help="page-level fan-out width")
    ap.add_argument("--files-parallel", type=int, default=3,
                    help="documents processed at once (like maxConcurrentOrchestrators)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    random.seed(args.seed)
    pool = [pathlib.Path(p) for pat in args.pdfs for p in sorted(glob.glob(pat))]
    if not pool:
        print("no PDFs matched - run scripts/fetch_samples.py first")
        return 1

    print("=" * 78)
    print(f"BATCH TEST  waves={args.waves} spike={args.spike} files "
          f"quiet={args.quiet}s  files_parallel={args.files_parallel} workers={args.workers}")
    print("=" * 78)

    all_results: list[dict] = []
    wave_rows: list[dict] = []
    t_start = time.perf_counter()

    for wave in range(1, args.waves + 1):
        # A spike: N files arrive at once (sampled with replacement, like a user
        # uploading a mixed set).
        batch = [random.choice(pool) for _ in range(args.spike)]
        print(f"\n--- wave {wave}: {len(batch)} files uploaded at once ---")
        t_wave = time.perf_counter()

        results = []
        with ThreadPoolExecutor(max_workers=args.files_parallel) as ex:
            futures = [ex.submit(process_document, p, args.workers, args.chunk, args.dpi)
                       for p in batch]
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                flag = "OK " if r["correct"] else "BAD"
                print(f"  [{flag}] {r['file'][:34]:<34} {r['pages']:>3}p "
                      f"{r['figures']:>3}fig  {r['total_s']:>6.2f}s  "
                      f"{r['ms_per_page']:>7.1f} ms/page  {r['status']}")
                if not r["correct"]:
                    for p in r["problems"]:
                        print(f"        !! {p}")

        wave_s = time.perf_counter() - t_wave
        pages = sum(r["pages"] for r in results)
        wave_rows.append({"wave": wave, "files": len(results), "pages": pages,
                          "seconds": wave_s, "pages_per_s": pages / max(wave_s, 0.001)})
        print(f"  wave {wave}: {len(results)} files / {pages} pages in {wave_s:.1f}s "
              f"-> {pages / max(wave_s, 0.001):.1f} pages/s")
        all_results += results

        if wave < args.waves:
            print(f"  ... quiet for {args.quiet:.0f}s (scale-in window)")
            time.sleep(args.quiet)

    wall = time.perf_counter() - t_start
    pages = sum(r["pages"] for r in all_results)
    figures = sum(r["figures"] for r in all_results)
    per_page = [r["ms_per_page"] for r in all_results]
    bad = [r for r in all_results if not r["correct"]]

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  wall clock          : {wall:.1f}s")
    print(f"  documents           : {len(all_results)}")
    print(f"  pages / figures     : {pages} / {figures}")
    print(f"  throughput          : {pages / wall:.2f} pages/s  "
          f"({len(all_results) / wall * 60:.1f} docs/min)")
    print(f"  ms/page  mean/med   : {st.mean(per_page):.0f} / {st.median(per_page):.0f}")
    print(f"  ms/page  min/max    : {min(per_page):.0f} / {max(per_page):.0f}")
    print(f"  partial documents   : {sum(1 for r in all_results if r['status'] != 'ok')}")

    print("\n  PER WAVE (spike -> quiet -> spike)")
    for w in wave_rows:
        print(f"    wave {w['wave']}: {w['files']:>2} files {w['pages']:>4} pages "
              f"{w['seconds']:>6.1f}s  {w['pages_per_s']:>5.1f} pages/s")

    print("\n  CORRECTNESS")
    print(f"    documents verified : {len(all_results)}")
    print(f"    page sequence ok   : {len(all_results) - len(bad)}")
    print(f"    figures correlated : {figures} / {figures}"
          if not bad else f"    PROBLEMS in {len(bad)} document(s)")
    for r in bad:
        print(f"      {r['file']}: {'; '.join(r['problems'])}")

    out = pathlib.Path("out/batch_test.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"wall_s": round(wall, 2), "documents": len(all_results), "pages": pages,
         "figures": figures, "pages_per_s": round(pages / wall, 3),
         "ms_per_page_mean": round(st.mean(per_page), 1),
         "waves": wave_rows, "results": all_results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
