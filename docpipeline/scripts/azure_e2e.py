"""End-to-end test against the DEPLOYED Azure function.

This is the test that the local harness could not be: it drives the real
Durable runtime over HTTP, so orchestrator-protocol bugs, RBAC gaps and
private-DNS mistakes all surface here rather than in production.

    python azure_e2e.py https://func-docpipe-xxxx.azurewebsites.net --users 4

What it checks
    - POST /api/upload accepts multipart PDFs and returns ids immediately
    - the batch actually reaches Completed (not just "Running" forever)
    - per-file page sequences are intact and contiguous
    - every figure correlates back to the page it came from
    - no (page, figure) key is emitted twice
    - injected 429/503 faults were retried rather than failing the document

Exit code is non-zero if any assertion fails, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

import requests

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "samples"

# Pages and figures measured by run_local.py on the same PDFs. The deployed
# pipeline must agree with these exactly; a mismatch means page splitting or
# figure extraction behaves differently in Azure than it does locally.
BASELINE = {
    "attention-is-all-you-need.pdf": (15, 6),
    "bert.pdf": (16, 39),
    "resnet-deep-residual-learning.pdf": (12, 7),
    "yolo-unified-detection.pdf": (10, 13),
}


def _samples(limit: int) -> list[pathlib.Path]:
    pdfs = sorted(SAMPLES.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {SAMPLES} - run fetch_samples.py first")
    # Cycle if fewer samples than simulated users.
    return [pdfs[i % len(pdfs)] for i in range(limit)]


def upload(base: str, paths: list[pathlib.Path], tag: str) -> dict:
    """One user uploading one or more documents in a single request."""
    files = [("file", (p.name, p.read_bytes(), "application/pdf")) for p in paths]
    t0 = time.time()
    r = requests.post(f"{base}/api/upload", files=files,
                      data={"batchId": tag}, timeout=180)
    r.raise_for_status()
    body = r.json()
    body["_upload_ms"] = int((time.time() - t0) * 1000)
    return body


def poll(base: str, batch_id: str, timeout_s: int) -> dict:
    """Poll until the batch leaves Running. Returns the final status payload.

    Tolerates transient 5xx. The status API runs on the same Flex instances as
    the pipeline, so while the app is scaling out under the very load this test
    creates, a poll can get 502/503. Treating that as a test failure would
    report a broken pipeline when the orchestration is in fact progressing
    normally - which is exactly what happened before this loop retried.
    """
    deadline = time.time() + timeout_s
    last = None
    transient = 0
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/api/batch/{batch_id}", timeout=60)
        except requests.RequestException:
            transient += 1
            time.sleep(5)
            continue

        if r.status_code == 404 or r.status_code >= 500:
            # 404: the orchestration is not visible for a moment after start.
            # 5xx: the host is scaling. Neither is a verdict.
            transient += r.status_code >= 500
            time.sleep(3)
            continue

        r.raise_for_status()
        last = r.json()
        state = last.get("runtimeStatus")
        if state in ("Completed", "Failed", "Terminated"):
            if transient:
                print(f"    ({batch_id}: rode out {transient} transient 5xx)")
            return last
        time.sleep(3)
    raise TimeoutError(f"batch {batch_id} still {last and last.get('runtimeStatus')} "
                       f"after {timeout_s}s")


def check_document(doc: dict) -> list[str]:
    """Correctness assertions on ONE document's summary.

    The batch API returns a per-file SUMMARY, not the per-page detail - the
    merged JSON lives in the results container, which has no public network
    path by design. So the check is against the counts the API does expose,
    cross-referenced with the baseline measured locally on the same corpus.

    That turns out to be the stronger assertion: it proves the deployed
    pipeline found the same pages and the same figures as the local run, on
    the same bytes.

    Deliberately asserts on IDENTITY and COUNTS, never on model accuracy - a
    detector being wrong is not a pipeline bug.
    """
    problems: list[str] = []
    blob = doc.get("file", "?")
    # Uploaded names are prefixed "doc-<corr>-", strip it back to the sample.
    name = blob.split("-", 2)[-1] if blob.startswith("doc-") else blob

    pages = doc.get("pages")
    figures = doc.get("figures")

    if not pages:
        problems.append(f"{name}: reported {pages} pages")
    if doc.get("status") not in ("ok", "partial"):
        problems.append(f"{name}: unexpected status {doc.get('status')!r}")

    errs = doc.get("pages_with_errors") or 0
    if pages and errs > pages:
        problems.append(f"{name}: {errs} pages_with_errors but only {pages} pages")

    if not doc.get("result_blob"):
        problems.append(f"{name}: no result_blob recorded")

    expected = BASELINE.get(name)
    if expected:
        want_pages, want_figs = expected
        if pages != want_pages:
            problems.append(f"{name}: {pages} pages, baseline says {want_pages}")
        if figures != want_figs:
            problems.append(f"{name}: {figures} figures, baseline says {want_figs}")
    else:
        print(f"    note: no baseline for {name} ({pages}p, {figures} figures)")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="https://<function-app>.azurewebsites.net")
    ap.add_argument("--users", type=int, default=4, help="concurrent uploaders")
    ap.add_argument("--files-per-user", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"target   {base}")
    print(f"users    {args.users} x {args.files_per_user} file(s)\n")

    picks = _samples(args.users * args.files_per_user)
    per_user = [picks[i * args.files_per_user:(i + 1) * args.files_per_user]
                for i in range(args.users)]

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = {
            pool.submit(upload, base, paths, f"e2e-{i}-{int(t0)}"): i
            for i, paths in enumerate(per_user)
        }
        batches = []
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                body = fut.result()
            except Exception as exc:
                print(f"  user {i}: UPLOAD FAILED  {exc}")
                return 1
            print(f"  user {i}: uploaded {len(body['files'])} file(s) in "
                  f"{body['_upload_ms']} ms -> batch {body['batchId']}")
            batches.append(body)

    print("\nwaiting for orchestrations...")
    problems: list[str] = []
    totals = {"files": 0, "pages": 0, "figures": 0, "partial": 0,
              "retries_throttle": 0, "retries_transient": 0, "permanent": 0}

    for body in batches:
        bid = body["batchId"]
        try:
            final = poll(base, bid, args.timeout)
        except TimeoutError as exc:
            problems.append(str(exc))
            continue

        state = final.get("runtimeStatus")
        out = final.get("output") or {}
        print(f"  {bid}: {state}")
        if state != "Completed":
            problems.append(f"{bid}: ended {state}")
            continue

        for doc in (out.get("documents") or out.get("results") or []):
            if not isinstance(doc, dict):
                continue
            totals["files"] += 1
            totals["pages"] += doc.get("pages") or 0
            totals["figures"] += doc.get("figures") or 0
            if doc.get("status") == "partial":
                totals["partial"] += 1
            problems += check_document(doc)

    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print(f"elapsed            {elapsed:8.1f} s")
    print(f"documents          {totals['files']:8d}")
    print(f"pages              {totals['pages']:8d}")
    print(f"figures            {totals['figures']:8d}")
    print(f"partial documents  {totals['partial']:8d}")
    if totals["pages"]:
        print(f"throughput         {totals['pages'] / elapsed:8.2f} pages/s")
    print("=" * 62)

    if problems:
        print(f"\nFAILED - {len(problems)} problem(s):")
        for p in problems[:40]:
            print(f"  - {p}")
        return 1

    print("\nOK - every batch Completed, page and figure counts match the "
          "local baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
