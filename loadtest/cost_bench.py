"""
Compare document-parsing performance and COST across ACA compute options.

Built for the PP-StructureV3 demo, where the real questions are "does this need
a GPU", "how big should a batch be", and "what does each option cost per page".

The workflow sends MANY images per call, so this sends a BATCH per request
(repeated `file` fields) and reports cost per page, not per request.

Usage:
  # sweep batch sizes on CPU
  python cost_bench.py --host https://<fqdn> --profile cpu-4 --batches 1,2,4,8

  # same sweep on serverless GPU
  python cost_bench.py --host https://<fqdn> --profile gpu-t4 --batches 1,2,4,8,16
"""
import argparse
import io
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw, ImageFont
from urllib3.util.retry import Retry

# Azure Container Apps Consumption plan, pay-as-you-go, USD.
# https://azure.microsoft.com/en-us/pricing/details/container-apps/
PRICING = {
    "vcpu_second": 0.000024,      # active
    "gib_second": 0.000003,       # active
    "gpu_t4_second": 0.00009,     # NC T4 v3, serverless
    "gpu_a100_second": 0.000651,  # NC A100 v4, serverless
}

# Compute profiles we can compare. GPU profiles are billed per GPU-second and
# the vCPU/memory of the node is included in that price.
PROFILES = {
    "cpu-2":  {"label": "2 vCPU / 4 GiB",  "vcpu": 2,  "gib": 4},
    "cpu-4":  {"label": "4 vCPU / 8 GiB",  "vcpu": 4,  "gib": 8},
    "cpu-8":  {"label": "8 vCPU / 16 GiB", "vcpu": 8,  "gib": 16},
    "gpu-t4": {"label": "NC8as-T4 serverless GPU",  "gpu": "gpu_t4_second"},
    "gpu-a100": {"label": "NC24-A100 serverless GPU", "gpu": "gpu_a100_second"},
}

_local = threading.local()


def session_for_thread(pool_size: int) -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(total=2, connect=2, read=0, status=0, backoff_factor=0.3)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.session = s
    return s


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_document_page(width=1240, height=1754):
    """A synthetic A4-ish page: title, two text columns, a figure block.

    Deliberately non-trivial so PP-StructureV3's layout detection has real work
    to do - a blank page would flatter the benchmark.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    title_f, head_f, body_f = _font(46), _font(30), _font(21)

    d.text((90, 80), "Quarterly Operations Report", fill=(0, 0, 0), font=title_f)
    d.line([(90, 145), (width - 90, 145)], fill=(0, 0, 0), width=3)

    body = ("The container platform processed a record number of documents this "
            "quarter, with throughput increasing across all regions. Latency "
            "remained within the agreed service levels for the reporting period. ")

    y = 190
    for col_x in (90, width // 2 + 20):
        y = 190
        d.text((col_x, y), "Section Overview", fill=(0, 0, 0), font=head_f)
        y += 50
        words, line = body.split(), ""
        for w in words * 3:
            if len(line) + len(w) > 42:
                d.text((col_x, y), line, fill=(20, 20, 20), font=body_f)
                y += 30
                line = ""
                if y > height // 2 - 40:
                    break
            line += w + " "

    fy = height // 2 + 40
    d.rectangle([90, fy, width - 90, fy + 380], outline=(0, 0, 0), width=2)
    d.rectangle([140, fy + 60, 520, fy + 320], fill=(70, 130, 180))
    d.ellipse([620, fy + 80, 900, fy + 300], fill=(200, 90, 60))
    d.text((90, fy + 400), "Figure 1. Regional throughput distribution.",
           fill=(20, 20, 20), font=body_f)

    y = fy + 460
    for i in range(6):
        d.text((90, y), f"{i+1}. Operational note regarding regional capacity "
                        f"and planned maintenance windows.", fill=(20, 20, 20), font=body_f)
        y += 34

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--endpoint", default="/parse")
    p.add_argument("--profile", required=True, choices=sorted(PROFILES),
                   help="ACA compute profile the app is running on (for costing)")
    p.add_argument("--batches", default="1,2,4,8",
                   help="comma-separated batch sizes (images per request)")
    p.add_argument("--rounds", type=int, default=3,
                   help="requests to send per batch size")
    p.add_argument("--concurrency", type=int, default=1,
                   help="simultaneous requests (batches in flight)")
    p.add_argument("--replicas", type=int, default=1,
                   help="replicas serving during the run (for costing)")
    p.add_argument("--timeout", type=float, default=900.0)
    args = p.parse_args()

    url = args.host.rstrip("/") + args.endpoint
    page = make_document_page()
    prof = PROFILES[args.profile]
    batches = [int(b) for b in args.batches.split(",")]

    try:
        info = requests.get(args.host.rstrip("/") + "/", timeout=120).json()
    except Exception:
        info = {}

    if "gpu" in prof:
        per_sec = PRICING[prof["gpu"]]
        basis = f"{prof['label']} @ ${per_sec:.6f}/s"
    else:
        per_sec = prof["vcpu"] * PRICING["vcpu_second"] + prof["gib"] * PRICING["gib_second"]
        basis = (f"{prof['vcpu']} vCPU x ${PRICING['vcpu_second']}/s + "
                 f"{prof['gib']} GiB x ${PRICING['gib_second']}/s")
    cluster_per_sec = per_sec * args.replicas

    print(f"Target      : {url}")
    print(f"Profile     : {args.profile}  ({prof['label']})")
    print(f"Billing     : {basis} = ${cluster_per_sec:.6f}/s x {args.replicas} replica(s)")
    print(f"Device      : {info.get('device','?')}   workers: {info.get('inference_workers','?')}")
    print(f"Page        : synthetic A4 document, {len(page)/1024:.0f} KB JPEG")
    print(f"Load        : {args.rounds} requests per batch size, concurrency {args.concurrency}\n")

    def call(batch_size):
        s = session_for_thread(args.concurrency)
        files = [("file", (f"page{i}.jpg", page, "image/jpeg")) for i in range(batch_size)]
        t0 = time.perf_counter()
        try:
            r = s.post(url, files=files, timeout=args.timeout)
        except Exception as exc:
            return 0, (time.perf_counter() - t0) * 1000, {}, str(exc)
        dt = (time.perf_counter() - t0) * 1000
        return r.status_code, dt, (r.json() if r.ok else {}), None

    print(f"{'batch':>6} {'reqs':>5} {'ok':>4} {'ms/req':>9} {'ms/page':>9} "
          f"{'infer/pg':>9} {'pages/s':>8} {'$/1k pages':>11}")
    print("-" * 78)

    rows = []
    for bs in batches:
        call(bs)  # warm this batch shape
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            res = list(ex.map(call, [bs] * args.rounds))
        wall = time.perf_counter() - t0

        ok = [r for r in res if r[0] == 200]
        if not ok:
            first = res[0][3] or f"HTTP {res[0][0]}"
            print(f"{bs:>6} {args.rounds:>5} {0:>4}  failed: {str(first)[:48]}")
            continue

        per_req = statistics.mean(r[1] for r in ok)
        per_page = per_req / bs
        infer_pg = statistics.mean(
            r[2].get("inference_ms_per_image", 0) for r in ok) or float("nan")
        pages_s = (len(ok) * bs) / wall
        cost_1k = (cluster_per_sec / pages_s) * 1000 if pages_s else float("nan")

        print(f"{bs:>6} {args.rounds:>5} {len(ok):>4} {per_req:>9,.0f} {per_page:>9,.0f} "
              f"{infer_pg:>9,.0f} {pages_s:>8.2f} {cost_1k:>11,.2f}")
        rows.append((bs, per_page, pages_s, cost_1k))

    if rows:
        best = min(rows, key=lambda r: r[3])
        base = rows[0]
        print(f"\nBEST BATCH SIZE : {best[0]}  ->  ${best[3]:,.2f} per 1k pages, "
              f"{best[2]:.2f} pages/s")
        if base[3] and best[3]:
            print(f"  vs batch {base[0]:<2}     :  {base[3]/best[3]:.2f}x cheaper, "
                  f"{best[2]/base[2]:.2f}x faster")
        print(f"\n  cost / page       : ${best[3]/1000:.6f}")
        print(f"  cost / 100k pages : ${best[3] * 100:,.2f}")
        print("\n  Excludes the monthly free grant (180k vCPU-s, 360k GiB-s, 2M requests)")
        print("  and request charges ($0.40 per million).")

    try:
        s = requests.get(args.host.rstrip("/") + "/stats", timeout=60).json()
        print(f"\n/stats: workers={s['inference_workers']} peak_running={s['peak_running']} "
              f"rss={s['rss_mb']} MB (baseline {s['rss_baseline_mb']})")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
