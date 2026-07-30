"""
Concurrency sweep benchmark — produces the numbers used in the README.

For each concurrency level it fires a fixed number of requests and reports
throughput and latency percentiles. Run it against a SINGLE replica (min=max=1)
to measure per-replica parallelism, and against an autoscaling app to measure
cluster throughput.

Usage:
  python bench.py --host http://localhost:8000 --endpoint /predict
  python bench.py --host https://<fqdn> --endpoint /ocr --levels 1,2,4,8,16 --requests 60
"""
import argparse
import io
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw
from urllib3.util.retry import Retry

# A requests.Session is not safe to share across many threads: its connection
# pool defaults to 10 and thrashes (or raises SSLEOFError) under higher
# concurrency. Give every worker thread its own session and pool.
_local = threading.local()


def session_for_thread(pool_size: int, retries: int = 2) -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        # Retry only CONNECTION-level failures. While ACA adds replicas, the
        # ingress can reset in-flight connections; a real client retries those.
        # read=0 so we never silently re-run an inference that did execute.
        retry = Retry(total=retries, connect=retries, read=0, status=0,
                      backoff_factor=0.3, raise_on_status=False)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.session = s
    return s


def build_image() -> bytes:
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 300, 220], fill=(70, 130, 180))
    d.ellipse([360, 90, 560, 260], fill=(210, 90, 60))
    d.text((70, 300), "HELLO AZURE CONTAINER APPS", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--endpoint", default="/predict")
    p.add_argument("--levels", default="1,2,4,8,16")
    p.add_argument("--requests", type=int, default=40, help="requests per level")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=2,
                   help="connection-level retries (0 to see raw scale-out resets)")
    args = p.parse_args()

    url = args.host.rstrip("/") + args.endpoint
    payload = build_image()
    levels = [int(x) for x in args.levels.split(",")]
    max_conc = max(levels)

    def one_call():
        session = session_for_thread(max_conc, args.retries)
        files = {"file": ("sample.jpg", payload, "image/jpeg")}
        t0 = time.perf_counter()
        try:
            r = session.post(url, files=files, timeout=args.timeout)
        except Exception as exc:
            return 0, (time.perf_counter() - t0) * 1000, None, None, str(exc)
        dt = (time.perf_counter() - t0) * 1000
        body = r.json() if r.ok else {}
        return r.status_code, dt, body.get("inference_ms"), body.get("replica"), None

    print(f"Target: {url}")
    print(f"{'conc':>5} {'reqs':>5} {'ok':>4} {'RPS':>7} {'p50 ms':>8} "
          f"{'p95 ms':>8} {'max ms':>8} {'infer ms':>9} {'replicas':>9}")
    print("-" * 76)

    for conc in levels:
        # Warm the path so the first level isn't penalised by TLS/cold start.
        one_call()

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            results = list(ex.map(lambda _: one_call(), range(args.requests)))
        wall = time.perf_counter() - t0

        ok = [r for r in results if r[0] == 200]
        errs = [r for r in results if r[0] != 200]
        lats = sorted(r[1] for r in ok)
        infers = [r[2] for r in ok if r[2] is not None]
        replicas = {r[3] for r in ok if r[3]}

        if not lats:
            first = errs[0][4] if errs and errs[0][4] else f"HTTP {errs[0][0]}" if errs else "?"
            print(f"{conc:>5} {args.requests:>5} {0:>4}  all requests failed: {first[:60]}")
            continue

        p50 = statistics.median(lats)
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
        rps = len(ok) / wall
        avg_infer = statistics.mean(infers) if infers else float("nan")

        suffix = f"  ({len(errs)} failed)" if errs else ""
        print(f"{conc:>5} {args.requests:>5} {len(ok):>4} {rps:>7.2f} {p50:>8.1f} "
              f"{p95:>8.1f} {lats[-1]:>8.1f} {avg_infer:>9.1f} {len(replicas):>9}{suffix}")

        if errs:
            # Show WHY requests failed - transport errors and HTTP errors look
            # very different and point at different fixes.
            kinds = {}
            for st, _, _, _, exc in errs:
                key = (exc.split("(")[0].strip()[:70] if exc else f"HTTP {st}")
                kinds[key] = kinds.get(key, 0) + 1
            for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
                print(f"        {v:>4} x {k}")

    # Server-side view of how much parallelism was actually reached.
    try:
        s = session_for_thread(max_conc, args.retries).get(
            args.host.rstrip("/") + "/stats", timeout=15).json()
        print(f"\n/stats from one replica: {s}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
