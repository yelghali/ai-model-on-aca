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
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw


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
    args = p.parse_args()

    url = args.host.rstrip("/") + args.endpoint
    payload = build_image()
    levels = [int(x) for x in args.levels.split(",")]

    session = requests.Session()

    def one_call():
        files = {"file": ("sample.jpg", payload, "image/jpeg")}
        t0 = time.perf_counter()
        r = session.post(url, files=files, timeout=args.timeout)
        dt = (time.perf_counter() - t0) * 1000
        body = r.json() if r.ok else {}
        return r.status_code, dt, body.get("inference_ms"), body.get("replica")

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
        lats = sorted(r[1] for r in ok)
        infers = [r[2] for r in ok if r[2] is not None]
        replicas = {r[3] for r in ok if r[3]}

        if not lats:
            print(f"{conc:>5} {args.requests:>5} {0:>4}  all requests failed")
            continue

        p50 = statistics.median(lats)
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
        rps = len(ok) / wall
        avg_infer = statistics.mean(infers) if infers else float("nan")

        print(f"{conc:>5} {args.requests:>5} {len(ok):>4} {rps:>7.2f} {p50:>8.1f} "
              f"{p95:>8.1f} {lats[-1]:>8.1f} {avg_infer:>9.1f} {len(replicas):>9}")

    # Server-side view of how much parallelism was actually reached.
    try:
        s = session.get(args.host.rstrip("/") + "/stats", timeout=10).json()
        print(f"\n/stats from one replica: {s}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
