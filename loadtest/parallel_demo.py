"""
Show, in one terminal output, that a replica really runs several inferences at once.

Sends N distinct images concurrently, then prints:
  1. the detection each image got back (proving results are not cross-contaminated)
  2. a timeline showing which requests overlapped in time
  3. wall time vs. the sequential-equivalent time

Usage:
  python parallel_demo.py --host https://<fqdn>
  python parallel_demo.py --host https://<fqdn> --count 8 --concurrency 8
"""
import argparse
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw
from urllib3.util.retry import Retry

W, H, RADIUS, Y = 640, 480, 80, 240

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


def make_image(idx: int, total: int):
    """Each image puts its ellipse at a unique X, so results are identifiable."""
    lo, hi = RADIUS + 20, W - RADIUS - 20
    cx = lo + (hi - lo) * idx / max(total - 1, 1)
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - RADIUS, Y - RADIUS, cx + RADIUS, Y + RADIUS],
              fill=(200, 80, 50), outline=(20, 20, 20), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), cx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--endpoint", default="/predict")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    url = args.host.rstrip("/") + args.endpoint
    cases = [(i, *make_image(i, args.count)) for i in range(args.count)]

    # Server config, so the output is self-explanatory.
    try:
        info = requests.get(args.host.rstrip("/") + "/", timeout=30).json()
        workers = info.get("inference_workers")
    except Exception:
        workers = "?"

    print(f"Target             : {url}")
    print(f"INFERENCE_WORKERS  : {workers}   <- max parallel inferences per replica")
    print(f"Sending            : {args.count} distinct images, concurrency {args.concurrency}\n")

    t0 = time.perf_counter()

    def call(case):
        idx, payload, expected_cx = case
        s = session_for_thread(args.concurrency)
        start = time.perf_counter() - t0
        r = s.post(url, files={"file": (f"img{idx}.jpg", payload, "image/jpeg")},
                   timeout=args.timeout)
        end = time.perf_counter() - t0
        return idx, expected_cx, start, end, r.status_code, (r.json() if r.ok else r.text)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = sorted(ex.map(call, cases))

    wall = time.perf_counter() - t0

    # ---- per-image results -------------------------------------------------
    print("RESULTS - what each image got back")
    print(f"{'img':>3}  {'label':<10} {'conf':>5}  {'box centre x':>12}  "
          f"{'expected':>8}  {'infer ms':>8}  {'queue ms':>8}  {'total ms':>8}  match")
    print("-" * 96)

    infer_total = 0.0
    all_match = True
    for idx, exp_cx, start, end, status, body in results:
        if status != 200:
            print(f"{idx:>3}  HTTP {status}")
            all_match = False
            continue
        dets = body.get("detections") or []
        if not dets:
            print(f"{idx:>3}  {'(none)':<10}")
            all_match = False
            continue
        big = max(dets, key=lambda b: (b["box_xyxy"][2] - b["box_xyxy"][0])
                                      * (b["box_xyxy"][3] - b["box_xyxy"][1]))
        cx = (big["box_xyxy"][0] + big["box_xyxy"][2]) / 2
        ok = abs(cx - exp_cx) <= 60
        all_match &= ok
        infer_total += body["inference_ms"]
        print(f"{idx:>3}  {big['label']:<10} {big['confidence']:>5.2f}  {cx:>12.1f}  "
              f"{exp_cx:>8.0f}  {body['inference_ms']:>8.1f}  {body['queue_wait_ms']:>8.1f}  "
              f"{body['total_ms']:>8.1f}  {'OK' if ok else 'WRONG'}")

    # ---- overlap timeline --------------------------------------------------
    # A request's wall time = network + queue wait + inference. To show when the
    # model was ACTUALLY busy we reconstruct the server-side window:
    #   network overhead = client elapsed - server total_ms   (assume half each way)
    #   inference starts after the queue wait, and lasts inference_ms
    windows = []
    for idx, _, start, end, status, body in results:
        if status != 200 or not isinstance(body, dict):
            continue
        elapsed_ms = (end - start) * 1000
        net_ms = max(0.0, elapsed_ms - body["total_ms"])
        i_start = start + (net_ms / 2 + body["queue_wait_ms"]) / 1000
        i_end = i_start + body["inference_ms"] / 1000
        windows.append((idx, body.get("replica", "?")[-6:], i_start, i_end))

    if windows:
        span = max(w[3] for w in windows)
        width = 58
        print(f"\nTIMELINE - when each image was actually inside model.predict()")
        print(f"  (reconstructed from queue_wait_ms + inference_ms; "
              f"1 char = {span / width * 1000:.0f} ms)")
        print(f"{'img':>3}  {'replica':<8} 0s{' ' * (width - 5)}{span:.1f}s")
        print("-" * 88)
        for idx, rep, s_, e_ in sorted(windows, key=lambda w: w[2]):
            lead = int(s_ / span * width)
            bar = max(1, int((e_ - s_) / span * width))
            print(f"{idx:>3}  {rep:<8} {' ' * lead}{'#' * bar}")

        # How many inferences were running simultaneously, sampled finely.
        steps = 600
        conc = []
        for k in range(steps):
            t = span * k / steps
            conc.append(sum(1 for _, _, s_, e_ in windows if s_ <= t < e_))
        max_conc = max(conc) if conc else 0
        busy = [c for c in conc if c > 0]
        avg_conc = sum(busy) / len(busy) if busy else 0
    else:
        max_conc = avg_conc = 0

    # ---- summary -----------------------------------------------------------
    ok_n = sum(1 for r in results if r[4] == 200)
    replicas = {w[1] for w in windows}
    print(f"\nSUMMARY")
    print(f"  requests                 : {ok_n}/{args.count} succeeded")
    print(f"  every image own result   : {'YES' if all_match else 'NO'}")
    print(f"  replicas involved        : {len(replicas)}")
    print(f"  wall clock               : {wall:.2f} s")
    print(f"  sum of inference time    : {infer_total / 1000:.2f} s")
    print(f"  max overlap (estimated)  : {max_conc}   (approximate - see note)")
    print(f"  avg while busy           : {avg_conc:.2f}")
    print(f"\n  NOTE: the timeline splits network time evenly between request and")
    print(f"        response, which is only an approximation. peak_running below is")
    print(f"        measured inside the server and is the authoritative number.")

    try:
        s = requests.get(args.host.rstrip("/") + "/stats", timeout=30).json()
        print(f"\n/stats from one replica:")
        print(f"  inference_workers : {s['inference_workers']}")
        print(f"  peak_in_flight    : {s['peak_in_flight']}   (accepted: queued OR running)")
        print(f"  peak_running      : {s['peak_running']}   <- actually inside predict()")
        print(f"  rss_mb            : {s['rss_mb']} (baseline {s['rss_baseline_mb']})")
    except Exception as exc:
        print(f"\n/stats unavailable: {exc}")


if __name__ == "__main__":
    main()
