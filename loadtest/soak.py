"""
Long-running soak test for the ACA model APIs. Runs LOCALLY against the cloud API.

Simulates a realistic day: bursts, quiet periods and full idle gaps, so you can
measure the things that only show up over hours:

  * cold start / "time to create a new instance" - taken from each replica's own
    startup_ms, so it is measured, not inferred
  * how ACA scales out and back to zero as traffic comes and goes
  * retries and failures (ACA resets connections while adding replicas)
  * correctness of every page, under real concurrency
  * replica-seconds -> actual cost

Writes a CSV of every request plus a JSON summary, so a run can be analysed
later or compared between CPU and GPU deployments.

Usage:
  # 2-hour mixed-traffic soak against the CPU deployment
  python soak.py --host https://<fqdn> --profile cpu-4 --hours 2

  # quick 10-minute smoke of the same thing
  python soak.py --host https://<fqdn> --profile cpu-4 --minutes 10

  # GPU deployment
  python soak.py --host https://<fqdn> --profile gpu-t4 --hours 2

  # YOLO detection deployment
  python soak.py --host https://<fqdn> --model yolo --profile cpu-2 --hours 2
"""
import argparse
import csv
import io
import json
import random
import statistics
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
from urllib3.util.retry import Retry

# ACA Consumption plan, pay-as-you-go, USD.
# https://azure.microsoft.com/en-us/pricing/details/container-apps/
PRICING = {
    "vcpu_s": 0.000024, "gib_s": 0.000003,          # active
    "vcpu_s_idle": 0.000003, "gib_s_idle": 0.000003,  # idle (min-replicas kept warm)
    "gpu_t4_s": 0.00009, "gpu_a100_s": 0.000651,    # serverless GPU
    "per_million_requests": 0.40,
}

PROFILES = {
    "cpu-2":    {"label": "2 vCPU / 4 GiB",   "vcpu": 2, "gib": 4},
    "cpu-4":    {"label": "4 vCPU / 8 GiB",   "vcpu": 4, "gib": 8},
    "cpu-8":    {"label": "8 vCPU / 16 GiB",  "vcpu": 8, "gib": 16},
    "gpu-t4":   {"label": "NC8as-T4 serverless GPU",  "gpu": "gpu_t4_s"},
    "gpu-a100": {"label": "NC24-A100 serverless GPU", "gpu": "gpu_a100_s"},
}

# Traffic phases. weight = how often the phase is chosen.
#   batches       : how many batches to send during the phase
#   batch_range   : images per batch
#   gap_range     : seconds to sleep between batches
#   concurrency   : simultaneous batches in flight
PHASES = {
    "burst":  {"weight": 2, "minutes": (2, 5),  "batch_range": (4, 8), "gap_range": (0, 1),   "concurrency": 3},
    "steady": {"weight": 3, "minutes": (5, 12), "batch_range": (2, 4), "gap_range": (3, 12),  "concurrency": 1},
    "low":    {"weight": 3, "minutes": (5, 15), "batch_range": (1, 2), "gap_range": (30, 90), "concurrency": 1},
    "idle":   {"weight": 2, "minutes": (6, 20), "batch_range": (0, 0), "gap_range": (0, 0),   "concurrency": 0},
}

SAFE_ALPHABET = "ACEFHJKLMNPRTWXY"
_CONFUSIONS = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "Q": "O", "D": "O"})
_local = threading.local()


def _norm(s: str) -> str:
    return s.upper().replace(" ", "").replace("-", "").translate(_CONFUSIONS)


def _token(n: int) -> str:
    base, out = len(SAFE_ALPHABET), []
    for _ in range(4):
        out.append(SAFE_ALPHABET[n % base])
        n //= base
    return "".join(reversed(out))


# --- per-model correctness -------------------------------------------------
# Both checks assert IDENTITY (did this image's own result come back?), never
# model accuracy. A returned result that belongs to a different input is the
# cross-contamination bug this whole demo is about.

def _check_paddle(exp, item):
    """The page's unique OCR token must appear in its own text."""
    return _norm(exp) in _norm(item.get("text", ""))


def _check_yolo(exp, item, tol=60.0):
    """The largest box's X centre must match the ellipse we drew in THIS image.

    Returns None when the model detected nothing at all: that is an accuracy
    miss, not contamination, so it is counted separately.
    """
    dets = item.get("detections", [])
    if not dets:
        return None
    big = max(dets, key=lambda b: (b["box_xyxy"][2] - b["box_xyxy"][0])
                                  * (b["box_xyxy"][3] - b["box_xyxy"][1]))
    cx = (big["box_xyxy"][0] + big["box_xyxy"][2]) / 2
    return abs(cx - float(exp)) <= tol


MODELS = {
    "paddle": {"endpoint": "/parse",   "key": "pages",  "check": _check_paddle},
    "yolo":   {"endpoint": "/predict", "key": "images", "check": _check_yolo},
}


def session(pool: int, retries: int) -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        # Retry CONNECTION failures only. read=0 so a request that actually
        # reached the model is never silently repeated.
        r = Retry(total=retries, connect=retries, read=0, status=0, backoff_factor=0.5)
        a = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=r)
        s.mount("https://", a)
        s.mount("http://", a)
        _local.s = s
    return s


def _font(size):
    for n in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_page(idx: int, w=1240, h=1754) -> tuple:
    """A document page carrying a unique, OCR-friendly token for correctness."""
    tok = _token(idx)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((90, 80), "Quarterly Operations Report", fill=(0, 0, 0), font=_font(46))
    d.line([(90, 145), (w - 90, 145)], fill=(0, 0, 0), width=3)
    # The identifying token, large and unambiguous.
    d.text((90, 200), tok, fill=(0, 0, 0), font=_font(90))

    body = ("The container platform processed a record number of documents this "
            "quarter with throughput increasing across all regions. ")
    y = 340
    for _ in range(8):
        d.text((90, y), body[:70], fill=(20, 20, 20), font=_font(22))
        y += 32

    d.rectangle([90, y + 30, w - 90, y + 330], outline=(0, 0, 0), width=2)
    d.rectangle([140, y + 80, 520, y + 290], fill=(70, 130, 180))
    d.ellipse([620, y + 90, 900, y + 280], fill=(200, 90, 60))
    d.text((90, y + 360), "Figure 1. Regional throughput.", fill=(20, 20, 20), font=_font(22))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), tok


def make_detection_page(idx: int, total: int, w=640, h=480) -> tuple:
    """A detection image whose ellipse X centre uniquely identifies it."""
    radius, y = 80, h // 2
    lo, hi = radius + 20, w - radius - 20
    cx = lo + (hi - lo) * idx / max(total - 1, 1)
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - radius, y - radius, cx + radius, y + radius],
              fill=(200, 80, 50), outline=(20, 20, 20), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), cx


def replica_count(app: str, rg: str):
    """Ground truth from Azure; None if the CLI call fails."""
    try:
        out = subprocess.run(
            ["az", "containerapp", "replica", "list", "-n", app, "-g", rg, "-o", "json"],
            capture_output=True, text=True, timeout=90, shell=True,
        )
        if out.returncode != 0:
            return None
        return len(json.loads(out.stdout or "[]"))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--model", default="paddle", choices=sorted(MODELS),
                   help="which API contract to drive")
    p.add_argument("--endpoint", default="", help="override the model's default path")
    p.add_argument("--profile", required=True, choices=sorted(PROFILES))
    p.add_argument("--app", default="paddle-demo-cpu", help="ACA app name (replica polling)")
    p.add_argument("--rg", default="rg-aca-model-demo")
    p.add_argument("--hours", type=float, default=0)
    p.add_argument("--minutes", type=float, default=0)
    p.add_argument("--max-batches", type=int, default=0,
                   help="stop after this many batches (0 = unlimited)")
    p.add_argument("--budget-usd", type=float, default=5.0,
                   help="HARD STOP once estimated spend reaches this. Safety net "
                        "so a forgotten run cannot produce a surprise bill.")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    duration = (args.hours * 3600) + (args.minutes * 60)
    if duration <= 0:
        duration = 3600
    if args.seed:
        random.seed(args.seed)

    url = args.host.rstrip("/") + (args.endpoint or MODELS[args.model]["endpoint"])
    spec = MODELS[args.model]
    result_key, check = spec["key"], spec["check"]
    prof = PROFILES[args.profile]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_base = args.out or f"soak-{args.profile}-{stamp}"
    csv_path, json_path = Path(f"{out_base}.csv"), Path(f"{out_base}.json")

    # Per-replica-second rate, used both for the live budget guard and the report.
    if "gpu" in prof:
        per_replica_s = PRICING[prof["gpu"]]
        basis = f"{prof['label']} @ ${per_replica_s:.6f}/s"
    else:
        per_replica_s = prof["vcpu"] * PRICING["vcpu_s"] + prof["gib"] * PRICING["gib_s"]
        basis = f"{prof['vcpu']} vCPU + {prof['gib']} GiB @ ${per_replica_s:.6f}/s"

    # Pre-render a pool of pages so image generation never skews timing.
    if args.model == "yolo":
        pages = [make_detection_page(i, 64) for i in range(64)]
    else:
        pages = [make_page(i) for i in range(64)]

    print("=" * 74)
    print(f"SOAK TEST  {stamp}")
    print(f"  target    : {url}")
    print(f"  model     : {args.model}")
    print(f"  profile   : {args.profile} ({prof['label']})")
    print(f"  billing   : {basis}")
    print(f"  duration  : {duration/3600:.2f} h"
          + (f"  |  max {args.max_batches} batches" if args.max_batches else ""))
    print(f"  BUDGET    : ${args.budget_usd:.2f} hard stop")
    print(f"  retries   : {args.retries} (connection-level only)")
    print(f"  output    : {csv_path.name}, {json_path.name}")
    print("=" * 74 + "\n")

    rows = []
    lock = threading.Lock()
    startup_by_replica = {}      # replica -> startup_ms (cold start per instance)
    replica_first_seen = {}
    replica_samples = []         # (t, count) from Azure
    totals = Counter()

    t_start = time.time()
    t_end = t_start + duration

    def send(batch_size, phase, after_idle_s):
        s = session(4, args.retries)
        picks = random.sample(pages, batch_size)
        files = [("file", (f"p{i}.jpg", img, "image/jpeg")) for i, (img, _) in enumerate(picks)]
        expected = [tok for _, tok in picks]

        attempts, t0 = 0, time.perf_counter()
        status, body, err, detail = 0, {}, None, ""
        try:
            attempts = 1
            r = s.post(url, files=files, timeout=args.timeout)
            status = r.status_code
            if r.ok:
                body = r.json()
            else:
                # Keep a snippet so a 5xx can be attributed to the app
                # (JSON detail) vs the ACA ingress (HTML/plain text).
                detail = " ".join(r.text.split())[:160]
        except Exception as exc:
            err = type(exc).__name__
        dt_ms = (time.perf_counter() - t0) * 1000

        # Correctness: each page must return its OWN result.
        correct = wrong = nodet = 0
        if status == 200:
            got = body.get(result_key, [])
            for exp, item in zip(expected, got):
                verdict = check(exp, item)
                if verdict is None:
                    nodet += 1      # model found nothing: accuracy, not contamination
                elif verdict:
                    correct += 1
                else:
                    wrong += 1

        rep = body.get("replica")
        if rep:
            with lock:
                if rep not in replica_first_seen:
                    replica_first_seen[rep] = time.time() - t_start
                    if body.get("startup_ms"):
                        startup_by_replica[rep] = body["startup_ms"]

        row = {
            "t_s": round(time.time() - t_start, 1),
            "phase": phase,
            "batch": batch_size,
            "status": status,
            "ms": round(dt_ms, 1),
            "ms_per_image": round(dt_ms / max(batch_size, 1), 1),
            "inference_ms": body.get("inference_ms"),
            "queue_wait_ms": body.get("queue_wait_ms"),
            "replica": rep,
            "startup_ms": body.get("startup_ms"),
            "after_idle_s": round(after_idle_s, 1) if after_idle_s else "",
            "attempts": attempts,
            "correct": correct,
            "wrong": wrong,
            "no_detection": nodet,
            "error": err or "",
            "detail": detail,
        }
        with lock:
            rows.append(row)
            totals["batches"] += 1
            totals["images"] += batch_size
            totals["correct"] += correct
            totals["wrong"] += wrong
            totals["no_detection"] += nodet
            if status == 200:
                totals["ok"] += 1
            else:
                totals["failed"] += 1
                if err:
                    totals[f"err_{err}"] += 1
                else:
                    totals[f"err_HTTP{status}"] += 1
        return row

    # Background: poll Azure for the real replica count.
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            n = replica_count(args.app, args.rg)
            if n is not None:
                with lock:
                    replica_samples.append((round(time.time() - t_start, 1), n))
            stop.wait(30)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    idle_since = None
    phase_names = [k for k, v in PHASES.items() for _ in range(v["weight"])]
    stop_reason = "duration reached"

    def spent_so_far():
        """Live estimate: replica-seconds observed so far x the rate."""
        with lock:
            samples = list(replica_samples)
        if not samples:
            return 0.0
        secs = 0.0
        for (t1, n1), (t2, _) in zip(samples, samples[1:]):
            secs += n1 * (t2 - t1)
        # extrapolate the tail at the last observed count
        secs += samples[-1][1] * max(0.0, (time.time() - t_start) - samples[-1][0])
        return secs * per_replica_s

    def should_stop():
        nonlocal stop_reason
        if time.time() >= t_end:
            stop_reason = "duration reached"
            return True
        if args.max_batches and totals["batches"] >= args.max_batches:
            stop_reason = f"max-batches ({args.max_batches}) reached"
            return True
        est = spent_so_far()
        if est >= args.budget_usd:
            stop_reason = f"BUDGET STOP - estimated ${est:.2f} >= ${args.budget_usd:.2f}"
            return True
        return False

    try:
        while not should_stop():
            phase = random.choice(phase_names)
            cfg = PHASES[phase]
            mins = random.uniform(*cfg["minutes"])
            phase_end = min(time.time() + mins * 60, t_end)
            elapsed_h = (time.time() - t_start) / 3600

            print(f"[{elapsed_h:5.2f} h] phase={phase:<6} for {mins:4.1f} min", flush=True)

            if phase == "idle":
                idle_since = time.time()
                while time.time() < phase_end and not should_stop():
                    time.sleep(min(15, max(0, phase_end - time.time())))
                continue

            with ThreadPoolExecutor(max_workers=max(cfg["concurrency"], 1)) as ex:
                futures = []
                while time.time() < phase_end and not should_stop():
                    bs = random.randint(*cfg["batch_range"])
                    if bs <= 0:
                        break
                    after_idle = (time.time() - idle_since) if idle_since else 0
                    idle_since = None
                    futures.append(ex.submit(send, bs, phase, after_idle))
                    if len(futures) >= cfg["concurrency"]:
                        futures.pop(0).result()
                    time.sleep(random.uniform(*cfg["gap_range"]))
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        pass

            with lock:
                ok, fail = totals["ok"], totals["failed"]
                imgs, nb = totals["images"], totals["batches"]
            print(f"           so far: {nb} batches, {ok} ok / {fail} failed, "
                  f"{imgs} images, ~${spent_so_far():.2f}", flush=True)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nInterrupted - writing results so far\n")
    finally:
        stop.set()

    print(f"\nStopped: {stop_reason}")

    wall = time.time() - t_start

    # ---- write CSV --------------------------------------------------------
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- cost -------------------------------------------------------------
    # Integrate the sampled replica count over time -> replica-seconds.
    replica_seconds = 0.0
    peak_replicas = 0
    if len(replica_samples) >= 2:
        for (t1, n1), (t2, _) in zip(replica_samples, replica_samples[1:]):
            replica_seconds += n1 * (t2 - t1)
            peak_replicas = max(peak_replicas, n1)
        peak_replicas = max(peak_replicas, replica_samples[-1][1])
    elif replica_samples:
        peak_replicas = replica_samples[0][1]
        replica_seconds = replica_samples[0][1] * wall

    if "gpu" in prof:
        pass  # per_replica_s / basis already computed above
    compute_cost = replica_seconds * per_replica_s
    request_cost = totals["batches"] / 1_000_000 * PRICING["per_million_requests"]
    total_cost = compute_cost + request_cost
    imgs = totals["images"] or 1

    ok_rows = [r for r in rows if r["status"] == 200]
    lat = sorted(r["ms_per_image"] for r in ok_rows) if ok_rows else [0]
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]

    cold = sorted(startup_by_replica.values())
    after_idle_rows = [r for r in ok_rows if r["after_idle_s"] not in ("", None)]

    summary = {
        "started_utc": stamp,
        "duration_h": round(wall / 3600, 3),
        "model": args.model,
        "profile": args.profile,
        "billing_basis": basis,
        "batches_sent": totals["batches"],
        "images_sent": totals["images"],
        "succeeded": totals["ok"],
        "failed": totals["failed"],
        "retries_configured": args.retries,
        "correct_pages": totals["correct"],
        "wrong_pages": totals["wrong"],
        "no_detection_pages": totals["no_detection"],
        "p50_ms_per_image": round(p50, 1),
        "p95_ms_per_image": round(p95, 1),
        "instances_created": len(startup_by_replica),
        "startup_ms_min": cold[0] if cold else None,
        "startup_ms_median": statistics.median(cold) if cold else None,
        "startup_ms_max": cold[-1] if cold else None,
        "peak_replicas": peak_replicas,
        "replica_seconds": round(replica_seconds, 1),
        "compute_cost_usd": round(compute_cost, 4),
        "request_cost_usd": round(request_cost, 6),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_1k_images_usd": round(total_cost / imgs * 1000, 4),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"SOAK RESULTS  -  {args.model} / {args.profile} ({prof['label']})")
    print("=" * 74)
    print(f"\nTRAFFIC")
    print(f"  duration            : {wall/3600:.2f} h")
    print(f"  batches sent        : {totals['batches']:,}")
    print(f"  images sent         : {totals['images']:,}")
    print(f"  succeeded / failed  : {totals['ok']:,} / {totals['failed']:,}")
    for k, v in totals.items():
        if k.startswith("err_"):
            print(f"      {v:>4} x {k[4:]}")

    print(f"\nCORRECTNESS")
    print(f"  pages correct       : {totals['correct']:,}")
    print(f"  pages WRONG         : {totals['wrong']:,}  <- cross-contamination")
    if totals["no_detection"]:
        print(f"  no detection        : {totals['no_detection']:,}  (model accuracy, not contamination)")

    print(f"\nLATENCY (per image)")
    print(f"  p50                 : {p50:,.0f} ms")
    print(f"  p95                 : {p95:,.0f} ms")

    print(f"\nINSTANCES")
    print(f"  instances created   : {len(startup_by_replica)}")
    if cold:
        print(f"  startup min/med/max : {cold[0]:,.0f} / "
              f"{statistics.median(cold):,.0f} / {cold[-1]:,.0f} ms")
    print(f"  peak replicas       : {peak_replicas}")
    print(f"  replica-seconds     : {replica_seconds:,.0f}")
    if after_idle_rows:
        first = sorted(after_idle_rows, key=lambda r: -float(r["after_idle_s"]))[:5]
        print(f"  first request after an idle gap:")
        for r in first:
            print(f"      idle {float(r['after_idle_s'])/60:5.1f} min -> {r['ms']:>8,.0f} ms")

    print(f"\nCOST  (ACA Consumption, pay-as-you-go)")
    print(f"  basis               : {basis}")
    print(f"  compute             : ${compute_cost:,.4f}")
    print(f"  requests            : ${request_cost:,.6f}")
    print(f"  TOTAL               : ${total_cost:,.4f}")
    print(f"  per 1k images       : ${total_cost/imgs*1000:,.4f}")
    print(f"\n  (excludes the monthly free grant: 180k vCPU-s, 360k GiB-s, 2M requests)")
    print(f"\nWrote {csv_path.name} and {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
