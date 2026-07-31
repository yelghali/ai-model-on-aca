"""
Correctness-under-concurrency test.

This is the test that catches the *dangerous* half of the thread-safety bug:
sharing one model across threads does not crash, it silently returns image A's
results for image B. Throughput tests will never reveal that.

Method
------
Build N images that are individually identifiable:

  * YOLO  : each image has ONE ellipse at a position unique to that image.
            The returned bounding-box centre must match that image's position.
  * OCR   : each image contains a unique token (e.g. "CODE-0007").
            The returned text must contain that image's own token.

Send them all CONCURRENTLY, then verify every response against the image that
produced it. Any mismatch = cross-contamination between threads.

Usage
-----
  python correctness.py --host https://<fqdn> --endpoint /predict --count 24 --concurrency 8
  python correctness.py --host https://<fqdn> --endpoint /ocr     --count 24 --concurrency 8
"""
import argparse
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw, ImageFont
from urllib3.util.retry import Retry

W, H = 640, 480
RADIUS = 80
Y_CENTRE = 240

# One session per thread: a shared Session's connection pool defaults to 10 and
# thrashes (or raises SSLEOFError) at higher concurrency.
_local = threading.local()


def session_for_thread(pool_size: int) -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        # Retry CONNECTION failures only (ACA can reset connections while adding
        # replicas). read=0 so a request that actually reached the model is
        # never silently repeated - that would hide, not fix, a correctness bug.
        retry = Retry(total=3, connect=3, read=0, status=0,
                      backoff_factor=0.3, raise_on_status=False)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _local.session = s
    return s


def _font(size=44):
    for name in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# Characters OCR engines reliably distinguish. Deliberately excludes the classic
# confusions: 0/O/D/Q, 1/I/l, 5/S, 8/B, 2/Z, 6/G, U/V. Without this the test
# fails on OCR *character accuracy* and looks like a threading bug, when in fact
# each image got its own correct result.
SAFE_ALPHABET = "ACEFHJKLMNPRTWXY"

# Residual glyph confusions, folded away before comparing.
_CONFUSIONS = str.maketrans({
    "0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "Q": "O", "D": "O",
})


def _normalise(s: str) -> str:
    return s.upper().replace(" ", "").replace("-", "").translate(_CONFUSIONS)


def _token_for(idx: int) -> str:
    """Deterministic, unique, OCR-friendly token: 4 chars of SAFE_ALPHABET."""
    base = len(SAFE_ALPHABET)
    chars = []
    n = idx
    for _ in range(4):
        chars.append(SAFE_ALPHABET[n % base])
        n //= base
    return "".join(reversed(chars))


def make_detection_image(idx: int, total: int):
    """One ellipse whose X centre encodes the image index."""
    span_lo, span_hi = RADIUS + 20, W - RADIUS - 20
    cx = span_lo + (span_hi - span_lo) * idx / max(total - 1, 1)
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - RADIUS, Y_CENTRE - RADIUS, cx + RADIUS, Y_CENTRE + RADIUS],
              fill=(200, 80, 50), outline=(20, 20, 20), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), cx


def make_ocr_image(idx: int, total: int):
    """A unique, high-contrast token that must come back verbatim."""
    token = _token_for(idx)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 200), token, fill=(0, 0, 0), font=_font(96))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), token


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--endpoint", default="/predict")
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--tolerance", type=float, default=60.0,
                   help="max px error between expected and returned centre (YOLO)")
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    url = args.host.rstrip("/") + args.endpoint
    is_ocr = "ocr" in args.endpoint

    cases = []
    for i in range(args.count):
        payload, expected = (make_ocr_image if is_ocr else make_detection_image)(i, args.count)
        cases.append((i, payload, expected))

    session = requests.Session()

    def call(case):
        idx, payload, expected = case
        files = {"file": (f"img{idx}.jpg", payload, "image/jpeg")}
        s = session_for_thread(args.concurrency)
        try:
            r = s.post(url, files=files, timeout=args.timeout)
        except Exception as exc:
            return idx, expected, 0, f"request failed: {exc}"
        return idx, expected, r.status_code, (r.json() if r.ok else r.text)

    print(f"Target      : {url}")
    print(f"Images      : {args.count} (each individually identifiable)")
    print(f"Concurrency : {args.concurrency}\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(call, cases))

    ok = bad = err = 0
    failures = []
    replicas_seen = set()

    for idx, expected, status, body in sorted(results):
        if status != 200:
            err += 1
            failures.append(f"  image {idx}: HTTP {status}")
            continue

        if isinstance(body, dict) and body.get("replica"):
            replicas_seen.add(body["replica"])

        if is_ocr:
            text = _normalise(body.get("text") or "")
            want = _normalise(expected)
            if want in text:
                ok += 1
            else:
                bad += 1
                # Did we get ANOTHER image's token? That is real contamination,
                # as opposed to the model simply misreading a character.
                culprit = next(
                    (j for j in range(args.count)
                     if j != idx and _normalise(_token_for(j)) in text),
                    None,
                )
                hint = (f" -> this is image {culprit}'s token (CONTAMINATION)"
                        if culprit is not None else " (OCR misread, not contamination)")
                failures.append(
                    f"  image {idx}: expected {expected!r}, got {text!r}{hint}")
        else:
            boxes = body.get("detections") or []
            if not boxes:
                bad += 1
                failures.append(f"  image {idx}: expected a detection near x={expected:.0f}, got none")
                continue
            # Largest box is the ellipse.
            big = max(boxes, key=lambda b: (b["box_xyxy"][2] - b["box_xyxy"][0])
                                           * (b["box_xyxy"][3] - b["box_xyxy"][1]))
            cx = (big["box_xyxy"][0] + big["box_xyxy"][2]) / 2
            if abs(cx - expected) <= args.tolerance:
                ok += 1
            else:
                bad += 1
                # Which image DID this result belong to? Strong contamination signal.
                culprit = min(range(args.count),
                              key=lambda j: abs(make_detection_image(j, args.count)[1] - cx))
                failures.append(
                    f"  image {idx}: expected centre x={expected:.0f}, got x={cx:.0f} "
                    f"-> looks like image {culprit}'s result")

    total = len(results)
    print(f"correct       : {ok}/{total}")
    print(f"WRONG RESULT  : {bad}   <-- cross-contamination between threads")
    print(f"errors        : {err}")
    print(f"replicas used : {len(replicas_seen)}")

    if failures:
        print("\nfailures:")
        for f in failures[:20]:
            print(f)

    try:
        s = session.get(args.host.rstrip("/") + "/stats", timeout=15).json()
        print(f"\n/stats: {s}")
    except Exception:
        pass

    print("\nRESULT:", "PASS - every image got its own correct result"
          if bad == 0 and err == 0 else "FAIL")
    return 0 if (bad == 0 and err == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
