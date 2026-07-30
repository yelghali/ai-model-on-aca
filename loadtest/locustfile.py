"""
Locust load test that works against either demo (YOLO /predict or OCR /ocr).

Usage:
  locust -f locustfile.py --host https://<app-fqdn> --headless -u 100 -r 10 -t 3m
  locust -f locustfile.py --host http://localhost:8000 --endpoint /ocr

Custom options:
  --endpoint   inference path (default /predict)
  --image      path to the image to send (default: generated synthetic image)
"""
import io
import random

from locust import HttpUser, between, events, task


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--endpoint", type=str, default="/predict",
                        help="Inference endpoint path, e.g. /predict or /ocr")
    parser.add_argument("--image", type=str, default="",
                        help="Path to an image file to upload; synthetic if empty")


def _synthetic_image() -> bytes:
    """Build a small JPEG with text + shapes so both OCR and detection have something to chew on."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 300, 220], fill=(70, 130, 180))
    d.ellipse([360, 90, 560, 260], fill=(210, 90, 60))
    d.text((70, 300), "HELLO AZURE CONTAINER APPS", fill=(20, 20, 20))
    d.text((70, 340), f"load-test-{random.randint(1000, 9999)}", fill=(20, 20, 20))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        opts = self.environment.parsed_options
        self.endpoint = opts.endpoint
        if opts.image:
            with open(opts.image, "rb") as f:
                self.payload = f.read()
        else:
            self.payload = _synthetic_image()

    @task(10)
    def infer(self):
        files = {"file": ("sample.jpg", self.payload, "image/jpeg")}
        with self.client.post(self.endpoint, files=files, name=self.endpoint,
                              catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")

    @task(1)
    def health(self):
        self.client.get("/healthz", name="/healthz")
