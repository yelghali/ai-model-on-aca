# Benchmark findings — 2-hour soak, four deployments

Measured 2026-08-01 (YOLO PyTorch, PP-StructureV3 CPU/GPU) and 2026-08-07 (YOLO ONNX) on
Azure Container Apps, `swedencentral`, ACA environment `aca-model-demo-env`. Every number
below comes from [`loadtest/soak.py`](loadtest/soak.py) and the raw CSVs in
[`loadtest/`](loadtest/); nothing is extrapolated.

> **Read [Caveats](#caveats--read-this-before-quoting-any-number) before quoting anything.**
> The document workload is synthetic and deliberately easy, and one published figure
> was wrong (see [Cost](#cost)).

---

## Summary

| | **YOLO PyTorch** | **YOLO ONNX** | **PP-StructureV3 CPU** | **PP-StructureV3 GPU** |
|---|---:|---:|---:|---:|
| Compute | 2 vCPU / 4 GiB | 2 vCPU / 4 GiB | 4 vCPU / 8 GiB | NC8as-T4 (8 vCPU / 56 GiB) |
| Billing rate | $0.000060/s | $0.000060/s | $0.000120/s | $0.000090/s |
| Run duration | 2.00 h | 2.00 h | 2.00 h | 2.00 h |
| Batches sent | 1,047 | 2,697 | 177 | 274 |
| **Images handled** | 5,371 | **15,140** | 640 sent / **417 processed** | **967** |
| Succeeded / failed | 1,047 / **0** | 2,697 / **0** | 140 / **37** (21%) | 274 / **0** |
| Pages verified correct | 5,371 | 15,140 | 417 | 967 |
| **Wrong (contamination)** | **0** | **0** | **0** | **0** |
| **Inference per image** (mean) | **104 ms** | **109 ms** | **11,170 ms** | **855 ms** |
| Inference p90 | 156 ms | **118 ms** | 15,521 ms | 882 ms |
| End-to-end per image (mean) | 120 ms | 132 ms | 12,288 ms | 1,564 ms |
| **Cold start** (median) | 2,122 ms | **181 ms** | 25,642 ms | 31,709 ms |
| Image size in ACR | 528 MB | **131 MB** | 937 MB | **6.05 GB** |
| RSS at rest | 430 MB | **131 MB** | 2,686 MB | — |
| Instances created | 10 | 16 | 4 | 5 |
| Peak replicas | 6 | 7 | 2 | 3 |
| Replica-seconds | 10,042 | 17,804 | 9,314 | 9,172 |
| Utilisation (busy / allocated) | 5.6% | 9.3% | 50.0% | 9.0% |
| Total cost | $0.60 | $1.07 | $1.12 | $0.83 |
| **$ / 1k images processed** | $0.11 | **$0.07** | **$2.68** | **$0.85** |
| $ / 1k at 100% utilisation | $0.006 | $0.007 | $1.34 | $0.077 |
| **1 replica always on, per month** | **$47 - $158** | **$47 - $158** | **$95 - $315** | **$237** |

Total: **$3.62**, 22,118 images sent, 21,895 verified correct, **0 cross-contamination**.

### Cost of keeping 1 replica warm (to avoid cold start)

730 h/month = 2,628,000 seconds. ACA bills a replica held at `min-replicas` at the
**idle** rate when it is not processing, and the **active** rate when it is — so the true
figure sits between these bounds depending on your duty cycle.

| | Idle rate | Always-on **idle** | Always-on **active** | Cold start avoided |
|---|---:|---:|---:|---:|
| YOLO PyTorch (2 vCPU / 4 GiB) | $0.000018/s | **$47/mo** | $158/mo | 2.1 s |
| YOLO ONNX (2 vCPU / 4 GiB) | $0.000018/s | **$47/mo** | $158/mo | 0.18 s |
| PP-StructureV3 CPU (4 vCPU / 8 GiB) | $0.000036/s | **$95/mo** | $315/mo | 25.6 s |
| PP-StructureV3 GPU (T4) | **none** | **$237/mo** | $237/mo | 31.7 s |

Two conclusions worth acting on:

**Serverless GPU has no idle tier.** NC T4 v3 bills a flat $0.00009/s whenever allocated,
so a warm GPU costs **$237/month doing nothing** — and it is the one deployment whose cold
start (31.7 s process, plus minutes of node allocation and a 6 GB image pull) most tempts
you to keep it warm. That tension is the central cost decision for GPU on ACA.

**For YOLO ONNX, paying to avoid cold start is pointless.** A 181 ms start is invisible
behind a retry. Spending $47/month to dodge 0.18 s is not a trade worth making — run it at
`min-replicas 0`. For PP-StructureV3 CPU the same $95/month buys you 25.6 s, which may well
be worth it.

---

## Method

Each deployment ran for two hours against randomised traffic cycling through four
phases chosen at random:

| Phase | Duration | Batch size | Gap | Concurrency |
|---|---|---|---|---|
| `burst` | 2-5 min | 4-8 | 0-1 s | 3 |
| `steady` | 5-12 min | 2-4 | 3-12 s | 1 |
| `low` | 5-15 min | 1-2 | 30-90 s | 1 |
| `idle` | 6-20 min | — | — | 0 |

The `idle` gaps are the point: they let ACA scale to zero so the *next* request pays a
real cold start. Every run started from an identical `min-replicas 1` baseline, used
seed `4242`, and carried a `--budget-usd 25` hard stop.

Cold start is not inferred from the outside — each replica reports its own
`startup_ms` (process import → all models loaded and warmed) via `/stats`, and the
harness records the first sighting of every new replica.

---

## Latency per image

**Server-side inference** — excludes queueing and network:

| | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| YOLO PyTorch | **104** | 90 | 156 | 169 | 190 | 204 ms |
| YOLO ONNX | 109 | 111 | **118** | 121 | 129 | 190 ms |
| PP-StructureV3 CPU | **11,170** | 9,769 | 15,521 | 16,963 | 17,404 | 17,417 ms |
| PP-StructureV3 GPU | **855** | 856 | 882 | 889 | 903 | 920 ms |

**End-to-end** — client-side wall time, includes queue wait, network and ACA ingress:

| | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| YOLO PyTorch | **120** | 102 | 178 | 195 | 270 | 588 ms |
| YOLO ONNX | 132 | 131 | 148 | 155 | 205 | 6,218 ms |
| PP-StructureV3 CPU | **12,288** | 9,816 | 17,394 | 22,214 | 36,081 | 45,673 ms |
| PP-StructureV3 GPU | **1,564** | 918 | 2,811 | 3,043 | 3,852 | 4,291 ms |

Three observations:

**Mean exceeds median everywhere** — the distributions are right-skewed by bursts and
cold starts. For PP-StructureV3 CPU the mean (12.3 s) sits 25% above the median (9.8 s).
Use the mean for capacity and cost planning, p90/p95 for SLOs; the median alone
understates typical cost.

**GPU inference is remarkably flat** — mean 855 ms, p99 903 ms, max 920 ms: a spread of
~8%. Yet its end-to-end p90 is 2,811 ms, more than 3× the inference time. That gap is
almost entirely queue wait. The GPU replica is never the bottleneck; the batch is.

**On CPU the two track each other** (11,170 vs 12,288) because the model itself *is* the
bottleneck — there is little queueing left to explain.

---

## PyTorch vs ONNX Runtime — not faster, but far lighter

I expected ONNX Runtime to be 2-3× faster per image. **It is not.** Controlling for batch
size, the two are within noise of each other:

| Batch size | PyTorch ms/image | ONNX ms/image |
|---:|---:|---:|
| 1 | 146.2 | **110.5** |
| 2 | **100.9** | 109.3 |
| 3 | **88.2** | 109.8 |
| 4 | **98.4** | 109.8 |
| 5 | 104.1 | **108.3** |
| 6 | **101.3** | 109.5 |
| 7 | 107.7 | **109.0** |
| 8 | **109.8** | 109.2 |
| **mean** | **104.2** | 109.2 |

PyTorch is marginally *faster* at batches 3-6. The graph optimisations ONNX Runtime applies
(operator fusion, constant folding) do not beat what ATen already does for a 3.2 M-parameter
convnet on 2 vCPU. **The prediction was wrong and the data says so.**

Two genuine differences hide in that table:

**ONNX wins decisively at batch 1** — 110 ms vs 146 ms, 25% faster. That gap is *per-call
overhead*, not compute: Python dispatching through Ultralytics costs real time on a single
image and is amortised away by a batch. If your workload is one image per request, ONNX is
meaningfully faster.

**ONNX is far more predictable.** Its ms/image sits at ~109 regardless of batch size, while
PyTorch swings between 88 and 146. That shows up as p90: **118 ms vs 156 ms**. A static graph
with a fixed execution plan has less variance than an interpreter.

Where ONNX wins outright is everything except speed:

| | PyTorch | ONNX | |
|---|---:|---:|---|
| Cold start | 2,122 ms | **181 ms** | **11.7× faster** |
| Image in ACR | 528 MB | **131 MB** | **4.0× smaller** |
| RSS at rest | 430 MB | **131 MB** | **3.3× less** |
| Model copies for N threads | **N** (pool) | **1** (shared session) | |

Cold start is the one that changes system behaviour. Most of PyTorch's 2.1 s is simply
`import torch` — loading shared libraries and registering thousands of operators. The ONNX
container ships neither torch nor ultralytics and starts in under 200 ms.

That fed through to cost: **$0.07 vs $0.11 per 1k images**. Instances that appear in 181 ms
make scale-out nearly free, so the ONNX run reached 9.3% utilisation against PyTorch's 5.6%.
Part of that gap is luck — the ONNX run drew more `burst` phases (caveat 3) — so treat the
cost difference as directional, not exact.

**The architectural payoff is the shared session.** `InferenceSession.run()` is thread-safe,
so the `queue.Queue` model pool disappears: one copy of the weights serves every worker
thread. The pool in `yolo-demo` exists *only* because Ultralytics objects carry mutable
state. The price of dropping the framework is reimplementing letterboxing and NMS — about
60 lines of numpy in [`yolo-onnx-demo/app/main.py`](yolo-onnx-demo/app/main.py).

---

## Cold start — the number that decides your scaling strategy

| | min | median | max | Weights in RAM |
|---|---:|---:|---:|---:|
| YOLO ONNX | 160 | **181** | 224 ms | 131 MB |
| YOLO PyTorch | 2,062 | **2,122** | 2,242 ms | 430 MB |
| PP-StructureV3 CPU | 24,496 | **25,642** | 26,074 ms | 2,686 MB |
| PP-StructureV3 GPU | 29,781 | **31,709** | 33,182 ms | — |

A **175× spread** between fastest and slowest. This is the number that decides whether you
can run at `min-replicas 0`.

**Both YOLO variants can.** A 2.1 s cold start hides behind a retry; 181 ms is invisible.
Measured first request after a 12.9 min idle gap: **513 ms** (PyTorch), **499 ms** (ONNX) —
the replica was already back.

**PP-StructureV3 cannot, if latency matters.** 26-32 s means the first user after a quiet
period waits half a minute. Four models must load and warm, and on GPU CUDA context
creation is added on top — which is why the **GPU cold start is the slowest of the four**
despite having by far the fastest inference.

Measured first request after an idle gap on GPU: 12.2 min → 1,361 ms, 11.7 min → 3,134 ms,
11.5 min → 1,455 ms. These are *warm* because scale-down had not yet completed; a true
scale-from-zero pays the full 30 s.

There is also an infrastructure-layer cost the app never sees: allocating a GPU node and
pulling the multi-GB CUDA image took **several minutes** before the container started at
all. Budget for that when planning a serverless-GPU rollout.

Weights are baked into the image at build time by
[`paddle-demo/download_models.py`](paddle-demo/download_models.py). Without that, ~1 GB
would be fetched on first request and cold start would be far worse.

---

## Cost

### Correction to a previously published figure

`soak.py` divides total cost by images **sent**, not images **processed**. PP-StructureV3
CPU had 37 failed batches = **223 images that were never processed but still counted in the
denominator**. YOLO and GPU had zero failures and are unaffected.

| $/1k images | Originally reported | **Corrected** |
|---|---:|---:|
| YOLO | $0.11 | $0.11 ✅ |
| PP-StructureV3 CPU | $1.75 | **$2.68** ❌ (+53%) |
| PP-StructureV3 GPU | $0.85 | $0.85 ✅ |

*(The bug is still present in `soak.py`; the corrected value above was computed
separately from the raw CSV.)*

### These figures are upper bounds

The harness bills every replica-second at the **active** rate. ACA charges a reduced
**idle** rate for replicas held at min-replicas that are not processing. Utilisation was
low, so the real invoice is lower:

| $/1k processed | Upper (all active) | Lower (idle rate applied) |
|---|---:|---:|
| YOLO | $0.11 | **$0.04** |
| PP-StructureV3 CPU | $2.68 | **$1.74** |
| PP-StructureV3 GPU | $0.85 | **$0.85** |

**Serverless GPU has no idle discount.** Azure lists NC T4 v3 at a flat $0.00009/s with no
idle tier, while a CPU container drops from $0.000060/s to $0.000018/s when idle. A warm
GPU costs full price whether it is working or not — the strongest argument for
`min-replicas 0` on GPU despite the 32 s cold start.

### The number to use for capacity planning

Marginal cost at 100% utilisation, with no idle time:

| | $ / 1k images |
|---|---:|
| YOLO | **$0.0062** |
| PP-StructureV3 GPU | **$0.0770** |
| PP-StructureV3 CPU | **$1.3404** |

At full utilisation the **GPU is 17× cheaper per page than CPU** — a far wider gap than the
observed 3× suggests, because the GPU sat idle 91% of the soak and still paid full rate.

### Verified rates

All confirmed against [Azure Container Apps pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/)
(pay-as-you-go):

| | Rate |
|---|---|
| vCPU active | $0.000024/s |
| Memory active | $0.000003/GiB-s |
| vCPU / memory idle | $0.000003/s |
| NC T4 v3 | $0.00009/s |
| NC A100 v4 | $0.000651/s |
| Requests | $0.40/million (first 2M free) |

Monthly free grant, excluded from all figures above: 180,000 vCPU-s, 360,000 GiB-s,
2M requests.

---

## Why is the GPU cheaper than the CPU here?

Two independent effects that multiply.

**1. The GPU profile is a flat-rate bundle, and it is priced well below its parts.**

```
CPU container  4 vCPU + 8 GiB  = 4 × $0.000024 + 8 × $0.000003  = $0.000120/s
GPU profile    NC8as-T4 (flat)                                  = $0.000090/s
```

An NC8as-T4 node is **8 vCPU + 56 GiB + 1 T4**. The same CPU and memory priced à la carte
on the Consumption profile would be `8 × $0.000024 + 56 × $0.000003 = $0.000360/s` —
**4× the actual GPU price, with the T4 included at no extra cost.**

**2. It is 13× faster, so you rent it for 1/13 as long.**

```
CPU  11,170 ms × $0.000120/s = $0.001340 per page  →  $1.34 / 1k
GPU     855 ms × $0.000090/s = $0.000077 per page  →  $0.077 / 1k
```

Cheaper per second *and* far fewer seconds. Normally an accelerator costs more per second
and you gamble that speed compensates; for this workload there is no gamble to take.

Whether an A100 would help is doubtful: at $0.000651/s it must be ~7× faster than the T4
merely to break even, and PP-StructureV3's GPU inference is already flat (p99 within 6% of
the mean) with much of the wall time in serial pipeline stages a larger GPU cannot overlap.

---

## Reliability — the CPU deployment sheds load under burst

37 of 177 CPU batches failed. The breakdown is specific:

| Status | Count | Meaning |
|---|---:|---|
| 503 Service Unavailable | 22 | no healthy replica to route to |
| 507 Insufficient Storage | 11 | rejected at the edge |
| 500 | 4 | |
| Connection-level errors | **0** | every failure was an HTTP *response* |

**36 of the 37 occurred during `burst` phases**, on the largest batches (6-8 images). The
surviving replica reported `restartCount = 0` and logged only `200 OK`, so these were
rejected at the ACA edge and never reached model code. Nothing crashed; the platform
refused work it could not place.

The mechanism is arithmetic. A CPU batch of 8 pages takes ~90 s. A burst queues far faster
than replicas can be added, and each new replica needs a further **26 s** to load models
before it can help. The GPU, at 855 ms/page, drained each burst before a queue could build
and **failed nothing in 274 batches**.

The lesson is not "CPU is bad" but that **slow inference and slow cold start multiply**:
autoscaling cannot rescue you inside a burst. Keep warm replicas, cap batch size, or make
the request faster.

---

## Correctness — 0 contamination in 21,895 verified pages

Every image carries a marker unique to itself, and the assertion is on **identity, not
accuracy**:

- **PP-StructureV3** — the page's own OCR token must appear in its own returned text.
  The token alphabet excludes visually ambiguous glyphs (`0/O`, `1/I`, `5/S`, `8/B`),
  because an earlier run reported 21/24 "wrong" that turned out to be the OCR reading
  `0` as `O` — each result *had* matched its own image.
- **YOLO** — the largest returned box must be centred where *that* image's ellipse was
  drawn. If the model detects nothing at all the page counts as `no_detection`, never as
  contamination. Over 5,371 images `no_detection` was **0**.

**0 wrong pages** across all three deployments, including 1,047 concurrent YOLO batches at
peak 6 replicas. The per-request model pool holds under sustained multi-hour concurrency.

---

## Caveats — read this before quoting any number

**1. The document workload is synthetic and deliberately easy.** The benchmark page is
1240×1754 with **11 text lines / ~621 characters**, cleanly rendered, with the bottom ~40%
blank. A real scanned page carries 40-60 lines / 2,000-3,000 characters plus noise, skew
and JPEG artefacts. Recognition cost scales with the number of detected text crops, so a
dense real page is plausibly **4-6× more work**.

**2. Most of PP-StructureV3 is switched off.** The pipeline runs layout + OCR only:

```python
use_table_recognition        = False
use_chart_recognition        = False
use_formula_recognition      = False
use_doc_orientation_classify = False
use_doc_unwarping            = False
use_textline_orientation     = False
```

Four models actually load — `PP-DocBlockLayout`, `PP-DocLayout_plus-L`,
`PP-OCRv5_server_det`, `PP-OCRv5_server_rec`. The default pipeline runs more.

Together, (1) and (2) mean **855 ms/page is a floor, not a typical figure.** If you have
seen PP-StructureV3 take ~1 minute per page elsewhere, the difference is most likely the
full pipeline on a dense real page — or a first call that included model loading (our own
cold start is 25-32 s even with weights pre-baked), or a CPU-only `paddlepaddle` wheel
silently not using the GPU. The *ratios* here should roughly hold, since both CPU and GPU
ran the identical workload; the absolute latencies should not be quoted as
"PP-StructureV3 page latency".

**3. The same seed did not produce identical traffic.** Phase length is wall-clock, but the
number of batches inside a phase depends on how fast the server answers. A faster server
fits more batches, consumes more random numbers, and the RNG stream diverges after the
first phase. YOLO received 8× more traffic than PP-StructureV3 CPU not because it was
favoured but because it could absorb it. **Compare per-image metrics, not totals.**

**4. CPU vs GPU is not a clean single-variable comparison.** The T4 profile carries 8 vCPU
and 56 GiB; the CPU container had 4 vCPU and 8 GiB. Part of the GPU's resilience under
burst is memory headroom, not the accelerator. The per-page *speed* difference is genuinely
the GPU.

**5. PP-StructureV3 CPU latency describes only the requests that were served.** 21% of its
batches got a 5xx instead of a slow answer, so the user-visible picture is worse than the
latency table alone suggests.

---

## Reproduce

```powershell
cd loadtest
python soak.py --host https://<yolo-fqdn>       --model yolo   --profile cpu-2  --app yolo-demo       --hours 2 --budget-usd 25 --seed 4242 --out soak-yolo
python soak.py --host https://<yolo-onnx-fqdn>  --model yolo   --profile cpu-2  --app yolo-onnx-demo  --hours 2 --budget-usd 25 --seed 4242 --out soak-yolo-onnx
python soak.py --host https://<paddle-cpu-fqdn> --model paddle --profile cpu-4  --app paddle-demo-cpu --hours 2 --budget-usd 25 --seed 4242 --out soak-cpu
python soak.py --host https://<paddle-gpu-fqdn> --model paddle --profile gpu-t4 --app paddle-demo-gpu --hours 2 --budget-usd 25 --seed 4242 --out soak-gpu
```

Each run writes `<out>.csv` (one row per request) and `<out>.json` (summary).
**Scale everything back to zero afterwards** — a warm T4 costs ~$0.32/h:

```powershell
cd deploy
./scaledown.ps1
```

Raw data for the runs above: [`loadtest/soak-yolo.csv`](loadtest/soak-yolo.csv),
[`loadtest/soak-yolo-onnx.csv`](loadtest/soak-yolo-onnx.csv),
[`loadtest/soak-cpu.csv`](loadtest/soak-cpu.csv), [`loadtest/soak-gpu.csv`](loadtest/soak-gpu.csv).
