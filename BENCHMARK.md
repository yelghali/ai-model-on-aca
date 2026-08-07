# Benchmark findings — 2-hour soak, three deployments

Measured 2026-08-01 on Azure Container Apps, `swedencentral`, ACA environment
`aca-model-demo-env`. Every number below comes from
[`loadtest/soak.py`](loadtest/soak.py) and the raw CSVs in [`loadtest/`](loadtest/);
nothing is extrapolated.

> **Read [Caveats](#caveats--read-this-before-quoting-any-number) before quoting anything.**
> The document workload is synthetic and deliberately easy, and one published figure
> was wrong (see [Cost](#cost)).

---

## Summary

| | **YOLO `yolov8n`** | **PP-StructureV3 CPU** | **PP-StructureV3 GPU** |
|---|---:|---:|---:|
| Compute | 2 vCPU / 4 GiB | 4 vCPU / 8 GiB | NC8as-T4 (8 vCPU / 56 GiB) |
| Billing rate | $0.000060/s | $0.000120/s | $0.000090/s |
| Run duration | 2.00 h | 2.00 h | 2.00 h |
| Batches sent | 1,047 | 177 | 274 |
| **Images handled** | **5,371** | 640 sent / **417 processed** | **967** |
| Succeeded / failed | 1,047 / **0** | 140 / **37** (21%) | 274 / **0** |
| Pages verified correct | 5,371 | 417 | 967 |
| **Wrong (contamination)** | **0** | **0** | **0** |
| **Inference per image** (mean) | **104 ms** | **11,170 ms** | **855 ms** |
| End-to-end per image (mean) | 120 ms | 12,288 ms | 1,564 ms |
| **Cold start** (median) | **2.1 s** | **25.6 s** | **31.7 s** |
| Instances created | 10 | 4 | 5 |
| Peak replicas | 6 | 2 | 3 |
| Replica-seconds | 10,042 | 9,314 | 9,172 |
| Utilisation (busy / allocated) | 5.6% | 50.0% | 9.0% |
| Total cost | $0.60 | $1.12 | $0.83 |
| **$ / 1k images processed** | **$0.11** | **$2.68** | **$0.85** |
| $ / 1k at 100% utilisation | $0.006 | $1.34 | $0.077 |
| RSS at rest | 430 MB | 2,686 MB | — |

Total: **$2.55**, 6,978 images sent, 6,755 verified correct, **0 cross-contamination**.

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
| YOLO | **104** | 90 | 156 | 169 | 190 | 204 ms |
| PP-StructureV3 CPU | **11,170** | 9,769 | 15,521 | 16,963 | 17,404 | 17,417 ms |
| PP-StructureV3 GPU | **855** | 856 | 882 | 889 | 903 | 920 ms |

**End-to-end** — client-side wall time, includes queue wait, network and ACA ingress:

| | mean | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| YOLO | **120** | 102 | 178 | 195 | 270 | 588 ms |
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

## Cold start — the number that decides your scaling strategy

| | min | median | max | Weights in RAM |
|---|---:|---:|---:|---:|
| YOLO `yolov8n` | 2,062 | **2,122** | 2,242 ms | 430 MB |
| PP-StructureV3 CPU | 24,496 | **25,642** | 26,074 ms | 2,686 MB |
| PP-StructureV3 GPU | 29,781 | **31,709** | 33,182 ms | — |

**YOLO can safely run at `min-replicas 0`.** A 2.1 s cold start hides behind a retry.
Measured first request after a 12.9 min idle gap: **513 ms** — the replica was already back.

**PP-StructureV3 cannot, if latency matters.** 26-32 s means the first user after a quiet
period waits half a minute. Four models must load and warm, and on GPU CUDA context
creation is added on top — which is why the **GPU cold start is the slowest of the three**
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

## Correctness — 0 contamination in 6,755 verified pages

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
[`loadtest/soak-cpu.csv`](loadtest/soak-cpu.csv), [`loadtest/soak-gpu.csv`](loadtest/soak-gpu.csv).
