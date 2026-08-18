# Model serving concurrency and autoscaling

The model APIs use the same serving pattern: keep blocking inference off FastAPI's event
loop, isolate model wrappers that carry mutable state, bound in-process work, and align ACA's
HTTP scale rule with the real capacity of one replica.

## The three failure modes

### 1. Blocking the event loop

A naive asynchronous endpoint still blocks when it directly calls a CPU-bound model:

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = decode(await file.read())
    return model.predict(image)  # blocking work on the event-loop thread
```

While `predict()` runs, the process cannot accept another request or answer a health probe.
The APIs in this repository submit inference to an explicitly bounded executor:

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(state["executor"], _infer, image)
```

### 2. Sharing a stateful wrapper between threads

Ultralytics `YOLO`, EasyOCR `Reader`, and PaddleOCR's high-level pipeline wrappers retain
mutable per-call state. Concurrent calls through one instance can overwrite predictor input or
result buffers. Each worker therefore borrows an instance that no other worker can use:

```python
model = state["pool"].get()
try:
    return model.predict(image, verbose=False)
finally:
    state["pool"].put(model)
```

The executor size and pool size are identical. That invariant guarantees every running thread
owns one model and guarantees `pool.get()` does not become an unbounded second queue.

ONNX Runtime is the important exception in this repository. `InferenceSession.run()` supports
concurrent calls, so [the ONNX API](../yolo-onnx-demo/app/main.py) shares one session while still
offloading calls to a bounded executor. This removes duplicated model state and contributes to
its much lower 131 MB resting RSS.

### 3. Oversubscribing the CPU

PyTorch, ONNX Runtime, and Paddle can create their own native threads inside one inference.
Running several Python workers without capping those threads can create far more runnable work
than the container's vCPU limit.

The deployment sets worker and runtime thread counts explicitly. Do not size from
`os.cpu_count()` inside ACA; it can describe the host rather than the container limit.

| API | Outer concurrency | Inner-thread control | Model ownership |
|---|---|---|---|
| YOLO PyTorch | `INFERENCE_WORKERS` | `TORCH_THREADS_PER_WORKER` | one YOLO instance per worker |
| YOLO ONNX | `INFERENCE_WORKERS` | `ORT_INTRA_THREADS` | one shared session |
| EasyOCR | `INFERENCE_WORKERS` | `TORCH_THREADS_PER_WORKER` | one Reader per worker |
| PP-StructureV3 | `INFERENCE_WORKERS` | `CPU_THREADS_PER_WORKER` | one pipeline per worker |

As a starting constraint:

$$
\text{workers} \times \text{runtime threads per worker} \leq \text{container vCPU}
$$

## Match replica capacity to the ACA scale rule

There are two independent scaling axes:

| Axis | Mechanism | Purpose |
|---|---|---|
| Scale up | bounded executor and model pool/session | use the CPU already assigned to one replica |
| Scale out | ACA and KEDA HTTP concurrency | add replicas when one replica is full |

For these APIs, the default relationship is:

$$
\text{HTTP concurrentRequests per replica} = \texttt{INFERENCE\_WORKERS}
$$

If the scale threshold is higher than the worker count, requests queue inside a replica and ACA
does not add capacity soon enough. If it is lower, ACA creates replicas while existing workers
are idle. With two workers and ten replicas, the configured upper bound is 20 concurrent model
calls.

The relationship is implemented by [deploy/deploy.ps1](../deploy/deploy.ps1), which defaults
`-ConcurrentRequests` to `-InferenceWorkers`.

## How ACA HTTP scaling behaves over time

`concurrentRequests` is a target per replica, not a request-count quota and not
a timeout. ACA calculates the HTTP metric every 15 seconds from the preceding
15-second interval and uses:

$$
	ext{desired replicas} =
\left\lceil\frac{\text{current HTTP concurrency}}{\text{target concurrency}}\right\rceil
$$

For this pipeline, the target is 2 because each YOLO replica has two inference
workers:

| Concurrent `/predict` requests | Target replicas |
|---:|---:|
| 1-2 | 1 |
| 3-4 | 2 |
| 5-6 | 3 |
| 7-8 | 4 |

One `/predict` request can contain up to eight figures and still counts as one
HTTP request. ACA scales from request concurrency, not from the number of images,
pages, or documents inside a request.

The 15-second interval is a measurement window, not a process deadline. An
in-flight request remains open until FastAPI returns, the client disconnects, or
ingress times it out. Scale-down uses a 300-second stabilization window; scaling
the final replica to zero also has a 300-second cooldown after the HTTP load is
gone. The defaults therefore favor fast scale-out and deliberately slow
scale-in.

There are separate execution limits:

| Limit | Current pipeline | ACA maximum |
|---|---:|---:|
| Durable activity HTTP client | 120 seconds | application-controlled |
| Default ACA HTTP ingress | 240 seconds | fixed at 240 seconds |
| Premium ingress idle request timeout | not enabled | 30 minutes |
| Container shutdown grace after `SIGTERM` | 30 seconds | separately configurable |

Autoscaling is not a durability guarantee. Scale-in, revision deployment, or
platform maintenance can terminate a replica. ACA sends `SIGTERM` and then uses
the shutdown grace period before `SIGKILL`, so inference requests must be safe
to retry. `minReplicas = 1` removes scale-from-zero latency but does not prevent
restarts.

Official references:

- [ACA HTTP scaling](https://learn.microsoft.com/azure/container-apps/scale-app#http)
- [ACA scale behavior](https://learn.microsoft.com/azure/container-apps/scale-app#scale-behavior)
- [ACA ingress timeout](https://learn.microsoft.com/azure/container-apps/ingress-overview#protocol-types)
- [ACA container shutdown](https://learn.microsoft.com/azure/container-apps/application-lifecycle-management#shutdown)

## YOLO API, queue worker, or ACA Job

Keep synchronous `/predict` while one batched inference is comfortably below
the 120-second caller budget. It reuses an already-loaded model, returns results
directly to the Durable activity, and needs no second queue/result protocol.
Durable Functions remains the document job controller; HTTP is only the
invocation contract for model compute.

Switch to external queue-backed inference when one request routinely approaches
the timeout, has highly variable duration, or must survive caller disconnects:

```text
Durable activity -> input blob + queue message
ACA worker       -> YOLO -> result blob + completion message
Durable workflow -> external event, or durable timer plus status activity
```

A queue-scaled Container App is generally better for repeated inference because
one warm replica can process many messages. Use an ACA Job for coarse, isolated
work such as a whole dataset or scheduled batch. Starting one Job per small
figure chunk repeatedly pays container and model initialization cost.

Do not move PyTorch into the Durable Function App merely to remove HTTP. That
would couple model memory, cold start, deployment, and GPU requirements to the
orchestration tier. Change the transport to a queue when needed; keep the model
compute boundary on ACA.

## Prove concurrency from the server

Every API exposes `GET /stats` with separate counters:

| Field | Meaning |
|---|---|
| `in_flight` / `peak_in_flight` | requests accepted, including requests waiting for a worker |
| `running` / `peak_running` | requests currently inside model inference |
| `rss_mb` / `rss_baseline_mb` | process memory under load and after startup |

A large gap between `peak_in_flight` and `peak_running` is queueing. `peak_running` should never
exceed `INFERENCE_WORKERS`. The response from each inference also reports model time, total time,
queue wait, and the serving replica so client latency can be separated from model saturation.

## Measured evidence

With one YOLO PyTorch replica on 2 vCPU / 4 GiB, changing only the worker count produced:

| Workers | Requests | Concurrency | Correct | `peak_running` | p50 at concurrency 8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 8 | 24 / 24 | 1 | 1,326 ms |
| 2 | 24 | 8 | 24 / 24 | 2 | 229 ms |

The second worker greatly reduced queueing latency, but throughput improved by only about 20%
because one YOLO call already uses most of the 2 vCPU quota. Threads remove accidental
serialization; they do not create compute.

Scale-out was the throughput multiplier. At concurrency 24, measured throughput rose from 5.85
requests/s while scaling to 27.38 requests/s once five replicas were serving. Early scale-out
passes produced connection resets; two connect-only client retries produced 200 successful
responses out of 200. Production callers must retry transient connection failures without
blindly replaying requests that may already have reached inference.

EasyOCR behaved differently: two workers nearly doubled throughput because one inference did
not saturate both vCPUs. This is why worker count must be measured per model rather than copied
from YOLO.

The two-hour soaks verified zero cross-request contamination across 21,895 processed images for
the four primary deployments. See [the benchmark report](../BENCHMARK.md) for latency
percentiles, cold starts, failures, memory, cost, methodology, and caveats.

## Practical defaults

- Start with `workers x runtime threads <= vCPU`.
- Set ACA `concurrentRequests` equal to the worker count.
- Use one model wrapper per worker unless the runtime explicitly supports concurrent calls.
- Run [the correctness test](TESTING.md#1-correctness-under-concurrency) before load testing.
- Compare `peak_running`, queue wait, and inference time as concurrency rises.
- Use `min-replicas 0` for cost-sensitive YOLO deployments; keep a replica warm only when a
  measured startup delay violates the service objective.
- Scale every comparison app back to zero after testing.