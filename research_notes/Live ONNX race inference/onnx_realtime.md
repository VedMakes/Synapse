# Real-time ONNX inference for a fixed-tick racing evaluator

**Research date:** 2026-10-23  
**Scope:** ONNX Runtime Python inference in a Python-authoritative live/fixed-tick racing loop.  The goal is to improve latency behavior, deadline handling, measurement, and CPU repeatability without changing the model's existing input/output contract.

## Executive findings

1. **Keep the authoritative physics clock separate from wall-clock inference time.** Advance simulation by integer ticks and fixed `dt`; use wall time only to pace a live run and to enforce an explicitly agreed inference deadline. F1TENTH's environment follows this shape: `F110Env.step()` advances by its configured `timestep`, while its human renderer adds a sleep only for presentation. The closest ONNX racing comparator, `B3rK-3/ml-racing`, scores on simulated time and keeps rendering/inference wall time out of the score.
2. **For a small CPU policy with a batch-one contract, use one pre-created `InferenceSession`, one synchronous `session.run()` per control tick, explicit CPU provider, explicit one-thread settings, and a warm-up phase.** I/O binding is primarily valuable when the graph runs on a non-CPU execution provider or when CPU/device copies dominate. It is not automatically a win for a `[1, N] -> [1, 2]` CPU policy whose tensors are already NumPy arrays.
3. **Make tail latency the acceptance metric.** Record p50/p95/p99/p99.9/max, deadline misses, and end-to-end control-cycle time. Average latency alone can hide rare stalls that break a live race. ONNX Runtime's built-in profiling is useful for diagnosis, but should be disabled during the normal race because it adds measurement overhead and produces trace files.
4. **A Python timeout around a thread/future is not a hard inference deadline.** The native ORT call may continue after the Python caller stops waiting, and reusing the session can create overlap or resource contention. A real wall-clock cutoff needs either cooperative cancellation (`RunOptions.terminate`, tested carefully) or, for untrusted/strict competition runs, a separate worker process that can be killed and restarted. The existing Synapse isolated worker already uses the stronger process/container pattern.
5. **Determinism is a deployment profile, not one flag.** `use_deterministic_compute=True` helps, but repeatability also depends on fixed ORT version, graph optimization settings, execution provider, CPU feature set, thread counts, BLAS/OpenMP environment, input bytes, and action validation. One intra-op and one inter-op thread plus `ORT_SEQUENTIAL` is a sensible baseline for small policies; benchmark any wider configuration against p99 and repeated trajectory hashes.
6. **Do not batch the official control path merely to raise throughput.** The current ONNX profile is statically batch-one. Batching several entrants changes the scheduling/latency problem, can make one entrant wait for others, and requires a compatible model shape or contract change. Batch only in an offline throughput benchmark or when the already-frozen contract explicitly supports it.
7. **The safest first implementation direction is to preserve the current Synapse profile and strengthen observability/deadline policy around it.** Keep the model boundary and Python-authoritative physics unchanged; add/retain warm-up, monotonic latency accounting, bounded worker behavior, and an explicit “deadline miss => deterministic DNF/failure” rule rather than silently holding stale actions.

## Scope and terminology

- **Physics tick:** one fixed simulation step, e.g. `dt = 10 ms` / 100 Hz.
- **Control tick:** the cadence at which the policy is called, e.g. 25 Hz with its action held over four physics ticks.
- **Inference latency:** preferably measured as `session.run()` wall time after input preparation and before output validation. Also measure end-to-end policy-call time and full control-cycle time separately.
- **Deadline:** a wall-clock completion deadline for one policy call. It is not the same thing as the simulated race timeout.
- **Deterministic evaluation:** same frozen model, inputs, runtime, and environment produce the same validated actions and trajectory/replay hash. This is stronger than “the kernel selected deterministic algorithms” and weaker than a promise that outputs are bitwise equal across every CPU architecture.

## Source index

### Official ONNX Runtime documentation

| Source | What it contributes |
|---|---|
| [Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) | CPU defaults, intra-op vs inter-op semantics, sequential vs parallel execution, thread spinning, affinity, NUMA, and `session.dynamic_block_base`. |
| [I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) | Why CPU↔device copies can dominate non-CPU inference and how to bind device inputs/outputs. |
| [Python API summary](https://onnxruntime.ai/docs/api/python/api_summary.html) | `InferenceSession`, `SessionOptions`, `IOBinding`, `RunOptions`, `run_with_iobinding`, `run_async`, `terminate`, `use_deterministic_compute`, and provider precedence. |
| [Graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html) | Basic/extended/layout optimization levels and offline optimized-model serialization; warns that offline artifacts must match target hardware/options. |
| [Profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) | In-code JSON profiling, operator/thread timing, `onnxruntime_perf_test`, and Perfetto/Chrome tracing. |
| [Performance troubleshooting](https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html) | CPU/GPU placement issues and the `session.dynamic_block_base` option for reducing latency variance in some multithreaded workloads. |
| [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | CUDA I/O binding, custom streams, async synchronization, CUDA Graph constraints, warm-up cost, and fixed addresses/shapes. |
| [Device tensors](https://onnxruntime.ai/docs/performance/device-tensor.html) | Keeping tensors on-device, pinned memory, asynchronous copies, and when device-local buffers matter. |
| [ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) | Offline conversion and fixed/runtime optimization styles; model/runtime version compatibility tradeoffs. |
| [Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) | Static vs dynamic quantization; dynamic activation-scale computation adds inference work, while CPU speedups depend on hardware instructions and accuracy checks. |
| [Session option config keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h) | Canonical names and semantics for spinning, dynamic block sizing, denormal handling, thread affinity, and other session controls. |

### Open-source repositories reviewed

| Repository | Relevant files/patterns | Usefulness and caveat |
|---|---|---|
| [B3rK-3/ml-racing](https://github.com/B3rK-3/ml-racing) | [`README.md`](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/README.md), [`ghost_policy.py`](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/examples/ghost_policy.py), [`ghost_race.py`](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/examples/ghost_race.py), [`agents.md`](https://raw.githubusercontent.com/B3rK-3/ml-racing/main/agents.md) | Closest conceptual comparator: fixed policy contract, 25 Hz policy over 100 Hz physics, independent ghost worlds, simulation-time scoring, CPU ORT with one intra/inter-op thread. It explicitly documents **no hard per-inference wall-clock deadline**, so it is a useful timing reference and a clear example of the gap a strict live evaluator must address. |
| [f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) | [`f110_env.py`](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/f110_env.py), [`base_classes.py`](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/gym/f110_gym/envs/base_classes.py), [`waypoint_follow.py`](https://raw.githubusercontent.com/f1tenth/f1tenth_gym/main/examples/waypoint_follow.py) | Canonical open racing simulator. `timestep` is explicit, `step()` advances simulation time by that value, and rendering is separate. The base simulator holds controls while integrating physics, which is a good model for policy action hold intervals. |
| [autorope/donkeycar](https://github.com/autorope/donkeycar) | [`README.md`](https://github.com/autorope/donkeycar/blob/main/README.md), [`vehicle.py`](https://raw.githubusercontent.com/autorope/donkeycar/main/donkeycar/vehicle.py), [`keras.py`](https://raw.githubusercontent.com/autorope/donkeycar/main/donkeycar/parts/keras.py) | Python-authoritative vehicle loop and learned-pilot integration. It provides a practical parts pipeline, optional threaded camera parts, a target-rate loop, and percentile profiling. It warns that actual frequency can be lower when a part blocks; that is acceptable for a hobby vehicle loop but should be a failure/diagnostic condition in a competition with hard control deadlines. |
| [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples) | [`python/api/device_bindings.py`](https://raw.githubusercontent.com/microsoft/onnxruntime-inference-examples/main/python/api/device_bindings.py), [`python/api/README.md`](https://raw.githubusercontent.com/microsoft/onnxruntime-inference-examples/main/python/api/README.md) | Maintainer-owned examples for `OrtValue`, direct device tensors, I/O binding, and DLPack. The example creates a session once per demonstration and binds GPU tensors directly; it illustrates where I/O binding pays off, not an authoritative fixed-tick scheduler. |
| [AmineChr54/RL-autonomous-racing](https://github.com/AmineChr54/RL-autonomous-racing) | [`README.md`](https://raw.githubusercontent.com/AmineChr54/RL-autonomous-racing/main/README.md), [`main.py`](https://raw.githubusercontent.com/AmineChr54/RL-autonomous-racing/main/main.py) | Demonstrates vectorized racing and an ORT CPU/DirectML benchmark. It batches many tiny brains for training/benchmark throughput and observes that CPU wins for small populations while an accelerator can win for large batches. This supports keeping official one-car control latency separate from throughput-oriented training. |

## What the current Synapse implementation already establishes

These are local observations, not external recommendations; they were read to make the external research applicable without editing the codebase.

- [`racing_event/onnx_artifact.py`](../../racing_event/onnx_artifact.py) defines a restricted static graph profile: one static batch-one input named `observations`, one output named `actions`, float32 boundary tensors, standard ops only, no external data, no custom functions, bounded node/parameter/tensor counts, and a pinned `RUNTIME_VERSION = "1.20.1"`.
- Its session baseline is explicit: `intra_op_num_threads = inter_op_num_threads = 1`, `ORT_SEQUENTIAL`, `ORT_DISABLE_ALL`, CPU memory arena disabled, memory pattern disabled, and `session.use_deterministic_compute = "1"`. The probe calls the same input twice and rejects a different output. This is a strong repeatability baseline for a small restricted policy.
- Its inference path uses ordinary `session.run(["actions"], {"observations": values})`, not I/O binding. That is a reasonable default for CPU tensors and a two-element CPU output. I/O binding should be introduced only after measurement shows copy/conversion cost or after moving the model to a device EP.
- [`racing_event/isolated_policy.py`](../../racing_event/isolated_policy.py) is the strongest existing deadline pattern: a separate Docker worker, `--cpus=1`, memory/pid/file limits, network disabled, read-only filesystem, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, monotonic startup/action timeouts, bounded newline-delimited messages, sequence matching, and worker teardown after any exception/timeout. The process boundary is what makes a timeout enforceable.
- [`racing_event/evaluation.py`](../../racing_event/evaluation.py) is Python-authoritative and calls the policy synchronously before each environment step. `max_steps` is a simulated-step limit; trusted-local evaluation does not itself impose a per-call wall deadline. This keeps scoring deterministic but means a hung local model can hang the evaluator.
- [`racing_event/race.py`](../../racing_event/race.py) records whether inference was trusted-local or isolated and persists the runtime/deadline configuration in the race report. This is the right place conceptually for deadline/miss accounting, while leaving the model contract untouched.
- [`racing_event/model_artifact.py`](../../racing_event/model_artifact.py) allows generic NumPy artifacts to accept a batch up to 64, but the ONNX profile is explicitly batch-one. Do not transfer the generic artifact's batching allowance into ONNX evaluation without a contract decision.

## ONNX Runtime patterns and tradeoffs

### 1. Session creation and lifecycle

**Pattern:** create one session per frozen model before tick zero; validate its provider, inputs, outputs, and session options; run representative warm-up inputs; then start the race clock.

- Never load `InferenceSession` inside the control tick. Session creation includes model parsing, graph partitioning/optimization, allocator setup, constant prepacking, and possibly provider compilation.
- First-run latency can be higher even after session creation. CUDA Graph documentation explicitly calls out first-run allocation/capture cost; CPU models can also pay one-time allocator/prepacking/page-fault costs. Warm-up should exercise every output path and representative tensor shape.
- Keep model bytes/session alive for the full race. Do not rely on per-tick garbage collection or allocate a new input dictionary/model object unnecessarily.
- Pin and record: ORT version, model/artifact SHA-256, providers, graph optimization level, execution mode, intra/inter-op counts, memory arena/pattern settings, relevant environment variables, CPU model/flags, and process affinity/cgroup.
- If a model is loaded from bytes, keep the bytes alive if using any “use model bytes directly” option. The current Synapse loader avoids this complexity by validating bytes and creating the session directly without a long-lived zero-copy promise.

**Tradeoff:** a long-lived session improves latency and avoids startup variance, but session state/configuration becomes part of the reproducibility record. Recreating the session per entrant is fine before scoring; recreating it per tick is not.

### 2. CPU thread/session configuration

Recommended baseline for a small policy and one active call at a time:

```python
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.use_deterministic_compute = True
session = ort.InferenceSession(
    model_bytes,
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
```

The current Synapse ONNX profile is stricter by also disabling graph optimizations, CPU arena, and memory pattern. Treat that as the repeatability-first baseline.

- **Intra-op threads:** parallelism inside an operator. ORT defaults can use physical cores; explicit `1` prevents one policy call from consuming the machine and makes reduction order/scheduling simpler.
- **Inter-op threads:** parallelism across independent graph nodes. It matters with `ORT_PARALLEL`; it does not add value for a sequential tiny MLP.
- **`ORT_SEQUENTIAL`:** fewer scheduling degrees of freedom and usually appropriate for a single-path control policy. A branched/large graph may benefit from `ORT_PARALLEL`; only accept it after p99 and repeatability tests.
- **Thread spinning:** ORT says spinning can reduce wake-up latency but consumes CPU. On a dedicated core it may improve a tight real-time loop; on a shared evaluator it can steal time from physics/rendering or other entrants. Benchmark both `session.intra_op.allow_spinning`/`session.inter_op.allow_spinning` enabled and disabled. For an isolated worker with one CPU and infrequent calls, disabling spin is the more resource-predictable default; for a continuously active dedicated worker, a bounded spin duration can be tested.
- **Dynamic block sizing:** `session.dynamic_block_base=4` can reduce latency variance for some multithreaded models, per ORT troubleshooting docs. It is irrelevant or nearly irrelevant with intra-op `1`; do not add it just because it exists.
- **Affinity/CPU isolation:** one ORT thread count does not prevent the OS from migrating the Python process. If p99 matters, reserve a CPU via cgroup/cpuset or OS affinity and keep physics/inference resource allocation identical for every entrant. ORT's own affinity option is most useful for multi-thread sessions/NUMA, not a batch-one one-thread model.
- **Process-level math thread pools:** set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and equivalent MKL/BLIS variables before importing libraries in a worker. The existing isolated Docker command sets the first two. This prevents a lower-level library from silently creating a second thread pool.

**Tradeoff:** one thread can increase mean latency for a large CNN, but usually improves tail behavior and fairness for a small policy. A two-dimensional benchmark matrix (`intra_op` ∈ {1, 2, 4}, `execution_mode` ∈ {sequential, parallel}) is safer than assuming the fastest mean is the best live configuration.

### 3. Graph optimization and memory settings

- ORT's `ORT_ENABLE_ALL` is the documented default and can fuse/eliminate nodes. It normally improves inference performance, but fusions/layout changes can alter floating-point rounding and may be hardware-specific.
- Offline optimized models reduce session-start cost, but ORT warns that the same execution providers, optimization level, and compatible hardware must be used. An offline optimized artifact is not a portable replacement for the original contract unless the evaluator pins the target.
- `ORT_DISABLE_ALL` makes a restricted model's executed graph easier to reason about and is consistent with Synapse's current static profile. It can be slower. Compare it against `ORT_ENABLE_BASIC`/`ORT_ENABLE_EXTENDED`/`ORT_ENABLE_ALL` using the exact pinned runtime and a repeated trajectory check before changing the race profile.
- `enable_mem_pattern=True` and the CPU arena usually help repeated static-shape inference by reusing allocations. Disabling them can reduce allocator state/peak surprises and is defensible for a strict profile, but may increase allocation latency. Since Synapse has fixed static shapes and tiny output, benchmark these toggles; do not infer their effect from a different model.
- Do not enable `session.set_denormal_as_zero` solely for determinism. ORT's config-key comments warn it can hurt model accuracy. Use it only if the model is validated with that numerical policy.
- ORT format can bake in optimizations and reduce runtime/build size, but it is a deployment-format change rather than an inference API change. It is not needed for the current ONNX contract and should be considered only as a separately versioned artifact format.

### 4. I/O binding

#### CPU authoritative path

For CPU input/output already resident as contiguous NumPy arrays:

```python
action = session.run(["actions"], {"observations": observations})[0]
```

is the low-complexity baseline. The output is only `[1, 2]` in the current profile, so output-copy overhead is usually tiny relative to scheduling/validation and physics work. CPU I/O binding can still be benchmarked with an `OrtValue` or preallocated output, but it adds buffer-lifetime and binding-state complexity without the device-copy benefit emphasized by the official docs.

Safe CPU experiment:

- keep a contiguous float32 input buffer with the exact contract shape;
- bind the CPU input and a preallocated CPU output once if the Python API/version supports the required overloads;
- update the input buffer in place each tick;
- run `run_with_iobinding()` and compare full policy-call p50/p99 plus repeated output bytes;
- fall back to ordinary `run()` if any shape/provider/version behavior differs.

Do not reuse a mutable output buffer in a way that lets the physics loop observe a value while ORT is still writing it. Treat the call as synchronous unless the provider's synchronization semantics are explicitly managed.

#### GPU/accelerator path

I/O binding is materially more valuable when the input/output already live on CUDA/another device. Official docs explain that ordinary `Run()` can copy CPU input to the device and copy device output back to CPU; binding avoids those hidden copies. The device tensor docs add pinned-memory and asynchronous-copy patterns.

- Bind device-local input/output tensors and reuse their addresses.
- Keep shapes/addresses fixed if using CUDA Graphs. Update input contents in place rather than rebinding a new allocation.
- Warm up before timing; the first CUDA Graph run captures/allocates and is not representative.
- A custom compute stream and `disable_synchronize_execution_providers` can reduce synchronization, but the caller then owns stream synchronization. This is a poor first choice for a Python-authoritative deadline path.
- GPU/DirectML scheduling, copies, synchronization, and provider fallback make deterministic deadlines harder than a pinned CPU path. Use an accelerator only when measured CPU p99 cannot meet the control budget and the team accepts the additional reproducibility surface.

**Recommendation:** keep CPU `session.run()` for the current fixed batch-one CPU competition. Treat I/O binding as a measured optimization for a future device EP, not as a required ONNX contract change.

### 5. Latency measurement and profiling

Use a monotonic high-resolution clock (`time.perf_counter_ns()` or `time.monotonic_ns()` for deadline accounting), never wall-clock calendar time. Measure these layers independently:

1. observation snapshot/copy and input preparation;
2. ORT call (`session.run` or `run_with_iobinding`);
3. output validation/clipping and action conversion;
4. physics step(s) and sensor generation;
5. complete control-cycle wall time;
6. tick release jitter (`actual_release - scheduled_release`) in a paced live loop.

Protocol:

- Create/validate session before the race.
- Warm up with a representative fixed input until the first-run/allocator effect is gone; record warm-up count and times.
- Run at least several thousand control calls under the same CPU load, input shape, and action cadence as the live race.
- Report count, min, p50, p90, p95, p99, p99.9, max, standard deviation, deadline misses, and the number of calls exceeding one tick period. Store raw samples or a reproducible histogram, not only an average.
- Measure a cold-start phase separately: model parse/session creation, first call, and steady state.
- Run a repeated deterministic probe and, preferably, repeated complete evaluation. Compare output bytes or a trajectory hash; do not use approximate equality for a competition identity check unless the rules explicitly define a tolerance.
- Disable per-tick logging, JSON writes, rendering, and profiling in the timed path. Accumulate counters in memory and emit them at the end or from a non-authoritative reporter.

ORT's `SessionOptions.enable_profiling = True` produces a JSON trace with operator and threading timing and can be opened in Perfetto. Use it for diagnosis of a chosen session configuration, then rerun the acceptance benchmark with profiling off. `onnxruntime_perf_test` is useful for comparing thread/optimization knobs, but it is not a substitute for the real Python loop, IPC, observation preparation, or physics workload.

### 6. Fixed tick, scheduling, and action freshness

Recommended conceptual loop (no contract change):

```text
load and validate model/session
warm session
record live_start and tick_period
for tick = 0 .. max_ticks-1:
    scheduled_release = live_start + tick * tick_period
    snapshot = authoritative_state_at_tick(tick)
    deadline = scheduled_release + inference_budget
    action = infer(snapshot, deadline)
    if action misses deadline or is invalid:
        apply_the_agreed_failure_policy()
        break/mark entrant DNF
    apply action for the fixed control interval
    advance exactly the configured number of physics ticks
    if live mode: pace toward next scheduled release
```

Important details:

- Use an integer tick counter and `dt`; never advance simulation by the amount of wall time spent in inference.
- A late policy call must not make the evaluator silently “catch up” by changing `dt`, skipping physics, or awarding a different simulation time.
- Do not let rendering, replay encoding, or another entrant's inference feed back into the authoritative observation.
- In a multi-entrant race, serial calls provide straightforward fairness and avoid CPU contention. Parallel workers are possible only with explicit equal CPU/memory reservations and identical scheduling policy.
- If the engine is simulation-speed rather than wall-real-time, inference wall time should be a measured resource/deadline condition, not part of finish time. The B3rK comparator follows this separation.

### 7. Deadline enforcement

#### Cooperative in-process option

Python `RunOptions.terminate` is documented as a way to request termination of currently executing calls that use that `RunOptions`. A watchdog can set it, but this needs a small, version-pinned experiment:

- confirm the call returns promptly on the actual CPU EP and ORT version;
- confirm the session remains safe to close/recreate after termination;
- ensure the watchdog never leaves a still-running call sharing mutable input/output buffers;
- record the call as a miss/error and apply a deterministic race outcome.

Do not treat `future.result(timeout=...)`, `asyncio.wait_for`, or a Python thread timeout as a hard stop. They can stop waiting while native inference continues.

#### Strong option for untrusted/strict competition runs

Use the current Synapse `IsolatedPolicy` shape:

- frozen artifact snapshot;
- one model worker process/container;
- bounded stdin/stdout protocol with sequence numbers;
- `time.monotonic()` deadline in the parent;
- after timeout, malformed response, EOF, or invalid action: close/kill worker and mark the entrant failed/DNF;
- one inference in flight; never send a second request to a worker whose prior call may still be executing;
- record startup time, action timeout, worker exit reason, container image digest, CPU/memory limits, and miss count.

This is stronger than a thread because the OS can reclaim a stuck native call. It has IPC overhead, startup cost, operational complexity, and potentially larger p99; benchmark the full parent-to-action time, not only the worker's internal ORT call.

#### Failure semantics

For a fair live competition, choose the rule before freezing an event:

- **DNF/failure on miss:** simplest and most auditable; no stale action can keep a slow model moving.
- **Hold-last-action:** can be physically reasonable, but gives a slow model an implicit controller and must be part of the contract/rules. It also requires a bound on how long stale control may be held.
- **Safe neutral action:** e.g. zero/stop; deterministic but changes the vehicle trajectory and should be scored as an entrant failure or explicit safety behavior.

Do not switch among these after observing results. For the current strict evaluator, “deadline miss => entrant failure/DNF, no replacement controller” is the cleanest rule and aligns with existing invalid-inference handling.

### 8. Batching tradeoffs

| Situation | Likely result | Recommendation |
|---|---|---|
| One small policy, one `[1,N]` observation per control tick | Batch overhead and scheduling dominate; no meaningful throughput need | Keep batch one and ordinary `run()`. |
| One large shared model serving many independent agents, latency budget loose | Larger batch can improve arithmetic/weight reuse and CPU/GPU utilization | Benchmark only if the model contract already permits batch dimension and all agents can tolerate a common release barrier. |
| Several different participant models | Cannot generally combine into one ORT call without changing graph/model ownership | Use one session/worker per model, serially or with equal isolated resources. |
| GPU model with device-local tensors | Batch can amortize launches and improve throughput, but wait-to-fill adds latency | Use only with explicit freshness policy, fixed batch sizes, and p99 evidence. |
| Training/offline evaluation | Throughput is usually the goal | Batch freely in a separate benchmark/training path; do not use those numbers as live control latency. |

Batching introduces a **barrier**: the first observation waits for the rest of the batch. That raises input age and tail latency even if total examples/second improves. It also means a slow or unavailable entrant can delay other entrants unless the scheduler uses fixed-size dummy slots, which itself has fairness and resource implications. The current ONNX graph validation requires batch one, so official batching would be a model-contract change and is out of scope.

## Repository patterns in detail

### B3rK-3/ml-racing: closest race/evaluator analogue

The public README specifies a self-contained ONNX file, fixed `obs` float32 `[1,156]` input and `action` float32 `[1,2]` output, 25 Hz policy calls, 100 Hz physics, actions held across four 10 ms steps, independent ghost worlds, seeded observations, and simulation-time scoring. `ghost_policy.py` validates file size, rejects external tensor data, creates one CPU session, sets intra/inter-op to one, validates exact boundary metadata, checks finite actions, and clips the normalized action. `ghost_race.py` calls each policy once per control interval, then advances four physics steps, and uses wall time only to pace rendering.

The rules are especially useful because they are explicit about what is **not** enforced: no hard per-inference wall deadline, no process-memory quota, no sandbox, and no guarantee that a hung inference is interrupted. This is a good warning against assuming that a simulated timeout protects a live process. It also supports the current Synapse design choice to separate trusted-local preview from isolated evaluation.

### f1tenth/f1tenth_gym: physics tick separation

`F110Env` has an explicit `timestep` defaulting to 0.01 s, increments `current_time` by that value on every `step`, and returns the configured timestep as its reward. The renderer's human mode sleeps 5 ms after drawing; `human_fast` does not. `Simulator.step` applies the control input to each car, integrates dynamics with Euler or RK4, then performs scans/collision updates. This illustrates:

- keep the action interface at the control layer and hold it through a known number of physics updates;
- score from simulated ticks/time, not the time Python happened to spend in inference;
- keep rendering and diagnostic work outside the action/physics decision boundary;
- freeze the integrator and timestep because even a “same model contract” can produce incomparable races if physics changes.

### autorope/donkeycar: practical Python loop, not hard real-time

`Vehicle.start(rate_hz=10)` runs each part, then sleeps for the remaining period. If parts take longer, it logs a jitter warning but continues; the docstring explicitly says actual frequency may be lower when parts block. `PartProfiler` discards first/last samples and reports min/max/average and p50/p90/p99/p99.9. Threaded camera parts are allowed, while the policy part can remain in the main loop.

Useful pattern: profile each pipeline component and isolate asynchronous sensor acquisition from the decision step. Unsafe pattern for a strict competition: treating a max-rate loop plus warning as a deadline guarantee. Synapse should promote a missed control deadline to a recorded policy outcome, not only a log line.

### microsoft/onnxruntime-inference-examples: I/O binding evidence

`device_bindings.py` shows CPU/GPU provider selection, creation of device `OrtValue`s, binding input pointers, binding a device output, and direct binding to contiguous PyTorch tensors. The sample uses device-local buffers and optional DLPack, exactly the scenario where the official I/O binding documentation says copy avoidance matters. It does not establish that I/O binding accelerates CPU NumPy batch-one inference.

### AmineChr54/RL-autonomous-racing: throughput is a different objective

This repository simulates 200 cars with vectorized NumPy and benchmarks a whole-population ONNX forward pass on CPU vs DirectML. Its README reports that the tiny default population is faster on CPU because accelerator handoff costs dominate, while larger populations can favor the iGPU. This is a useful empirical reminder: hardware choice depends on workload size, and “more parallel” is not automatically lower control latency. Training and offline population evaluation can exploit batching; a one-car live policy should optimize age/tail latency instead.

## Recommended deployment profiles for Synapse

### Profile A: trusted local CPU preview

Use for development and deterministic local comparisons:

- one long-lived `InferenceSession` per model;
- CPU EP only;
- `intra_op=1`, `inter_op=1`, `ORT_SEQUENTIAL`;
- deterministic compute enabled;
- keep the current graph/memory settings until a benchmark justifies a change;
- warm-up before tick zero;
- `session.run()` with contiguous float32 batch-one input;
- monotonic latency histogram and deadline-miss counters;
- synchronous policy call; no threads/futures around the same session;
- repeat complete evaluation and compare trajectory hash.

This is the lowest-change path because it matches `racing_event/onnx_artifact.py` and `evaluation.py`.

### Profile B: isolated official evaluation/live competition

Use for participant-owned/untrusted models or an event that requires a hard wall bound:

- validate/snapshot artifact before start;
- start one pinned worker/container per entrant or a serial worker reused only by one entrant;
- enforce CPU/memory/pid/network/filesystem limits;
- set all math-library thread counts before ORT import;
- parent-side `monotonic` startup/action deadlines;
- bounded request/response with sequence IDs;
- kill/recreate worker after timeout or malformed output;
- no stale action or replacement controller unless rules explicitly say so;
- record worker version/image, deadline, misses, and reason in the race report;
- score from fixed simulation ticks, not elapsed inference wall time.

This matches the existing `isolated_policy.py` shape. The main tradeoff is IPC and worker-management overhead; include it in the measured deadline budget.

### Profile C: accelerator/device experiment

Only consider after CPU p99 fails the live budget:

- use `run_with_iobinding` with fixed device buffers;
- benchmark ordinary `run()` vs binding including all copies and synchronization;
- use a provider-specific latency setting, not throughput/multi-stream defaults;
- warm up and exclude first-run compilation/capture;
- disable provider fallback if silently moving work to CPU would violate the timing model;
- pin driver/runtime/provider versions and device identity;
- document that deterministic repeatability and deadline behavior may differ from CPU.

## Acceptance benchmark plan

For each candidate session profile, run the same frozen model and representative observation stream:

1. **Environment record:** ORT version, Python version, OS/kernel, CPU model/flags, process affinity, cgroup/container limits, provider list/options, ORT session options, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, model/artifact hashes.
2. **Cold phase:** session construction time, first call, first 10 calls, memory peak if available.
3. **Warm phase:** at least 10,000 policy calls or enough to cover the longest expected live race; alternate realistic observations rather than a single all-zero input.
4. **Load phase:** run the actual Python environment/physics loop, with rendering off and then with the intended live renderer if applicable. Test idle, normal, and CPU-contention conditions.
5. **Statistics:** p50/p95/p99/p99.9/max for ORT and end-to-end; tick jitter; deadline misses; worker restarts; action invalids; output hash/repeatability.
6. **Repeatability:** run the exact evaluation at least twice and compare action bytes/trajectory hashes. If hashes differ, preserve the first differing tick and inputs for diagnosis.
7. **Decision rule:** choose a profile that meets the p99/p99.9 budget and repeatability rule, not the profile with the best average. Keep a margin for Python input preparation, validation, physics, and scheduler jitter.

### Suggested latency budget accounting

For a control period `T_control`, reserve a measured budget rather than using all of it:

```text
T_control
  >= observation preparation
   + policy-call deadline
   + output validation/action conversion
   + required physics/sensor work
   + scheduler/jitter margin
```

The inference deadline should be an explicit constant agreed with the event rules. Do not derive it from the model's mean or from `T_control` alone. A model with p99.9 close to the whole period is not safely real-time even if its average is excellent.

## Recommendation matrix

| Change/pattern | Contract change? | Expected benefit | Risk/tradeoff | Recommendation |
|---|---:|---|---|---|
| Pre-create/warm one session | No | Removes startup/first-run spikes | Must record warm-up boundary | Apply. |
| Explicit CPU EP, one intra/inter thread, sequential mode | No | Lower contention and easier repeatability | May hurt large-model mean throughput | Apply as baseline; benchmark wider settings. |
| `use_deterministic_compute` | No | Requests deterministic kernels where supported | Not a cross-hardware bitwise guarantee | Apply and verify with repeated trajectories. |
| Keep `session.run()` on CPU | No | Simple, low risk for tiny output | May allocate/copy small arrays | Apply; benchmark I/O binding before changing. |
| CPU I/O binding/preallocated output | No | May reduce Python/allocator overhead | Buffer lifetime/API complexity; often negligible for `[1,2]` | Optional benchmark only. |
| GPU I/O binding/device tensors | No shape change, but provider change | Avoids device copies | Provider/driver/synchronization nondeterminism; deadline complexity | Only after CPU p99 fails. |
| `ORT_ENABLE_ALL` | No | Often faster due to fusions | Numerical/graph/runtime variability | Benchmark offline; keep current restricted profile if repeatability wins. |
| `enable_mem_pattern`/CPU arena | No | Reuse static allocations | Peak memory/state behavior changes | Benchmark with fixed model; do not assume. |
| `session.dynamic_block_base=4` | No | Lower variance for some multithreaded models | No value with one thread; extra knob | Skip for current one-thread baseline. |
| Batching entrants | Usually yes | Higher throughput | Queueing/input age, fairness barrier, shape/contract change | Do not use official path. |
| Python thread/future timeout | No | Can bound caller wait | Does not stop native inference safely | Do not use as hard deadline. |
| Process/container worker kill on timeout | No model change | Enforceable wall deadline and isolation | IPC/startup overhead and operations | Use for strict/untrusted runs. |
| Hold last action after timeout | Rules change | Physically smooth fallback | Slow model gets implicit control; stale data | Avoid unless frozen in event rules. |

## Bottom line for safe application

The highest-confidence changes that preserve existing model contracts are operational: load/warm once, keep the current batch-one CPU `session.run()` path, make thread/math settings explicit, use a monotonic latency histogram, keep the simulation clock integer/fixed, and use the existing isolated worker for an enforceable deadline. I/O binding and batching are workload-specific optimizations rather than defaults. Treat p99.9 deadline compliance and repeated trajectory hashes as release gates, and record every runtime/session/resource setting needed to reproduce the result.

## Additional references

- Python monotonic high-resolution timer: [`time.perf_counter_ns`](https://docs.python.org/3/library/time.html#time.perf_counter_ns)
- ONNX Runtime Python `RunOptions.terminate` and `SessionOptions.use_deterministic_compute`: [Python API summary](https://onnxruntime.ai/docs/api/python/api_summary.html)
- ORT configuration-key source for spinning, dynamic blocks, denormal behavior, and affinity: [`onnxruntime_session_options_config_keys.h`](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
- Current Synapse sources consulted: [`onnx_artifact.py`](../../racing_event/onnx_artifact.py), [`isolated_policy.py`](../../racing_event/isolated_policy.py), [`evaluation.py`](../../racing_event/evaluation.py), [`race.py`](../../racing_event/race.py), [`model_artifact.py`](../../racing_event/model_artifact.py)
