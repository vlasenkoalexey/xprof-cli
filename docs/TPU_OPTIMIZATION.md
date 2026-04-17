# TPU Performance Optimization Guide

Practical guide to diagnosing and optimizing JAX/PyTorch-XLA/TensorFlow workloads on TPUs, synthesized from multiple public sources (see [Sources](#10-sources) below). Gotchas and process are emphasized.

---

## 1. TPU Hardware Basics

Understanding the hardware is prerequisite to interpreting profiles.

### Memory hierarchy

```
Registers (fastest)
    ↕
VMEM / scratchpad (on-chip, MB-scale)
    ↕
HBM (GBs, high-bandwidth — but much slower than VMEM)
    ↕
Host RAM (over PCIe)
```

### Peak numbers by generation

| Generation | Peak bf16 FLOPs/s | HBM bandwidth | Critical intensity |
|------------|-------------------|---------------|-------------------|
| TPU v5e    | 1.97e14           | 820 GB/s      | **240 FLOPs/byte** |
| TPU v5p    | ~5e14             | ~2.8 TB/s     | ~180              |
| TPU v6e (Trillium) | 9.1e14   | 1.6 TB/s      | ~570              |

**The 240 rule (v5e):** a `[B, D] × [D, F]` matmul in bf16 is compute-bound when `B > 240`. Below that, the chip is waiting on HBM.

### MXU tile sizes

- TPU v5e and earlier: **128×128** systolic array
- TPU v6e (Trillium) and later: **256×256**

Matrix dimensions must be multiples of these for full occupancy. Anything smaller wastes a significant fraction of the MXU.

---

## 2. Profiling Workflow

### Capture (once per change)

**JAX:**
```python
with jax.profiler.trace("/tmp/profiles", create_perfetto_link=False):
    # Run a few representative steps after warmup
    result = model_step(batch).block_until_ready()
```

**PyTorch/XLA:**
```python
import torch_xla.debug.profiler as xp
server = xp.start_server(9012)
# After model warmup:
xp.trace('localhost:9012', '/tmp/profiles/', duration_ms=2000)
```

**Critical:** always skip the first 1–3 steps (JIT/XLA compilation time distorts results). Profile after steady state.

### Serve with xprof

```bash
pip install xprof
xprof --logdir=/tmp/profiles --port=8791
# Visit http://localhost:8791
```

### AI-assisted analysis with xprof MCP

```bash
# Start the MCP server (once):
XPROF_URL=http://localhost:8791 \
XPROF_LOGDIR=/tmp/profiles \
python -m xprof_mcp.server.xprof_mcp_server --transport http

# Connect Claude Code:
claude mcp add --transport http --scope user xprof http://localhost:8792/mcp
```

### Investigation order

1. **`get_overview(run)`** — step time, host vs device split, bottleneck category
2. **`get_top_hlo_ops(run)`** — which ops consume the most time/FLOPs/memory
3. **`get_op_profile(run)`** — per-program breakdown (use when `hlo_stats` is empty, common for inference)
4. **`list_hlo_modules(run)` → `get_hlo_module_content(run, module)`** — inspect compiled HLO graph
5. **`get_hlo_neighborhood(run, instruction_name)`** — root-cause a specific slow op (BFS traversal)
6. **`get_memory_profile(run)`** — HBM usage, peak allocation, fragmentation
7. **`aggregate_xplane_events(run, plane_regex="/device:TPU:0")`** — timeline kernel statistics

---

## 3. The Roofline Model — Diagnosing Bottlenecks

Every optimization starts here. A kernel is either:

- **Compute-bound**: limited by FLOPs/second of the MXU. Adding bandwidth won't help.
- **Memory-bandwidth-bound**: limited by HBM bytes/second. Adding FLOPs won't help.

```
Arithmetic intensity = FLOPs ÷ bytes
```

If `intensity > (peak_FLOPs / peak_bw)`, the op is compute-bound. Otherwise it's memory-bound.

**Reading xprof's Roofline tool:** each dot is a kernel. Dots above the diagonal line are compute-bound; dots below it are memory-bandwidth-bound. Dots far below the line are severely under-utilizing HBM bandwidth (padding, small shapes, control overhead).

### The ICI roofline (multi-chip)

When tensors are sharded across chips, the crossover changes. For TP across `N` chips on TPU v5e ICI:

```
Compute-bound when: sharded_dim D > ~8755  (not the batch dim!)
```

This means a model sharded into 532-per-chip along a hidden dim of 8512÷16 is memory-bandwidth-limited at the ICI level, not the HBM level. Fix: reshard to 8 chips → 1064-per-chip → 62% FLOP utilization vs 46%.

---

## 4. Gotcha Catalogue

### 4.1 Dimension alignment — the #1 silent killer

**Problem:** Tensor shapes that aren't multiples of the MXU tile size (128/256) cause padding. The padded elements are computed but their results discarded — pure waste.

**Rules:**
- Batch size: multiple of 64 (8 per core) for minimum safety; multiple of 1024 for best efficiency
- Hidden/feature dimensions: multiple of 128; multiples of 256 preferred for v6e
- Sharded dims: check the per-chip size, not the global size

**Diagnosis:** Open Memory Viewer in xprof — shows padding percentage per tensor. Op Stats will show low FLOP utilization.

**Example:** Hidden dim 8512, sharded across 16 chips → 532 per chip. 532 is not a multiple of 128 or 256 → significant MXU waste. Reshard to 8 chips → 1064 per chip (close to 1024) → 16% throughput improvement.

### 4.2 Wrong dtype — bfloat16 as the first fix

**Problem:** Using fp32 weights for inference is the most common unforced performance error. The MXU natively accepts bf16; fp32 inputs require a cast on every matmul call.

**Fix:**
```python
# JAX
model = model.astype(jnp.bfloat16)

# PyTorch/XLA
model = model.to(torch.bfloat16)
```

**Expected gain:** 17% device-time improvement is typical (measured on production workloads). Quality impact in serving: negligible (weights are constant, no gradient accumulation noise).

**Diagnosis:** Open HLO Graph Viewer → inspect matmul weights → look for `f32` type. If present, it's casting to bf16 before compute.

### 4.3 Materialized broadcasts — a fusion failure

**Problem:** Some broadcast patterns cannot be fused with their consumers and land in HBM as full-size temporaries.

```python
# This broadcast CANNOT fuse with argmax → materialized in HBM:
tf.argmax(tf.add(vector, zero_matrix), axis=0)
```

**Diagnosis:** Look for large `bitcast`, `copy`, or `broadcast` ops at the top of the HLO Op Stats. Use `get_hlo_neighborhood` to see if the broadcast is the bottleneck.

**Fix:** Restructure the computation so the consumer can appear inside the same fusion as the broadcast.

### 4.4 Deferred execution surprises (PyTorch/XLA)

**Problem:** In torch_tpu's default `kDeferAndFuse` mode, ops accumulate until a "materialization trigger". `.item()`, `.cpu()`, `print(tensor)`, and conditional branches on tensor data all trigger synchronous device→host transfer and potentially a new compilation.

**Common anti-patterns:**
```python
# BAD: triggers device sync on every step
loss_val = loss.item()  # inside training loop
print(f"Loss: {loss}")   # same

# BAD: shape-dependent branch defeats fusion
if tensor.shape[0] == 1:  # fine (static shape)
    ...
if tensor.sum() > 0:      # BAD: evaluates tensor
    ...
```

**Debugging mode:** set `TPU_DEFER_NEVER=1` to get one-op-at-a-time dispatch, which makes errors point at the offending op instead of the materialization site.

### 4.5 Dynamic shapes break JIT caching

**Problem:** XLA compiles a separate executable per unique input shape. Variable-length batches, growing KV caches, or sequence-length-varying inputs each trigger a recompile.

**Fix — StaticCache for inference:**
```python
# Dynamic cache (re-traces every decode step):
model = AutoModelForCausalLM.from_pretrained(...)
# → use StaticCache:
from transformers import StaticCache
model._cache = StaticCache(config, max_batch_size=1, max_cache_len=2048)
```

**Measured impact (Llama-2-7B, 50 tokens, TPU v6e):**

| Setup | Wall time | Speedup |
|-------|-----------|---------|
| DynamicCache, eager | 130.9 s | 1.0× |
| StaticCache, eager | 88.4 s | 1.5× |
| StaticCache + `jax.jit` | **14.8 s** | **8.8×** |

**For inference serving:** pre-compile all shapes at startup. Every unique `(batch_size, seq_len)` combination needs its own compiled program — warm up all of them before accepting requests.

### 4.6 High rematerialization

**Problem:** Rematerialization (activation checkpointing) saves HBM by recomputing tensors during the backward pass. Too much remat wastes compute time.

**Diagnosis:** Check the "HLO Op Stats" xprof tab — look for "Time spent on rematerialization". Check Memory Profile — if HBM headroom is large, you're over-rematerializing.

**Fix — tune the remat policy:**
```python
# JAX: control which ops are kept vs recomputed
@functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
def layer_fn(x): ...

# MaxText: remat_policy config option
# PyTorch/XLA: torch.distributed.algorithms.checkpoint_wrapper
```

**Rule of thumb:** if memory headroom > 20%, try reducing remat. Example: changing policy in Gemini M SFT (VLP): 5891 ms → 5594 ms.

### 4.7 All-reduce in tensor parallelism

**Problem:** Tensor parallelism within a device slice is efficient (ICI bandwidth), but across slices (DCN) it degrades severely.

**Rule:** Keep TP ≤ 8 within an ICI island. Measured: ~43% degradation going TP=8 (intra-node) → TP=16 (inter-node).

**Diagnosis:** `aggregate_xplane_events(run, event_regex="all-reduce")` — check count, avg duration, and variance.

### 4.8 KV cache is memory-bandwidth-bound during decode

**Problem:** During autoregressive decoding, the GPU/TPU streams the entire KV cache through HBM every step. With a batch of 1 (or small batch), the token/byte ratio is extremely low — deeply memory-bandwidth-bound.

**Fix:**
- Increase decode batch size (amortizes KV read across more tokens)
- Use continuous batching / paged attention where supported
- Use splash/flash attention to keep attention computation in VMEM

**Reading the profile:** high `all-reduce` time + high `dynamic-slice`/`update-slice` time in xprof is characteristic of decode. The overview will show device duty cycle < 60%.

---

## 5. Training-Specific Optimizations

### Activation checkpointing (selective is best)

| Variant | Compute overhead | Memory saved |
|---------|-----------------|--------------|
| Full AC | +30–40% | ~100% of activations |
| Selective AC | **+~2.7%** | **~70% of activations** |
| None | 0% | 0% |

Selective AC is the default choice for production training: checkpoint only cheap-to-recompute ops (norms, elementwise) and keep expensive ones (attention matmul output).

### Scan layers (large models)

For models with N identical transformer layers, using `jax.lax.scan` (or `torchprime`'s `scan_layers`) reduces XLA compile time from O(N) to O(1):

```python
# torchprime: set model.scan_layers=True in config
# JAX/Flax: use nn.scan decorator
# torchax: use ScannedModule wrapper
```

**Critical gotcha:** scan complicates 2D sharding propagation in the backward pass. If you see OOM during backward with scan enabled, add explicit `shard_as` annotations to stabilize the sharding.

### Sharding mesh design checklist

- [ ] Place high-bandwidth collectives (FSDP AllGather/ReduceScatter) on ICI axes
- [ ] Place low-bandwidth collectives (DP gradient sync) on DCN axes
- [ ] Per-chip hidden dim is a multiple of 128 (256 for v6e)
- [ ] TP degree ≤ 8 within ICI island
- [ ] Pipeline parallelism only across pod boundaries (DCN)

---

## 6. Inference-Specific Optimizations

### Attention: always use flash/splash attention

Standard attention materializes an N×N attention matrix in HBM. For N=4096, that's 4096²×2 bytes = 32 MB per layer per request — pure memory-bandwidth waste.

| Implementation | Platform | How to enable |
|----------------|----------|---------------|
| Splash Attention | TPU (recommended) | `attention_kernel=splash_attention` in torchprime/torch_xla config |
| FlashAttention-2/3 | GPU (Triton) | `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2+) |
| tokamax `dot_product_attention` | GPU-first | via tokamax library |

### Quantization

Lower precision shifts the roofline toward compute-bound:

| Precision | Critical batch (v5e) | Notes |
|-----------|---------------------|-------|
| bf16 | B > 240 | Native TPU format |
| int8 weights, bf16 compute | B > ~120 | 2× memory bandwidth benefit |
| int8 weights + int8 compute | B > ~240 | FLOPs and bytes both halve |

**Recommended path:**
1. Start with bf16 (if not already)
2. Apply AQT int8 quantization to weight matrices
3. Use Profile-Guided Quantization (PGQ) to select per-layer config
4. Validate quality with held-out eval

```python
# AQT (JAX):
from aqt.jax.v2 import config as aqt_config
aqt_cfg = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=None)
```

### Batching for serving

Setting the max batch size too high increases latency; too low wastes throughput.

**Process:**
1. Measure device time at batch sizes: 1, 2, 4, 8, 16, 32, 64
2. Set `max_batch_size` where latency is 30–50% of P50 latency target
3. Set `batch_timeout_micros = (1 / throughput) × batch_size`
4. Warm up all shapes before accepting traffic (one compilation per shape)

**Gotcha:** latency scales linearly with batch size once you're compute-bound. Past that point, larger batches don't improve throughput per chip — they only increase latency.

---

## 7. XLA Compiler Flags

For profiling and debugging compilation:

```bash
# Dump HLO at every pass (needed for diff_hlo_stages):
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps \
                  --xla_dump_hlo_as_text \
                  --xla_dump_hlo_pass_re=.*"

# Then use xprof MCP tools:
# list_hlo_dump_modules("/tmp/hlo_dumps")
# get_hlo_dump("my_fn", "before_optimizations")
# diff_hlo_stages("my_fn", "before_optimizations", "after_optimizations")
# get_hlo_dump_neighborhood("fusion.3", "my_fn", "after_optimizations")
```

Useful XLA flags for performance:

| Flag | Effect |
|------|--------|
| `--xla_tpu_enable_async_collective_fusion=true` | Overlap all-reduce with compute |
| `--xla_enable_async_all_gather=true` | Pipelined weight all-gather (FSDP) |
| `--xla_tpu_megacore_fusion_allow_ags=true` | More aggressive fusion |
| `--xla_tpu_enable_latency_hiding_scheduler=true` | Better op scheduling |

### Memory flags (runtime)

```bash
# Pre-mapped DMA buffer (increase if seeing RESOURCE_EXHAUSTED on DMA):
export TPU_PREMAPPED_BUFFER_SIZE=8589934592  # 8 GB

# Disable tcmalloc for large embedding workloads (DLRM):
unset LD_PRELOAD
```

---

## 8. Decision Tree: What's Slow?

```
Start: get_overview(run)
│
├─ High device idle (> 30%)?
│   ├─ Host-device gap? → input pipeline bound
│   │   Fix: prefetch data, use grain/tf.data with parallel reads
│   └─ Collective waits? → all-reduce/all-gather bound
│       Fix: overlap comms with compute (async collectives),
│            reduce TP degree, switch DP to FSDP
│
├─ Device busy but slow ops?
│   └─ get_top_hlo_ops(run) → find hottest ops
│       │
│       ├─ Matmuls with low FLOP utilization (< 40%)?
│       │   → Dimension misalignment
│       │   Fix: check per-chip hidden dim is multiple of 128/256
│       │
│       ├─ Large bitcast/copy/broadcast ops?
│       │   → Layout mismatch or materialized broadcast
│       │   Fix: get_hlo_neighborhood → find the fusion boundary
│       │
│       ├─ High rematerialization time?
│       │   → Tune remat policy (less remat if HBM has headroom)
│       │
│       └─ dynamic-slice / update-slice dominant?
│           → KV cache decode (memory-bandwidth bound)
│           Fix: increase batch size, splash attention
│
└─ Training: OOM?
    ├─ Enable selective activation checkpointing (+2.7% compute, -70% activation memory)
    ├─ Use scan layers (reduce compile-time OOM for large models)
    └─ Enable FSDP to shard optimizer states + params
```

---

## 9. Quick Reference

### Dimension rules

| Dimension | Minimum | Preferred |
|-----------|---------|-----------|
| Batch per core | 8 | 128 |
| Total batch | multiple of 64 | multiple of 1024 |
| Hidden / feature dim (global) | multiple of 128 | multiple of 256 |
| Sharded dim (per chip) | multiple of 128 | multiple of 256 |

### Memory budget per parameter (training, bf16 mixed)

| Component | Bytes/param |
|-----------|-------------|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam m, v (fp32) | 4 + 4 |
| **Total** | **~16** |

### Common xprof MCP calls for an investigation

```
list_runs()
get_overview("2026_04_15_06_34_57")
get_top_hlo_ops("2026_04_15_06_34_57", limit=20)
get_op_profile("2026_04_15_06_34_57", top_n=10)
get_memory_profile("2026_04_15_06_34_57")
aggregate_xplane_events("2026_04_15_06_34_57",
    plane_regex="/device:TPU:0", top_n=20)
list_hlo_modules("2026_04_15_06_34_57")
get_hlo_neighborhood("2026_04_15_06_34_57", "fusion.42")
```

---

## 10. Sources

This guide synthesizes content from:

- [Cloud TPU Performance Guide](https://cloud.google.com/tpu/docs/performance-guide) (Google Cloud Docs, 2026)
- [How to Scale Your Model (2025)](https://jax-ml.github.io/scaling-book/) — the scaling book; Chapters 1 (roofline), 3 (sharding), 7 (inference)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (HuggingFace / nanotron) — selective AC numbers
