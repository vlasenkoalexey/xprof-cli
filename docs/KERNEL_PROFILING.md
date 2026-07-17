# Kernel Profiling / LLO Analysis Guide

How to profile *inside* a Pallas/Mosaic TPU kernel with the xprof-mcp LLO tool
suite: per-functional-unit utilization, Mosaic pipeline stage times, VLIW
bundle listings, and the lowered MLIR — the signals that turn "this kernel is
slow" into "the MXU is starved because vector loads dominate, in these
bundles, from this source line."

All mechanisms below were validated on TPU v6e (2026-07-09) with a Pallas
`emit_pipeline` matmul. The libtpu dump flags are **undocumented** — they may
drift across libtpu releases (see [Flag discovery](#flag-discovery)).

---

## 1. Capture

### Trace-side (cheap — enable for any kernel work)

```bash
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true \
                  --xla_xprof_register_llo_debug_info=true" \
python your_pallas_workload.py     # capture with jax.profiler.trace(logdir)
```

This adds three line families to each `/device:TPU:N` plane in the
`.xplane.pb`:

| Line | Contents | Consumed by |
|---|---|---|
| `_counters_` | `% util` per ~1µs window for **MXU, Scalar ALU, Vector ALU, Vector Load, Vector Store, XLU, Vector EUP**; counts for **Vector Fills / Vector Spills** | `get_llo_utilization` |
| `Tensor Core` | `bundle.<id>` markers; `long_name` carries the LLO **bundle address** (`0x1ee`) | join key into `get_llo_bundles` |
| `XLA TraceMe` | Mosaic pipeline stages: `ep_wait_in`, `ep_copy_in`, `ep_run_kernel`, your `jax.named_scope`s, `ep_copy_out`, `ep_wait_out`, `ep_initialize_*`, `ep_finalize` | `get_kernel_stage_breakdown` |

**Semantics caveat:** the `% util` values are LLO *static slot occupancy*
laid over measured time windows — static content, measured time alignment.
They are not raw runtime hardware counters. True runtime counter sampling
(`tpu_enable_periodic_counter_sampling`) requires **TPU v7 (Ironwood)+** and
is a *silent no-op* on earlier generations (run completes, no tracks, no
error).

### Dump-side (verbose — enable for drilldowns)

```bash
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/jf_dump \
                  --xla_jf_dump_llo_text=true \
                  --xla_jf_dump_llo_static_gaps=true \
                  --xla_mosaic_dump_to=/tmp/mosaic_dump" \
python your_pallas_workload.py
export XLA_JF_DUMP_DIR=/tmp/jf_dump
```

> **Trap:** the LLO dumper writes to its **own** `--xla_jf_dump_to`
> directory. XLA's `--xla_dump_to` does *not* receive LLO dumps — with only
> that flag, `--xla_jf_dump_llo_text=true` is accepted and silently writes
> nothing.

Output is one file per (LLO program × compiler pass) — ~126 passes for a
kernel program, thousands of files per run (filter with
`--xla_jf_dump_llo_pass_label_regex`). The checkpoints the tools parse:

| File | Contents |
|---|---|
| `*-final_bundles.txt` | the VLIW bundle listing: scalar/vector/DMA instructions per bundle, HLO attribution comments, loop-region markers |
| `*-hlo-static-per-bundle-utilization.txt` | per-bundle issue-slot matrix over `MXU, XLU, VALU, VPOP, EUP, VLOAD, VLOAD:FILL, VSTORE, VSTORE:SPILL, SALU` + a capacity row |
| `*-schedule-analysis_*.txt` | bundle counts attributed per HLO instruction and per opcode |
| `*-static_gap_analysis_*.txt` | static gap analysis |

`--xla_mosaic_dump_to` additionally dumps the **textual Mosaic MLIR** after
every Mosaic pass (`post-deserialization` … `post-finalize-llo`).

---

## 2. Analysis flow

```
check_kernel_profiling(run)                     were the flags on? (always first)
        │
list_kernel_invocations(run)                    which kernels, how long, spans
        │
get_llo_utilization(run, kernel=...)            per-unit % util → bottleneck verdict
get_kernel_stage_breakdown(run, kernel=...)     ep_* stage times → wait_ratio
        │  (drilldown, needs --xla_jf_dump_to)
list_llo_programs(dump_dir)                     programs × pass checkpoints
get_llo_schedule_analysis(dump_dir, program)    bundles per HLO op / opcode
get_llo_static_utilization(dump_dir, program)   slot occupancy + hot address ranges
get_llo_bundles(dump_dir, program,
                address_range="0x34c-0x4ce")    the actual VLIW bundles
        │  (structural audit)
get_custom_call_mlir(kernel=..., ...)           did it lower/tile as planned?
```

### Reading the results

- **`get_llo_utilization` verdict**: `dominant_unit` + `memory_bound_signal`
  (Vector Load util > MXU util) + `scalar_bound_signal` (Scalar ALU > MXU).
  Example failure signatures:
  - MXU high & steady → compute-bound, tune block shapes for more MACs/byte.
  - Vector Load ≫ MXU → HBM-bandwidth / pipelining problem.
  - Scalar ALU high with MXU idle → scalar-core serialization (address math,
    control flow) stalling the pipeline.
  - `Vector Spills` nonzero → register pressure; shrink live ranges/tiles.
- **`get_kernel_stage_breakdown` wait_ratio** =
  `(ep_wait_in + ep_wait_out) / ep_run_kernel`. High (>~0.3) means compute is
  starved waiting on DMAs — increase `pl.Buffered(buffer_count=N)` on the
  starving input, or enlarge blocks. (Validated: a single-buffered input
  showed `wait_ratio 3.98`; the same kernel triple-buffered dropped to well
  under 1 with a ~30% wall-clock gain.)
- **Trace ↔ dump join**: `Tensor Core` marker `long_name`s and
  `get_llo_static_utilization`'s `hot_ranges` both yield `0x...` bundle
  addresses that plug into `get_llo_bundles(address_range=...)` — from "the
  utilization dip at t=..." to the exact instructions.

---

## 3. Tool reference

Trace-side (need `tensorflow-cpu` + `XPROF_LOGDIR`):

| Tool | One-liner |
|---|---|
| `check_kernel_profiling(run)` | Flags-active audit; per-device line presence, counter units, stages |
| `list_kernel_invocations(run, kernel_regex?)` | Kernel executions + duration stats + spans (the `kernel` handles) |
| `get_llo_utilization(run, kernel?, start/end_time_ps?, timeline_buckets?)` | Per-unit `% util` stats + bottleneck verdict |
| `get_kernel_stage_breakdown(run, kernel?)` | `ep_*` + named_scope stage times, `wait_ratio` |

Dump-side (no server / no TF needed; `dump_dir` arg or `XLA_JF_DUMP_DIR`):

| Tool | One-liner |
|---|---|
| `list_llo_programs(dump_dir)` | Programs (kernel/fusion/infrastructure) × available checkpoints |
| `get_llo_schedule_analysis(dump_dir, program)` | Total/empty bundles; per-HLO and per-opcode attribution |
| `get_llo_static_utilization(dump_dir, program)` | Occupancy vs capacity per unit; saturated %; dominant-unit hot ranges |
| `get_llo_bundles(dump_dir, program, address_range?, grep?, limit?)` | Windowed VLIW listing (hard-capped; always narrow) |
| `get_llo_fit_summary(dump_dir, program?, top_stalls?, diff_dump_dir?)` | ~30-line composed digest: VMEM vs limit + headroom, MXU width (128/256 lanes), spill/fill rate, timeline classes + stall runs, ranked levers, machine-readable verdict class |
| `get_device_wall_report(run, kernel?, measure_json?, baseline_run?, floor_ms?)` | Device-busy vs wall p50 dual report; labeled `wall_ratio` (deployable) / `device_ratio` (device-framing); `floor_ms` stamps sub-physical claims PHYSICALLY_IMPOSSIBLE |
| `get_custom_call_mlir(kernel?, mosaic_dump_dir?, hlo_dump_dir?)` | Lowered Mosaic MLIR: full text from `--xla_mosaic_dump_to`, or a bytecode structural summary (op counts + named scopes) decoded from the HLO dump's `custom_call_config.body` |

---

## 4. Flag discovery

The `xla_jf_*` flags are not in any public documentation. Two discovery
vectors, useful when a libtpu upgrade changes behavior:

1. Any run with `XLA_FLAGS="--xla_dump_to=<dir>"` writes a per-module
   `*.flagfile` listing **every effective jellyfish/TPU flag with its
   value** — the ground truth for what your libtpu accepts.
2. `strings libtpu.so | grep -i llo` enumerates candidate flag names.

Known-good set (libtpu ~2026-07, v6e):
`--xla_jf_dump_to`, `--xla_jf_dump_llo_text`, `--xla_jf_dump_llo_html`,
`--xla_jf_dump_llo_proto`, `--xla_jf_dump_llo_static_gaps`,
`--xla_jf_dump_llo_pass_label_regex`, `--xla_jf_dump_llo_critical_paths_to`,
`--xla_jf_dump_static_llo_profile_to`, `--xla_mosaic_dump_to`,
`--xla_mosaic_enable_llo_source_annotations`,
`--xla_tpu_include_hlo_statistics_in_llo_dump`.
