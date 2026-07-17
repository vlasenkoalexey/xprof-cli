"""Regenerate the committed test fixtures on a TPU VM.

NOT run in CI — fixtures are committed artifacts; this script documents
exactly how they were produced so they can be regenerated when the
capture format changes. Run on a v6e (or later) single-chip TPU VM with
jax[tpu] installed.

The trace fixture is a Pallas emit_pipeline matmul captured WITH the two
kernel-profiling LIBTPU flags, so the LLO trace-side tools light up:

    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true \
                      --xla_xprof_register_llo_debug_info=true \
                      --xla_jf_dump_to=<fixtures>/jf_dump \
                      --xla_jf_dump_llo_text=true \
                      --xla_mosaic_dump_to=<fixtures>/mosaic_dump" \
    python tests/fixtures/capture_fixtures.py <fixtures-dir>

Note: the LLO dumper uses --xla_jf_dump_to; XLA's --xla_dump_to does NOT
receive LLO dumps. After capture, keep artifacts small (<5 MB each):
one-host xplane.pb, a handful of jf_dump checkpoint files, one mosaic
pass file, and the after_optimizations HLO text.
"""

import os
import sys


def capture(fixtures_dir: str) -> None:
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    logdir = os.path.join(fixtures_dir, "logdir")

    # Simple blocked matmul via pallas_call — enough to produce a Mosaic
    # custom call with pipeline stages. m,k,n = 2048,1024,2048;
    # bm,bk,bn = 512,512,1024 (matches the committed 2026-07-09 fixture).
    def kernel(x_ref, y_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] += jnp.dot(
            x_ref[...], y_ref[...], preferred_element_type=jnp.float32
        )

        @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
        def _():
            o_ref[...] = acc_ref[...].astype(o_ref.dtype)

    m, k, n = 2048, 1024, 2048
    bm, bk, bn = 512, 512, 1024

    from jax.experimental.pallas import tpu as pltpu

    matmul = pl.pallas_call(
        kernel,
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, h: (i, h)),
            pl.BlockSpec((bk, bn), lambda i, j, h: (h, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, h: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    )

    x = jnp.ones((m, k), jnp.bfloat16)
    y = jnp.ones((k, n), jnp.bfloat16)

    fn = jax.jit(functools.partial(matmul))
    fn(x, y).block_until_ready()  # compile outside the trace

    with jax.profiler.trace(logdir):
        for _ in range(3):
            fn(x, y).block_until_ready()

    print(f"Captured under {logdir}/plugins/profile/<session>/")
    print("Rename the session dir to 'testrun' and keep one host's"
          " .xplane.pb; prune jf_dump/mosaic_dump to a few files.")
    print("jf_dump checkpoints the tests need: final_bundles,"
          " *per-bundle-utilization, schedule-analysis_final_bundles, and"
          " one allocation-bearing IR pass (e.g. `original` or `post-ra`)"
          " for get_llo_fit_summary's VMEM section.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    capture(sys.argv[1])
