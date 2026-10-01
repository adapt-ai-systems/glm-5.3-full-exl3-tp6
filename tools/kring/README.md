# kring — the CUPTI library that made our TP6 prefill hang go away

`libkring.so` is a small CUPTI injection library we wrote for hang forensics. It records every kernel and graph
launch (name, stream, correlation id) into an mmap ring per CUDA process, so you can see which kernel a stuck GPU
is parked in without attaching a debugger.

Why it is here: with it loaded, the TP6 prefill hang described in RESULTS.md "Known problems" stopped reproducing
on our cluster (pbench hung 5 of 9 runs without it, 0 of 5 runs plus 40 soak pairs with it). We never root-caused
the hang, so treat this as a workaround for a timing race, not a fix.

## Build (on a Spark)

    tools/kring/build.sh            # CUDA_TARGET=... to point at another CUDA 13 sbsa tree

## Use with the TP6 launcher

Copy `libkring.so` to every node, then add to the container's run flags (all ranks):

    --volume /path/to/kring:/opt/kring:ro \
    -e CUDA_INJECTION64_PATH=/opt/kring/libkring.so -e KRING_ACTIVITY=0

`KRING_ACTIVITY=0` is the launch-ring-only mode we ran with: no measurable cost (2.10 vs 2.10 us per launch in a
200k tiny-kernel worst case). Full mode (completion activity too) adds ~1.9 us per launch and changes kernel timing;
use it only to diagnose a hang, never for benchmark numbers. Each CUDA process prints
`kring: tracing kernel launches into /tmp/kring/...` when it loads.

## Read a ring

    docker cp <container>:/tmp/kring /tmp/kr && tools/kring/spark-kring /tmp/kr

Per stream, the newest launch with a completion is where the stream got to; the next launch is what the GPU is
running or stuck in (needs full mode for completions).

Self-test: `test_spin.cu` runs a spin kernel with three kernels queued behind it.

Gotcha: `InitializeInjection` must return 1. On 0, libcuda unloads the library and CUPTI jumps into unmapped code.
