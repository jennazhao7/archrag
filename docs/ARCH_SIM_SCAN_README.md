# Architecture Simulation Scan Plan

This document defines the architecture simulation work split and the minimum scan matrix for the final project.

The goal is to compare the architecture behavior of:

- **Plaintext non-clustered retrieval**: query vs many document vectors.
- **Plaintext clustered retrieval**: query vs centroid vectors.
- **FHE clustered retrieval**: encrypted query vs plaintext centroid vectors.

We are not trying to run the full RAG pipeline inside gem5. We isolate the distance/similarity kernels and collect architecture statistics.

## Work Split

### Plaintext Person

Responsible for:

- `arch/build/pt_nonclustered`
  - Simulates plaintext query vs document matrix.
  - Work shape: `N x d`.
- `arch/build/pt_clustered`
  - Simulates plaintext query vs centroid matrix.
  - Work shape: `K x d`.
- Runs the cache hierarchy scan for both kernels.
- Reports runtime, simulated instructions, CPI/IPC, cache misses, and memory references.

Recommended baseline sizes:

```text
non-clustered: N=4096, d=384, metric=l2, iters=3
clustered:     K=64,   d=384, metric=l2, iters=20
```

Here `N=4096` means **4096 document vectors** for the non-clustered full scan. It is not a cluster count. Clustered plaintext should use `K=64` centroids.

Use the same sizes for every architecture setting so the runs are comparable.

### FHE Person

Responsible for:

- `openfhe_compute_distances`
  - Simulates encrypted query vs plaintext centroid distance.
  - Work shape: `1 encrypted query x 1 centroid`.
- Runs bounded gem5 slices because full OpenFHE completion is too slow.
- Reports results as an architecture profile of the FHE hotspot, not a completed full pipeline run.

Recommended FHE setting:

```text
SAMPLE_CENTROIDS=1
GEM5_MAXINSTS=100000000
OPENFHE_BATCH_SIZE=1
OPENFHE_NUM_THREADS=1
GEM5_CPU_TYPE=AtomicSimpleCPU
```

## CPU Model

Use this for all required scans:

```text
AtomicSimpleCPU
```

Reason: it is the only practical CPU model for the FHE workload under the time constraint. It still gives useful instruction count, CPI/IPC, cache access, cache miss, and memory reference statistics.

Avoid using `TimingSimpleCPU` or O3 CPUs for the main matrix unless there is extra time.

## Required Scan Matrix

Run these three cache hierarchy settings for each workload.

| Name | Caches | L2 | Purpose |
| --- | --- | --- | --- |
| `nocache` | off | off | Lower-bound cache support |
| `l1` | on | off | L1-only locality |
| `l1_l2` | on | on | Baseline cache hierarchy |

This creates:

```text
plaintext non-clustered: 3 runs
plaintext clustered:     3 runs
FHE clustered:           3 runs
```

Total minimum: **9 gem5 runs**.

## Optional FHE Cache Line Scan

If time allows, run this for FHE only with L1+L2 enabled:

```text
cacheline_size=32
cacheline_size=64
cacheline_size=128
```

Purpose: FHE has huge ciphertext data movement. The miss rate may be low, but total memory traffic is large, so cache line size is a useful secondary knob.

## Plaintext Setup

From repo root:

```bash
make -C arch
```

This builds:

```text
arch/build/pt_nonclustered
arch/build/pt_clustered
arch/build/pt_centroids_file
```

Quick native checks:

```bash
# Non-clustered full scan: 4096 document vectors, not 4096 clusters.
./arch/build/pt_nonclustered 4096 384 1 3

# Clustered centroid scan: 64 centroids.
./arch/build/pt_clustered 64 384 1 20
```

Both should print `done`, the problem size, elapsed time, and a checksum.

## Plaintext gem5 Commands

Set these once:

```bash
export GEM5_DIR=/home/jzhao7/archrag/tools/gem5
export GEM5_BIN=$GEM5_DIR/build/X86/gem5.opt
export GEM5_SE=$GEM5_DIR/configs/deprecated/example/se.py
export LD_LIBRARY_PATH=/home/jzhao7/miniconda3/envs/gem5env/lib:${LD_LIBRARY_PATH:-}
```

### Plaintext Non-Clustered

These runs use `4096 384 1 3`, meaning:

```text
N=4096 document vectors
d=384 embedding dimensions
metric=1 negative squared L2
iters=3 repeated kernel iterations
```

No cache:

```bash
$GEM5_BIN --outdir=m5out/pt_nonclustered_nocache \
  $GEM5_SE \
  --cmd=/home/jzhao7/archrag/arch/build/pt_nonclustered \
  --options="4096 384 1 3" \
  --cpu-type=AtomicSimpleCPU \
  --mem-size=4GB
```

L1 only:

```bash
$GEM5_BIN --outdir=m5out/pt_nonclustered_l1 \
  $GEM5_SE \
  --cmd=/home/jzhao7/archrag/arch/build/pt_nonclustered \
  --options="4096 384 1 3" \
  --cpu-type=AtomicSimpleCPU \
  --mem-size=4GB \
  --caches
```

L1 + L2:

```bash
$GEM5_BIN --outdir=m5out/pt_nonclustered_l1_l2 \
  $GEM5_SE \
  --cmd=/home/jzhao7/archrag/arch/build/pt_nonclustered \
  --options="4096 384 1 3" \
  --cpu-type=AtomicSimpleCPU \
  --mem-size=4GB \
  --caches \
  --l2cache
```

### Plaintext Clustered

You can use the wrapper:

These runs use `K=64`, meaning **64 centroids**, not 4096 clusters.

No cache:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
GEM5_OUTDIR=m5out/pt_clustered_nocache \
GEM5_ENABLE_CACHES=0 \
GEM5_ENABLE_L2=0 \
K=64 D=384 METRIC=1 ITERS=20 \
scripts/run_gem5_clustered.sh
```

L1 only:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
GEM5_OUTDIR=m5out/pt_clustered_l1 \
GEM5_ENABLE_CACHES=1 \
GEM5_ENABLE_L2=0 \
K=64 D=384 METRIC=1 ITERS=20 \
scripts/run_gem5_clustered.sh
```

L1 + L2:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
GEM5_OUTDIR=m5out/pt_clustered_l1_l2 \
GEM5_ENABLE_CACHES=1 \
GEM5_ENABLE_L2=1 \
K=64 D=384 METRIC=1 ITERS=20 \
scripts/run_gem5_clustered.sh
```

## FHE gem5 Commands

Use the wrapper. It performs native setup/encryption, optionally times the same OpenFHE distance kernel natively, and then runs the gem5 profile slice.

No cache:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
OPENFHE_BIN_DIR=/home/jzhao7/RAGPIR/openfhe_core/build/bin \
WORK_DIR=data/fhe_runtime_one_distance_nocache \
GEM5_OUTDIR=m5out/fhe_one_distance_nocache_100m \
GEM5_MAXINSTS=100000000 \
GEM5_ENABLE_CACHES=0 \
GEM5_ENABLE_L2=0 \
SAMPLE_CENTROIDS=1 \
QUERY_INDEX=0 \
OPENFHE_BATCH_SIZE=1 \
OPENFHE_NUM_THREADS=1 \
RUN_NATIVE_VERIFY=0 \
RUN_NATIVE_FHE_PROFILE=1 \
scripts/run_fhe_gem5_pipeline.sh
```

L1 only:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
OPENFHE_BIN_DIR=/home/jzhao7/RAGPIR/openfhe_core/build/bin \
WORK_DIR=data/fhe_runtime_one_distance_l1 \
GEM5_OUTDIR=m5out/fhe_one_distance_l1_100m \
GEM5_MAXINSTS=100000000 \
GEM5_ENABLE_CACHES=1 \
GEM5_ENABLE_L2=0 \
SAMPLE_CENTROIDS=1 \
QUERY_INDEX=0 \
OPENFHE_BATCH_SIZE=1 \
OPENFHE_NUM_THREADS=1 \
RUN_NATIVE_VERIFY=0 \
RUN_NATIVE_FHE_PROFILE=1 \
scripts/run_fhe_gem5_pipeline.sh
```

L1 + L2:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
OPENFHE_BIN_DIR=/home/jzhao7/RAGPIR/openfhe_core/build/bin \
WORK_DIR=data/fhe_runtime_one_distance_l1_l2 \
GEM5_OUTDIR=m5out/fhe_one_distance_l1_l2_100m \
GEM5_MAXINSTS=100000000 \
GEM5_ENABLE_CACHES=1 \
GEM5_ENABLE_L2=1 \
SAMPLE_CENTROIDS=1 \
QUERY_INDEX=0 \
OPENFHE_BATCH_SIZE=1 \
OPENFHE_NUM_THREADS=1 \
RUN_NATIVE_VERIFY=0 \
RUN_NATIVE_FHE_PROFILE=1 \
scripts/run_fhe_gem5_pipeline.sh
```

## Optional FHE Cache Line Commands

Run only with L1+L2 enabled:

```bash
GEM5_EXTRA_ARGS="--cacheline_size=32"
GEM5_EXTRA_ARGS="--cacheline_size=64"
GEM5_EXTRA_ARGS="--cacheline_size=128"
```

Example:

```bash
GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
OPENFHE_BIN_DIR=/home/jzhao7/RAGPIR/openfhe_core/build/bin \
WORK_DIR=data/fhe_runtime_one_distance_line32 \
GEM5_OUTDIR=m5out/fhe_one_distance_l1_l2_line32_100m \
GEM5_MAXINSTS=100000000 \
GEM5_ENABLE_CACHES=1 \
GEM5_ENABLE_L2=1 \
GEM5_EXTRA_ARGS="--cacheline_size=32" \
SAMPLE_CENTROIDS=1 \
QUERY_INDEX=0 \
OPENFHE_BATCH_SIZE=1 \
OPENFHE_NUM_THREADS=1 \
RUN_NATIVE_VERIFY=0 \
RUN_NATIVE_FHE_PROFILE=1 \
scripts/run_fhe_gem5_pipeline.sh
```

## Metrics To Record

From each `stats.txt`, record:

```text
simInsts
system.cpu.numCycles
system.cpu.cpi
system.cpu.ipc
system.cpu.commitStats0.numMemRefs
system.cpu.commitStats0.numLoadInsts
system.cpu.commitStats0.numStoreInsts
system.cpu.dcache.demandAccesses::total
system.cpu.dcache.demandMisses::total
system.cpu.dcache.demandMissRate::total
system.cpu.icache.demandMissRate::total
system.l2.overallMissRate::total     # only for L2 runs
```

Also record whether the workload completed:

- Plaintext should complete.
- FHE may stop at the instruction cap. That is expected and should be reported as a bounded profile slice.

## Expected Interpretation

Plaintext kernels are dense vector loops. They should finish under gem5 and show how non-clustered full scan differs from clustered centroid scan.

FHE clustered is the same logical retrieval stage, but with encrypted query arithmetic. It has much higher instruction count and memory traffic because OpenFHE CKKS uses large polynomial/RNS ciphertext structures. For FHE, the key result is not whether the full distance completes under gem5, but how the architecture statistics compare for the same instruction budget across cache settings.
