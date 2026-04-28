#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run openfhe_compute_distances under gem5 (SE mode).

This script targets the FHE centroid-distance executable invoked by
src/fhe/fhe_wrapper.py (OpenFHECoreSubprocessBackend).

Required environment variables:
  OPENFHE_CONTEXT_DIR     Path to OpenFHE context/key directory
  OPENFHE_CENTROIDS_FILE  Path to plaintext centroid rows file
  OPENFHE_ENCRYPTED_QUERY Path to encrypted_query.bin
  OPENFHE_ENCRYPTED_NORM  Path to encrypted_norm_squared.bin
  OPENFHE_OUTPUT_DIR      Output directory for encrypted distance bundles

gem5/OpenFHE configuration:
  GEM5_DIR                Path to gem5 repo (required if GEM5_BIN unset)
  GEM5_BIN                gem5 binary path (default: $GEM5_DIR/build/X86/gem5.opt)
  GEM5_SE_PY              gem5 SE config script (default: $GEM5_DIR/configs/deprecated/example/se.py if present)
  GEM5_HOST_LIB_DIR       Host lib dir for gem5 shared libs (auto-detected when possible)
  OPENFHE_BIN_DIR         Directory containing openfhe binaries (default: bin/openfhe)
  OPENFHE_COMPUTE_BIN     Path to openfhe_compute_distances binary
                          (default: $OPENFHE_BIN_DIR/openfhe_compute_distances)
  OPENFHE_BATCH_SIZE      Batch size (default: 64)
  OPENFHE_NUM_THREADS     Optional thread count (>0 adds --num-threads)
  GEM5_CPU_TYPE           CPU type (default: AtomicSimpleCPU)
  GEM5_MEM_SIZE           Memory size (default: 4GB)
  GEM5_ENABLE_CACHES      1 to pass --caches (default: 1)
  GEM5_ENABLE_L2          1 to pass --l2cache (default: 1)
  GEM5_OUTDIR             Output directory (default: <repo>/m5out/openfhe_distances)
  GEM5_MAXINSTS           Optional max guest instructions (adds --maxinsts)
  GEM5_TAKE_CHECKPOINTS   Optional checkpoint spec, e.g. 100000000
                          or <tick,period> passed to --take-checkpoints
  GEM5_CHECKPOINT_DIR     Optional checkpoint directory
  GEM5_CHECKPOINT_RESTORE Optional checkpoint index/inst to restore
  GEM5_RESTORE_WITH_CPU   Optional CPU type for restoring checkpoints
  GEM5_FAST_FORWARD       Optional instructions to fast-forward
  GEM5_EXTRA_ARGS         Extra gem5 args appended as-is

Example:
  GEM5_DIR=/path/to/gem5 \
  OPENFHE_BIN_DIR=/path/to/openfhe/bin \
  OPENFHE_CONTEXT_DIR=data/fhe_runtime/context \
  OPENFHE_CENTROIDS_FILE=data/fhe_runtime/query_x/distances/centroids.txt \
  OPENFHE_ENCRYPTED_QUERY=data/fhe_runtime/query_x/encrypted_query.bin \
  OPENFHE_ENCRYPTED_NORM=data/fhe_runtime/query_x/encrypted_norm_squared.bin \
  OPENFHE_OUTPUT_DIR=data/fhe_runtime/query_x/distances \
  scripts/run_gem5_openfhe.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GEM5_DIR="${GEM5_DIR:-}"
GEM5_BIN="${GEM5_BIN:-${GEM5_DIR:+$GEM5_DIR/build/X86/gem5.opt}}"
if [[ -z "${GEM5_SE_PY:-}" && -n "$GEM5_DIR" ]]; then
  if [[ -f "$GEM5_DIR/configs/deprecated/example/se.py" ]]; then
    GEM5_SE_PY="$GEM5_DIR/configs/deprecated/example/se.py"
  else
    GEM5_SE_PY="$GEM5_DIR/configs/example/se.py"
  fi
fi

if [[ -z "${GEM5_BIN:-}" || -z "${GEM5_SE_PY:-}" ]]; then
  echo "Error: set GEM5_DIR or both GEM5_BIN and GEM5_SE_PY." >&2
  exit 1
fi
if [[ ! -x "$GEM5_BIN" ]]; then
  echo "Error: gem5 binary is not executable: $GEM5_BIN" >&2
  exit 1
fi
if [[ ! -f "$GEM5_SE_PY" ]]; then
  echo "Error: gem5 config script not found: $GEM5_SE_PY" >&2
  exit 1
fi

GEM5_HOST_LIB_DIR="${GEM5_HOST_LIB_DIR:-}"
if [[ -z "$GEM5_HOST_LIB_DIR" ]]; then
  GEM5_REQUIRED_PYTHON_LIB="$(ldd "$GEM5_BIN" 2>/dev/null | awk '/libpython.*not found/ {print $1; exit}')"
  for candidate in \
    "${CONDA_PREFIX:-}/lib" \
    "$HOME/miniconda3/envs/gem5env/lib" \
    "$HOME/anaconda3/envs/gem5env/lib"; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -n "$GEM5_REQUIRED_PYTHON_LIB" && -f "$candidate/$GEM5_REQUIRED_PYTHON_LIB" ]]; then
      GEM5_HOST_LIB_DIR="$candidate"
      break
    fi
    if [[ -z "$GEM5_REQUIRED_PYTHON_LIB" ]] && compgen -G "$candidate/libpython*.so*" >/dev/null; then
      GEM5_HOST_LIB_DIR="$candidate"
      break
    fi
  done
fi
if [[ -n "$GEM5_HOST_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$GEM5_HOST_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OPENFHE_BIN_DIR="${OPENFHE_BIN_DIR:-bin/openfhe}"
OPENFHE_COMPUTE_BIN="${OPENFHE_COMPUTE_BIN:-$OPENFHE_BIN_DIR/openfhe_compute_distances}"
OPENFHE_BATCH_SIZE="${OPENFHE_BATCH_SIZE:-64}"
OPENFHE_NUM_THREADS="${OPENFHE_NUM_THREADS:-0}"

OPENFHE_CONTEXT_DIR="${OPENFHE_CONTEXT_DIR:-}"
OPENFHE_CENTROIDS_FILE="${OPENFHE_CENTROIDS_FILE:-}"
OPENFHE_ENCRYPTED_QUERY="${OPENFHE_ENCRYPTED_QUERY:-}"
OPENFHE_ENCRYPTED_NORM="${OPENFHE_ENCRYPTED_NORM:-}"
OPENFHE_OUTPUT_DIR="${OPENFHE_OUTPUT_DIR:-}"

if [[ -z "$OPENFHE_CONTEXT_DIR" || -z "$OPENFHE_CENTROIDS_FILE" || -z "$OPENFHE_ENCRYPTED_QUERY" || -z "$OPENFHE_ENCRYPTED_NORM" || -z "$OPENFHE_OUTPUT_DIR" ]]; then
  echo "Error: OPENFHE_CONTEXT_DIR, OPENFHE_CENTROIDS_FILE, OPENFHE_ENCRYPTED_QUERY, OPENFHE_ENCRYPTED_NORM, and OPENFHE_OUTPUT_DIR are required." >&2
  exit 1
fi

if [[ ! -x "$OPENFHE_COMPUTE_BIN" ]]; then
  echo "Error: OpenFHE compute binary is not executable: $OPENFHE_COMPUTE_BIN" >&2
  exit 1
fi
if [[ ! -d "$OPENFHE_CONTEXT_DIR" ]]; then
  echo "Error: context dir not found: $OPENFHE_CONTEXT_DIR" >&2
  exit 1
fi
if [[ ! -f "$OPENFHE_CENTROIDS_FILE" ]]; then
  echo "Error: centroids file not found: $OPENFHE_CENTROIDS_FILE" >&2
  exit 1
fi
if [[ ! -f "$OPENFHE_ENCRYPTED_QUERY" ]]; then
  echo "Error: encrypted query file not found: $OPENFHE_ENCRYPTED_QUERY" >&2
  exit 1
fi
if [[ ! -f "$OPENFHE_ENCRYPTED_NORM" ]]; then
  echo "Error: encrypted norm file not found: $OPENFHE_ENCRYPTED_NORM" >&2
  exit 1
fi

mkdir -p "$OPENFHE_OUTPUT_DIR"

GEM5_CPU_TYPE="${GEM5_CPU_TYPE:-AtomicSimpleCPU}"
GEM5_MEM_SIZE="${GEM5_MEM_SIZE:-4GB}"
GEM5_ENABLE_CACHES="${GEM5_ENABLE_CACHES:-1}"
GEM5_ENABLE_L2="${GEM5_ENABLE_L2:-1}"
GEM5_OUTDIR="${GEM5_OUTDIR:-$REPO_ROOT/m5out/openfhe_distances}"
GEM5_MAXINSTS="${GEM5_MAXINSTS:-}"
GEM5_TAKE_CHECKPOINTS="${GEM5_TAKE_CHECKPOINTS:-}"
GEM5_CHECKPOINT_DIR="${GEM5_CHECKPOINT_DIR:-}"
GEM5_CHECKPOINT_RESTORE="${GEM5_CHECKPOINT_RESTORE:-}"
GEM5_RESTORE_WITH_CPU="${GEM5_RESTORE_WITH_CPU:-}"
GEM5_FAST_FORWARD="${GEM5_FAST_FORWARD:-}"
GEM5_EXTRA_ARGS="${GEM5_EXTRA_ARGS:-}"
mkdir -p "$GEM5_OUTDIR"

WORKLOAD_OPTIONS="--context-dir $OPENFHE_CONTEXT_DIR --centroids-file $OPENFHE_CENTROIDS_FILE --encrypted-query $OPENFHE_ENCRYPTED_QUERY --encrypted-norm $OPENFHE_ENCRYPTED_NORM --output-dir $OPENFHE_OUTPUT_DIR --batch-size $OPENFHE_BATCH_SIZE"
if [[ "$OPENFHE_NUM_THREADS" =~ ^[0-9]+$ ]] && [[ "$OPENFHE_NUM_THREADS" -gt 0 ]]; then
  WORKLOAD_OPTIONS="$WORKLOAD_OPTIONS --num-threads $OPENFHE_NUM_THREADS"
fi

CMD=(
  "$GEM5_BIN"
  "--outdir=$GEM5_OUTDIR"
  "$GEM5_SE_PY"
  "--cmd=$OPENFHE_COMPUTE_BIN"
  "--options=$WORKLOAD_OPTIONS"
  "--cpu-type=$GEM5_CPU_TYPE"
  "--mem-size=$GEM5_MEM_SIZE"
)

if [[ "$GEM5_ENABLE_CACHES" == "1" ]]; then
  CMD+=("--caches")
fi
if [[ "$GEM5_ENABLE_L2" == "1" ]]; then
  CMD+=("--l2cache")
fi
if [[ -n "$GEM5_MAXINSTS" ]]; then
  CMD+=("--maxinsts=$GEM5_MAXINSTS")
fi
if [[ -n "$GEM5_TAKE_CHECKPOINTS" ]]; then
  CMD+=("--take-checkpoints=$GEM5_TAKE_CHECKPOINTS")
fi
if [[ -n "$GEM5_CHECKPOINT_DIR" ]]; then
  CMD+=("--checkpoint-dir=$GEM5_CHECKPOINT_DIR")
fi
if [[ -n "$GEM5_CHECKPOINT_RESTORE" ]]; then
  CMD+=("--checkpoint-restore=$GEM5_CHECKPOINT_RESTORE")
fi
if [[ -n "$GEM5_RESTORE_WITH_CPU" ]]; then
  CMD+=("--restore-with-cpu=$GEM5_RESTORE_WITH_CPU")
fi
if [[ -n "$GEM5_FAST_FORWARD" ]]; then
  CMD+=("--fast-forward=$GEM5_FAST_FORWARD")
fi
if [[ -n "$GEM5_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($GEM5_EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "gem5 run complete."
echo "Stats: $GEM5_OUTDIR/stats.txt"
