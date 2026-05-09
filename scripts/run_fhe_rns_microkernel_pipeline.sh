#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Export real OpenFHE distance operands natively, then run a standalone real-sized
RNS distance microkernel under gem5.

This avoids OpenFHE context/key/ciphertext deserialization inside gem5. gem5
starts at a small binary that reads flat arrays and immediately enters the ROI.

Required/important environment variables:
  OPENFHE_BIN_DIR          Directory containing openfhe_export_distance_operands
                           and fhe_rns_distance_kernel
                           (default: /home/jzhao7/RAGPIR/openfhe_core/build/bin)
  OPENFHE_CENTROIDS_FILE   Centroid rows file
  OPENFHE_ENCRYPTED_QUERY  encrypted_query.bin
  OPENFHE_ENCRYPTED_NORM   encrypted_norm_squared.bin
  WORK_DIR                 Export/output dir (default: data/fhe_rns_microkernel)
  GEM5_DIR                 Path to gem5 repo (default: tools/gem5)
  GEM5_OUTDIR              gem5 output dir (default: m5out/fhe_rns_microkernel)
  GEM5_CPU_TYPE            CPU type (default: AtomicSimpleCPU)
  GEM5_ENABLE_CACHES       1 to enable L1 caches (default: 1)
  GEM5_ENABLE_L2           1 to enable L2 cache (default: 1)
  GEM5_EXTRA_ARGS          Extra gem5 args
  KERNEL_REPEATS           Repeat kernel inside ROI (default: 1)

Example:
  source data/fhe_runtime_finish_one_distance/gem5_profile.env
  scripts/run_fhe_rns_microkernel_pipeline.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OPENFHE_BIN_DIR="${OPENFHE_BIN_DIR:-/home/jzhao7/RAGPIR/openfhe_core/build/bin}"
OPENFHE_CENTROIDS_FILE="${OPENFHE_CENTROIDS_FILE:-}"
OPENFHE_ENCRYPTED_QUERY="${OPENFHE_ENCRYPTED_QUERY:-}"
OPENFHE_ENCRYPTED_NORM="${OPENFHE_ENCRYPTED_NORM:-}"
WORK_DIR="${WORK_DIR:-data/fhe_rns_microkernel}"
GEM5_DIR="${GEM5_DIR:-$REPO_ROOT/tools/gem5}"
GEM5_BIN="${GEM5_BIN:-$GEM5_DIR/build/X86/gem5.opt}"
GEM5_SE_PY="${GEM5_SE_PY:-$GEM5_DIR/configs/deprecated/example/se.py}"
GEM5_OUTDIR="${GEM5_OUTDIR:-m5out/fhe_rns_microkernel}"
GEM5_CPU_TYPE="${GEM5_CPU_TYPE:-AtomicSimpleCPU}"
GEM5_MEM_SIZE="${GEM5_MEM_SIZE:-4GB}"
GEM5_ENABLE_CACHES="${GEM5_ENABLE_CACHES:-1}"
GEM5_ENABLE_L2="${GEM5_ENABLE_L2:-1}"
GEM5_EXTRA_ARGS="${GEM5_EXTRA_ARGS:-}"
KERNEL_REPEATS="${KERNEL_REPEATS:-1}"

EXPORT_BIN="$OPENFHE_BIN_DIR/openfhe_export_distance_operands"
KERNEL_BIN="$OPENFHE_BIN_DIR/fhe_rns_distance_kernel"

if [[ -z "$OPENFHE_CENTROIDS_FILE" || -z "$OPENFHE_ENCRYPTED_QUERY" || -z "$OPENFHE_ENCRYPTED_NORM" ]]; then
  echo "Error: OPENFHE_CENTROIDS_FILE, OPENFHE_ENCRYPTED_QUERY, and OPENFHE_ENCRYPTED_NORM are required." >&2
  exit 1
fi
for required in "$EXPORT_BIN" "$KERNEL_BIN" "$GEM5_BIN" "$GEM5_SE_PY" "$OPENFHE_CENTROIDS_FILE" "$OPENFHE_ENCRYPTED_QUERY" "$OPENFHE_ENCRYPTED_NORM"; do
  if [[ ! -e "$required" ]]; then
    echo "Error: required path missing: $required" >&2
    exit 1
  fi
done

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

mkdir -p "$WORK_DIR" "$GEM5_OUTDIR"

echo "== Export real FHE-sized RNS operands =="
"$EXPORT_BIN" \
  --centroids-file "$OPENFHE_CENTROIDS_FILE" \
  --encrypted-query "$OPENFHE_ENCRYPTED_QUERY" \
  --encrypted-norm "$OPENFHE_ENCRYPTED_NORM" \
  --output-dir "$WORK_DIR"

OPERANDS="$WORK_DIR/distance_operands.bin"
KERNEL_OUTPUT="$WORK_DIR/kernel_output.json"

WORKLOAD_OPTIONS="--operands $OPERANDS --output $KERNEL_OUTPUT --repeats $KERNEL_REPEATS --gem5-roi 1"

CMD=(
  "$GEM5_BIN"
  "--outdir=$GEM5_OUTDIR"
  "$GEM5_SE_PY"
  "--cmd=$KERNEL_BIN"
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
if [[ -n "$GEM5_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($GEM5_EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo
echo "== Run gem5 RNS distance microkernel =="
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "RNS microkernel pipeline complete."
echo "Operands: $OPERANDS"
echo "Kernel output: $KERNEL_OUTPUT"
echo "gem5 stats: $GEM5_OUTDIR/stats.txt"
