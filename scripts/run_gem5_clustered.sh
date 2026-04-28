#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the plaintext clustered kernel under gem5 (SE mode).

Environment variables (optional unless noted):
  GEM5_DIR                Path to gem5 repo (required if GEM5_BIN unset)
  GEM5_BIN                gem5 binary path (default: $GEM5_DIR/build/X86/gem5.opt)
  GEM5_SE_PY              gem5 SE config script (default: $GEM5_DIR/configs/deprecated/example/se.py if present)
  GEM5_HOST_LIB_DIR       Host lib dir for gem5 shared libs (auto-detected when possible)
  GEM5_CPU_TYPE           CPU type (default: AtomicSimpleCPU)
  GEM5_MEM_SIZE           Memory size (default: 4GB)
  GEM5_ENABLE_CACHES      1 to pass --caches (default: 1)
  GEM5_ENABLE_L2          1 to pass --l2cache (default: 1)
  GEM5_OUTDIR             Output directory (default: <repo>/m5out/pt_clustered)
  GEM5_EXTRA_ARGS         Extra gem5 args appended as-is
  ARCH_BUILD              1 to auto-build arch kernel if missing (default: 1)
  K                       Number of centroids (default: 64)
  D                       Embedding dimension (default: 128)
  METRIC                  0=dot, 1=-L2^2 (default: 1)
  ITERS                   Iterations in workload (default: 20)

Examples:
  GEM5_DIR=/path/to/gem5 scripts/run_gem5_clustered.sh
  GEM5_DIR=/path/to/gem5 K=256 D=384 ITERS=50 scripts/run_gem5_clustered.sh
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

ARCH_BUILD="${ARCH_BUILD:-1}"
WORKLOAD="$REPO_ROOT/arch/build/pt_clustered"
if [[ ! -x "$WORKLOAD" ]]; then
  if [[ "$ARCH_BUILD" == "1" ]]; then
    make -C "$REPO_ROOT/arch"
  fi
fi
if [[ ! -x "$WORKLOAD" ]]; then
  echo "Error: workload binary not found/executable: $WORKLOAD" >&2
  exit 1
fi

K="${K:-64}"
D="${D:-128}"
METRIC="${METRIC:-1}"
ITERS="${ITERS:-20}"

GEM5_CPU_TYPE="${GEM5_CPU_TYPE:-AtomicSimpleCPU}"
GEM5_MEM_SIZE="${GEM5_MEM_SIZE:-4GB}"
GEM5_ENABLE_CACHES="${GEM5_ENABLE_CACHES:-1}"
GEM5_ENABLE_L2="${GEM5_ENABLE_L2:-1}"
GEM5_OUTDIR="${GEM5_OUTDIR:-$REPO_ROOT/m5out/pt_clustered}"
GEM5_EXTRA_ARGS="${GEM5_EXTRA_ARGS:-}"

mkdir -p "$GEM5_OUTDIR"

OPTIONS="$K $D $METRIC $ITERS"
CMD=(
  "$GEM5_BIN"
  "--outdir=$GEM5_OUTDIR"
  "$GEM5_SE_PY"
  "--cmd=$WORKLOAD"
  "--options=$OPTIONS"
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

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo
echo "gem5 run complete."
echo "Stats: $GEM5_OUTDIR/stats.txt"
