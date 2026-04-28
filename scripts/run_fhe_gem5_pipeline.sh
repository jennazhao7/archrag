#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the practical FHE + gem5 profiling pipeline.

This pipeline uses native execution for correctness/setup and gem5 for a bounded
profile of the real OpenFHE encrypted distance kernel:

  1) optionally verify full native clustered FHE correctness
  2) prepare OpenFHE context, encrypted query, and sampled centroid rows
  3) optionally time the exact one-distance OpenFHE kernel natively
  4) run openfhe_compute_distances under gem5 with an instruction cap

Important environment variables:
  GEM5_DIR                 Path to gem5 repo (default: tools/gem5)
  OPENFHE_BIN_DIR          Directory containing OpenFHE binaries
                           (default: /home/jzhao7/RAGPIR/openfhe_core/build/bin)
  QUERY                    Query text
  WORK_DIR                 Native OpenFHE artifact dir (default: data/fhe_runtime_gem5_profile)
  GEM5_OUTDIR              gem5 output dir (default: m5out/openfhe_profile_10m)
  GEM5_MAXINSTS            Max guest instructions (default: 10000000)
  GEM5_CPU_TYPE            gem5 CPU type (default: AtomicSimpleCPU)
  GEM5_TAKE_CHECKPOINTS    Optional gem5 --take-checkpoints value
  GEM5_CHECKPOINT_DIR      Optional gem5 checkpoint directory
  GEM5_CHECKPOINT_RESTORE  Optional gem5 checkpoint restore index
  GEM5_RESTORE_WITH_CPU    Optional gem5 restore CPU type
  GEM5_FAST_FORWARD        Optional gem5 fast-forward instruction count
  GEM5_ENABLE_CACHES       1 to enable L1 caches (default: 1)
  GEM5_ENABLE_L2           1 to enable L2 cache (default: 1)
  GEM5_EXTRA_ARGS          Extra gem5 args, e.g. --cacheline_size=32
  SAMPLE_CENTROIDS         Number of centroid rows to simulate (default: 1)
  CENTROID_START           First centroid row for contiguous sampling (default: 0)
  CENTROID_INDICES         Optional comma-separated centroid rows; overrides SAMPLE_CENTROIDS
  QUERY_EMBEDDINGS         Precomputed query embedding matrix (default: data/processed/query_embeddings.npy)
  QUERY_INDEX              Query embedding row to use for gem5 prep (default: 0)
  QUERY_ID                 Optional query id; overrides QUERY_INDEX when set
  USE_PRECOMPUTED_QUERY_EMBEDDING
                           1 uses QUERY_EMBEDDINGS instead of loading HF model (default: 1)
  OPENFHE_BATCH_SIZE       OpenFHE distance batch size (default: 1)
  OPENFHE_NUM_THREADS      OpenFHE distance threads (default: 1)
  OPENFHE_PARAM_PRESET     standard or toy (default: standard)
                           toy tries smaller params; may be rejected by OpenFHE
  RUN_NATIVE_VERIFY        1 to run full native FHE/plaintext verification first (default: 0)
  RUN_NATIVE_FHE_PROFILE   1 to time the same OpenFHE distance kernel natively (default: 1)

Example:
  GEM5_DIR=/home/jzhao7/archrag/tools/gem5 \
  OPENFHE_BIN_DIR=/home/jzhao7/RAGPIR/openfhe_core/build/bin \
  RUN_NATIVE_VERIFY=1 \
  GEM5_MAXINSTS=10000000 \
  scripts/run_fhe_gem5_pipeline.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GEM5_DIR="${GEM5_DIR:-$REPO_ROOT/tools/gem5}"
OPENFHE_BIN_DIR="${OPENFHE_BIN_DIR:-/home/jzhao7/RAGPIR/openfhe_core/build/bin}"
QUERY="${QUERY:-aspirin reduces heart attack risk}"
MODEL_NAME="${MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${DEVICE:-}"
CLUSTER_CENTROIDS="${CLUSTER_CENTROIDS:-data/processed/cluster_centroids.npy}"
QUERY_EMBEDDINGS="${QUERY_EMBEDDINGS:-data/processed/query_embeddings.npy}"
QUERY_IDS="${QUERY_IDS:-data/processed/query_ids.json}"
QUERY_INDEX="${QUERY_INDEX:-0}"
QUERY_ID="${QUERY_ID:-}"
USE_PRECOMPUTED_QUERY_EMBEDDING="${USE_PRECOMPUTED_QUERY_EMBEDDING:-1}"
DOC_EMBEDDINGS="${DOC_EMBEDDINGS:-data/processed/doc_embeddings.npy}"
DOC_IDS="${DOC_IDS:-data/processed/doc_ids.json}"
CLUSTER_DOC_IDS="${CLUSTER_DOC_IDS:-data/processed/cluster_doc_ids.json}"
WORK_DIR="${WORK_DIR:-data/fhe_runtime_gem5_profile}"
GEM5_OUTDIR="${GEM5_OUTDIR:-m5out/openfhe_profile_10m}"
GEM5_MAXINSTS="${GEM5_MAXINSTS:-10000000}"
GEM5_CPU_TYPE="${GEM5_CPU_TYPE:-AtomicSimpleCPU}"
GEM5_TAKE_CHECKPOINTS="${GEM5_TAKE_CHECKPOINTS:-}"
GEM5_CHECKPOINT_DIR="${GEM5_CHECKPOINT_DIR:-}"
GEM5_CHECKPOINT_RESTORE="${GEM5_CHECKPOINT_RESTORE:-}"
GEM5_RESTORE_WITH_CPU="${GEM5_RESTORE_WITH_CPU:-}"
GEM5_FAST_FORWARD="${GEM5_FAST_FORWARD:-}"
GEM5_ENABLE_CACHES="${GEM5_ENABLE_CACHES:-1}"
GEM5_ENABLE_L2="${GEM5_ENABLE_L2:-1}"
GEM5_EXTRA_ARGS="${GEM5_EXTRA_ARGS:-}"
SAMPLE_CENTROIDS="${SAMPLE_CENTROIDS:-1}"
CENTROID_START="${CENTROID_START:-0}"
CENTROID_INDICES="${CENTROID_INDICES:-}"
OPENFHE_BATCH_SIZE="${OPENFHE_BATCH_SIZE:-1}"
OPENFHE_NUM_THREADS="${OPENFHE_NUM_THREADS:-1}"
OPENFHE_PARAM_PRESET="${OPENFHE_PARAM_PRESET:-standard}"
if [[ "$OPENFHE_PARAM_PRESET" == "toy" ]]; then
  OPENFHE_POLY_MODULUS_DEGREE="${OPENFHE_POLY_MODULUS_DEGREE:-8192}"
  OPENFHE_COEFF_MOD_BIT_SIZES="${OPENFHE_COEFF_MOD_BIT_SIZES:-50,30,30,50}"
else
  OPENFHE_POLY_MODULUS_DEGREE="${OPENFHE_POLY_MODULUS_DEGREE:-16384}"
  OPENFHE_COEFF_MOD_BIT_SIZES="${OPENFHE_COEFF_MOD_BIT_SIZES:-60,40,40,60}"
fi
RUN_NATIVE_VERIFY="${RUN_NATIVE_VERIFY:-0}"
RUN_NATIVE_FHE_PROFILE="${RUN_NATIVE_FHE_PROFILE:-1}"

if [[ ! -x "$OPENFHE_BIN_DIR/openfhe_compute_distances" ]]; then
  echo "Error: OpenFHE compute binary not executable: $OPENFHE_BIN_DIR/openfhe_compute_distances" >&2
  exit 1
fi
if [[ ! -x "$GEM5_DIR/build/X86/gem5.opt" ]]; then
  echo "Error: gem5 binary not executable: $GEM5_DIR/build/X86/gem5.opt" >&2
  exit 1
fi
for required in "$CLUSTER_CENTROIDS" "$DOC_EMBEDDINGS" "$DOC_IDS" "$CLUSTER_DOC_IDS"; do
  if [[ ! -e "$required" ]]; then
    echo "Error: required pipeline artifact is missing: $required" >&2
    exit 1
  fi
done
if [[ "$USE_PRECOMPUTED_QUERY_EMBEDDING" == "1" ]]; then
  for required in "$QUERY_EMBEDDINGS" "$QUERY_IDS"; do
    if [[ ! -e "$required" ]]; then
      echo "Error: required query artifact is missing: $required" >&2
      exit 1
    fi
  done
fi

echo "== FHE/gem5 pipeline configuration =="
echo "query: $QUERY"
echo "openfhe bin dir: $OPENFHE_BIN_DIR"
echo "gem5 dir: $GEM5_DIR"
echo "centroid sample: ${CENTROID_INDICES:-start=$CENTROID_START count=$SAMPLE_CENTROIDS}"
if [[ "$USE_PRECOMPUTED_QUERY_EMBEDDING" == "1" ]]; then
  echo "query embedding: ${QUERY_ID:-index=$QUERY_INDEX} (precomputed)"
else
  echo "query embedding: encoded from QUERY text"
fi
echo "gem5 max instructions: $GEM5_MAXINSTS"
echo "gem5 cpu type: $GEM5_CPU_TYPE"
echo "gem5 caches: L1=$GEM5_ENABLE_CACHES L2=$GEM5_ENABLE_L2 extra_args=${GEM5_EXTRA_ARGS:-none}"
echo "OpenFHE params: preset=$OPENFHE_PARAM_PRESET ring_dim=$OPENFHE_POLY_MODULUS_DEGREE coeff_bits=$OPENFHE_COEFF_MOD_BIT_SIZES"
if [[ "$OPENFHE_PARAM_PRESET" == "toy" ]]; then
  echo "note: toy params are for simulation experiments only and may be rejected by this OpenFHE build."
fi
echo

if [[ "$RUN_NATIVE_VERIFY" == "1" ]]; then
  echo "== Stage 1: native FHE correctness check =="
  NATIVE_VERIFY_CMD=(
    python scripts/verify_fhe_clustered_consistency.py
    --backend openfhe
    --query "$QUERY"
    --metric l2
    --doc-embeddings "$DOC_EMBEDDINGS"
    --doc-ids "$DOC_IDS"
    --cluster-centroids "$CLUSTER_CENTROIDS"
    --cluster-doc-ids "$CLUSTER_DOC_IDS"
    --model-name "$MODEL_NAME"
    --openfhe-bin-dir "$OPENFHE_BIN_DIR"
    --openfhe-work-dir "$WORK_DIR/native_verify"
    --openfhe-poly-modulus-degree "$OPENFHE_POLY_MODULUS_DEGREE"
    --openfhe-coeff-mod-bit-sizes "$OPENFHE_COEFF_MOD_BIT_SIZES"
    --openfhe-batch-size "$OPENFHE_BATCH_SIZE"
    --openfhe-num-threads "$OPENFHE_NUM_THREADS"
  )
  if [[ -n "$DEVICE" ]]; then
    NATIVE_VERIFY_CMD+=(--device "$DEVICE")
  fi
  "${NATIVE_VERIFY_CMD[@]}"
  echo
else
  echo "== Stage 1: native FHE correctness check skipped =="
  echo "Set RUN_NATIVE_VERIFY=1 to validate full native FHE vs plaintext scores."
  echo
fi

echo "== Stage 2: prepare real OpenFHE artifacts for gem5 =="
PREP_CMD=(
  python scripts/prepare_gem5_openfhe_profile.py
  --query "$QUERY"
  --query-embeddings "$QUERY_EMBEDDINGS"
  --query-ids "$QUERY_IDS"
  --query-index "$QUERY_INDEX"
  --cluster-centroids "$CLUSTER_CENTROIDS"
  --sample-centroids "$SAMPLE_CENTROIDS"
  --centroid-start "$CENTROID_START"
  --model-name "$MODEL_NAME"
  --openfhe-bin-dir "$OPENFHE_BIN_DIR"
  --work-dir "$WORK_DIR"
  --poly-modulus-degree "$OPENFHE_POLY_MODULUS_DEGREE"
  --coeff-mod-bit-sizes "$OPENFHE_COEFF_MOD_BIT_SIZES"
  --batch-size "$OPENFHE_BATCH_SIZE"
  --num-threads "$OPENFHE_NUM_THREADS"
)
if [[ "$USE_PRECOMPUTED_QUERY_EMBEDDING" == "1" ]]; then
  PREP_CMD+=(--use-precomputed-query-embedding)
else
  PREP_CMD+=(--no-use-precomputed-query-embedding)
fi
if [[ -n "$DEVICE" ]]; then
  PREP_CMD+=(--device "$DEVICE")
fi
if [[ -n "$CENTROID_INDICES" ]]; then
  PREP_CMD+=(--centroid-indices "$CENTROID_INDICES")
fi
if [[ -n "$QUERY_ID" ]]; then
  PREP_CMD+=(--query-id "$QUERY_ID")
fi
"${PREP_CMD[@]}"
echo

ENV_FILE="$WORK_DIR/gem5_profile.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: expected env file missing: $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ "$RUN_NATIVE_FHE_PROFILE" == "1" ]]; then
  echo "== Stage 3: native timing of the exact OpenFHE distance kernel =="
  NATIVE_PROFILE_OUTDIR="$WORK_DIR/native_distance_profile"
  NATIVE_PROFILE_LOG="$WORK_DIR/native_distance_profile.log"
  rm -rf "$NATIVE_PROFILE_OUTDIR"
  mkdir -p "$NATIVE_PROFILE_OUTDIR"
  NATIVE_PROFILE_CMD=(
    "$OPENFHE_BIN_DIR/openfhe_compute_distances"
    --context-dir "$OPENFHE_CONTEXT_DIR"
    --centroids-file "$OPENFHE_CENTROIDS_FILE"
    --encrypted-query "$OPENFHE_ENCRYPTED_QUERY"
    --encrypted-norm "$OPENFHE_ENCRYPTED_NORM"
    --output-dir "$NATIVE_PROFILE_OUTDIR"
    --batch-size "$OPENFHE_BATCH_SIZE"
  )
  if [[ "$OPENFHE_NUM_THREADS" =~ ^[0-9]+$ ]] && [[ "$OPENFHE_NUM_THREADS" -gt 0 ]]; then
    NATIVE_PROFILE_CMD+=(--num-threads "$OPENFHE_NUM_THREADS")
  fi
  /usr/bin/time -f "native_wall_seconds=%e\nnative_max_rss_kb=%M" \
    -o "$NATIVE_PROFILE_LOG" \
    "${NATIVE_PROFILE_CMD[@]}" | tee -a "$NATIVE_PROFILE_LOG"
  echo "Native profile log: $NATIVE_PROFILE_LOG"
  echo "Native profile outputs: $NATIVE_PROFILE_OUTDIR"
  echo
else
  echo "== Stage 3: native OpenFHE distance timing skipped =="
  echo "Set RUN_NATIVE_FHE_PROFILE=1 to time the same kernel before gem5."
  echo
fi

echo "== Stage 4: run bounded gem5 profile of OpenFHE distance kernel =="
GEM5_DIR="$GEM5_DIR" \
GEM5_OUTDIR="$GEM5_OUTDIR" \
GEM5_CPU_TYPE="$GEM5_CPU_TYPE" \
GEM5_MAXINSTS="$GEM5_MAXINSTS" \
GEM5_TAKE_CHECKPOINTS="$GEM5_TAKE_CHECKPOINTS" \
GEM5_CHECKPOINT_DIR="$GEM5_CHECKPOINT_DIR" \
GEM5_CHECKPOINT_RESTORE="$GEM5_CHECKPOINT_RESTORE" \
GEM5_RESTORE_WITH_CPU="$GEM5_RESTORE_WITH_CPU" \
GEM5_FAST_FORWARD="$GEM5_FAST_FORWARD" \
GEM5_ENABLE_CACHES="$GEM5_ENABLE_CACHES" \
GEM5_ENABLE_L2="$GEM5_ENABLE_L2" \
GEM5_EXTRA_ARGS="$GEM5_EXTRA_ARGS" \
OPENFHE_BIN_DIR="$OPENFHE_BIN_DIR" \
OPENFHE_CONTEXT_DIR="$OPENFHE_CONTEXT_DIR" \
OPENFHE_CENTROIDS_FILE="$OPENFHE_CENTROIDS_FILE" \
OPENFHE_ENCRYPTED_QUERY="$OPENFHE_ENCRYPTED_QUERY" \
OPENFHE_ENCRYPTED_NORM="$OPENFHE_ENCRYPTED_NORM" \
OPENFHE_OUTPUT_DIR="$OPENFHE_OUTPUT_DIR" \
OPENFHE_BATCH_SIZE="$OPENFHE_BATCH_SIZE" \
OPENFHE_NUM_THREADS="$OPENFHE_NUM_THREADS" \
scripts/run_gem5_openfhe.sh

echo
echo "Pipeline complete."
echo "Manifest: $WORK_DIR/gem5_profile_manifest.json"
echo "gem5 stats: $GEM5_OUTDIR/stats.txt"
