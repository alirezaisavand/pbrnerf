#!/usr/bin/env bash
set -uo pipefail

# ---------- Config you can tweak ----------
# Default scenes if none are passed on the command line:
DEFAULT_SCENES=(bear bunny cat crown dragon helmet)

# You can override these via env when calling:
#   TAGS=exp1 CONS_LOSS=0.1 SPEC_LOSS=0.2 ./run_many_glossyreal.sh bear bunny
TAGS="${TAGS:-debug}"
CONS_LOSS="${CONS_LOSS:-0.0}"
SPEC_LOSS="${SPEC_LOSS:-0.0}"

# Paths & names
DATA_ROOT="${DATA_ROOT:-/workspace/datasets/GlossyReal}"
CONFIG_PATH="${CONFIG_PATH:-configs/config_GlossySynthetic_pbrnerf.json}"
OUT_DIR="${OUT_DIR:-outputs}"
LOG_DIR="${LOG_DIR:-logs_glossyreal}"

# Prefix for --name (use GlossyReal by default; set NAME_PREFIX to keep your old GlossySynthetic prefix if you want)
NAME_PREFIX="${NAME_PREFIX:-pbrnerf_GlossyReal}"

# Where the 'code' folder is (robust to where you launch from)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"
mkdir -p "$LOG_DIR"

# ---------- Scenes to run ----------
if (( $# > 0 )); then
  SCENES=("$@")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi

echo "Scenes to run: ${SCENES[*]}"
echo "Tags=$TAGS  CONS_LOSS=$CONS_LOSS  SPEC_LOSS=$SPEC_LOSS"
echo "Data root: $DATA_ROOT"
echo "Config: $CONFIG_PATH"
echo "Logs: $LOG_DIR"
echo "Name prefix: $NAME_PREFIX"
echo

cd "$CODE_DIR" || { echo "Cannot cd to $CODE_DIR"; exit 1; }

FAILED=()

for SCENE in "${SCENES[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="../$LOG_DIR/${SCENE}_${ts}.log"
  run_name="${NAME_PREFIX}_${SCENE}"

  echo "================================================================"
  echo "[$(date)] Starting scene: $SCENE"
  echo "Log: $log_file"
  echo "================================================================"

  set +e  # allow failure but continue
  OPENCV_IO_ENABLE_OPENEXR=1 python3 training/train.py \
    "${DATA_ROOT}/${SCENE}" \
    "${OUT_DIR}" \
    --name "${run_name}" \
    --tags "${TAGS}" \
    --override_cons_weighting "${CONS_LOSS}" \
    --override_spec_weighting "${SPEC_LOSS}" \
    --config_path "${CONFIG_PATH}" \
    2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ $exit_code -ne 0 ]]; then
    echo "[$(date)] ❌ Scene '${SCENE}' failed with exit code ${exit_code}."
    FAILED+=("$SCENE")
  else
    echo "[$(date)] ✅ Scene '${SCENE}' finished successfully."
  fi
  echo
done

echo "====================== SUMMARY ======================"
if (( ${#FAILED[@]} )); then
  echo "Failed scenes: ${FAILED[*]}"
  exit 1
else
  echo "All scenes completed successfully."
fi
