#!/usr/bin/env bash
set -uo pipefail

# ---------- Config you can tweak ----------
# Default list if you don't pass scenes on the command line:
DEFAULT_SCENES=(angel bell cat crown dragon bunny helmet teapot)

# Optional: set via env when calling, e.g. TAGS=exp1 CONS_LOSS=0.1 SPEC_LOSS=0.2 ./run_many.sh angel bell
TAGS="${TAGS:-debug}"
CONS_LOSS="${CONS_LOSS:-0.0}"
SPEC_LOSS="${SPEC_LOSS:-0.0}"
DATA_ROOT="${DATA_ROOT:-/workspace/datasets/GlossySynthetic}"
CONFIG_PATH="${CONFIG_PATH:-configs/config_GlossySynthetic_pbrnerf.json}"
OUT_DIR="${OUT_DIR:-outputs}"
LOG_DIR="${LOG_DIR:-logs}"

# Where your 'code' directory lives relative to this script.
# This makes the script robust no matter where you launch it from.
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
echo

cd "$CODE_DIR" || { echo "Cannot cd to $CODE_DIR"; exit 1; }

FAILED=()

for SCENE in "${SCENES[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="../$LOG_DIR/${SCENE}_${ts}.log"
  run_name="pbrnerf_GlossySynthetic_${SCENE}"

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
