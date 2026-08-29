#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex25_prompt_mechanism_baseline.py"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
RUN_PREFIX="${RUN_PREFIX:-ex25_prompt_mechanism_baseline}"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_llama31_n200}"

EVAL_SIZE="${EVAL_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SEED="${SEED:-12}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"

STEERING_LAYER="${STEERING_LAYER:-15}"
STEERING_ALPHA="${STEERING_ALPHA:-1.0}"

OUT_DIR="${ROOT}/outputs/ex25_prompt_mechanism_baseline"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

"${PY}" "${SCRIPT}" \
  --persona-project "${ROOT}" \
  --epitome-project "${EPITOME}" \
  --model-name "${MODEL_NAME}" \
  --eval-size "${EVAL_SIZE}" \
  --sample-seed "${SAMPLE_SEED}" \
  --seed "${SEED}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --steering-layer "${STEERING_LAYER}" \
  --steering-alpha "${STEERING_ALPHA}" \
  --output-dir "outputs/ex25_prompt_mechanism_baseline" \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "[done] log=${LOG_FILE}"
