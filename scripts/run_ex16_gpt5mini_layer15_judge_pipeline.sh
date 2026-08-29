#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex16_gpt5mini_layer15_judge_batch.py"

RUN_PREFIX="${RUN_PREFIX:-ex16_gpt5mini_layer15_judge}"
RUN_NAME="${RUN_PREFIX}"
MODE="${MODE:-run}"                           # submit | wait | run
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
LAYER="${LAYER:-15}"
ALPHAS="${ALPHAS:--1,0,1,2}"
TASKS="${TASKS:-ER,EX,IP}"
MODEL_TAGS="${MODEL_TAGS:-llama31,qwen25,mistral7b}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_WAIT_MINUTES="${MAX_WAIT_MINUTES:-180}"
EXPECTED_ROWS="${EXPECTED_ROWS:-7200}"

OUT_DIR="${ROOT}/outputs/ex16_gpt5mini_layer15_judge"
LOG_DIR="${ROOT}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

BATCH_ID_FLAG=""
if [[ -n "${BATCH_ID:-}" ]]; then
  BATCH_ID_FLAG="--batch-id ${BATCH_ID}"
fi

echo "[start] $(date '+%F %T') Ex16 pipeline"
echo "[info] mode=${MODE} model=${JUDGE_MODEL} layer=${LAYER} alphas=${ALPHAS}"
echo "[info] tasks=${TASKS} model_tags=${MODEL_TAGS}"
echo "[info] expected_rows=${EXPECTED_ROWS}"
echo "[info] output_dir=${OUT_DIR}"
echo "[info] log=${LOG_FILE}"

cd "${ROOT}"

"${PY}" "${SCRIPT}" \
  --mode "${MODE}" \
  --persona-project "${ROOT}" \
  --output-dir "outputs/ex16_gpt5mini_layer15_judge" \
  --run-name "${RUN_NAME}" \
  --model "${JUDGE_MODEL}" \
  --layer "${LAYER}" \
  --alphas="${ALPHAS}" \
  --tasks "${TASKS}" \
  --model-tags "${MODEL_TAGS}" \
  --poll-interval "${POLL_INTERVAL}" \
  --max-wait-minutes "${MAX_WAIT_MINUTES}" \
  --expected-rows "${EXPECTED_ROWS}" \
  ${BATCH_ID_FLAG} \
  ${OVERWRITE_FLAG} \
  2>&1 | tee "${LOG_FILE}"

echo "[done] $(date '+%F %T') Ex16 pipeline finished"
echo "[done] run_name=${RUN_NAME}"
echo "[done] log=${LOG_FILE}"
