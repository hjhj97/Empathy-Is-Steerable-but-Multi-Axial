#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex15_mutual_orthogonal_complement.py"

LAYERS="${LAYERS:-3,7,11,15,19,23,27,31}"
ALPHAS="${ALPHAS:--1 0 1 2}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-matched}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_PREFIX="${RUN_PREFIX:-ex15_mutual_orthogonal_complement}"
OUT_DIR="${ROOT}/outputs/ex15_mutual_orthogonal_complement"
LOG_DIR="${ROOT}/logs"
PIPELINE_LOG="${LOG_DIR}/${RUN_PREFIX}_pipeline.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

read -r -a ALPHAS_ARR <<< "${ALPHAS}"
if [[ "${#ALPHAS_ARR[@]}" -eq 0 ]]; then
  echo "[error] ALPHAS is empty. expected example: '-1 0 1 2'" >&2
  exit 1
fi

echo "[start] $(date '+%F %T') run_prefix=${RUN_PREFIX}" | tee "${PIPELINE_LOG}"
echo "[info] layers=${LAYERS} alphas=${ALPHAS} residual_scale=${RESIDUAL_SCALE}" | tee -a "${PIPELINE_LOG}"

cmd=(
  "${PY}" "${SCRIPT}"
  --persona-project "${ROOT}"
  --epitome-project "${EPITOME}"
  --layers "${LAYERS}"
  --alphas "${ALPHAS_ARR[@]}"
  --residual-scale "${RESIDUAL_SCALE}"
  --batch-size "${BATCH_SIZE}"
  --output-dir "outputs/ex15_mutual_orthogonal_complement"
)

if [[ "${SKIP_STEERING:-0}" == "1" ]]; then
  cmd+=(--skip-steering)
fi

if [[ "${SKIP_CLASSIFICATION:-0}" == "1" ]]; then
  cmd+=(--skip-classification)
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}.log"
echo "[done] $(date '+%F %T') Ex15 pipeline finished" | tee -a "${PIPELINE_LOG}"
echo "[done] pipeline_log=${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
