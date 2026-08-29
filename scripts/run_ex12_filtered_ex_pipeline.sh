#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex12_filtered_ex_layer_sweep.py"

DATASET_PATH="${DATASET_PATH:-${ROOT}/dataset/filtered/explorations-reddit-filtered.csv}"
VECTOR_SIZE_PER_CLASS="${VECTOR_SIZE_PER_CLASS:-100}"
EVAL_SIZE="${EVAL_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
LEVEL_THRESHOLD="${LEVEL_THRESHOLD:-1}"
ALPHAS="${ALPHAS:--1 0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_PREFIX="${RUN_PREFIX:-ex12_filtered_ex}"
OUT_DIR="${ROOT}/outputs/ex12_filtered_ex"
LOG_DIR="${ROOT}/logs"
PIPELINE_LOG="${LOG_DIR}/${RUN_PREFIX}_pipeline.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

read -r -a ALPHAS_ARR <<< "${ALPHAS}"
if [[ "${#ALPHAS_ARR[@]}" -eq 0 ]]; then
  echo "[error] ALPHAS is empty. expected example: '-1 0 1 2'" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

run_one() {
  local tag="$1"
  local model_name="$2"
  local layers="$3"
  local trust_remote_code="$4"

  local run_name="${RUN_PREFIX}_${tag}_n${EVAL_SIZE}"
  local log_file="${LOG_DIR}/${run_name}.log"

  echo "[start] $(date '+%F %T') tag=${tag} model=${model_name} layers=${layers}" | tee -a "${PIPELINE_LOG}"

  local -a cmd=(
    "${PY}" "${SCRIPT}"
    --persona-project "${ROOT}"
    --epitome-project "${EPITOME}"
    --dataset-path "${DATASET_PATH}"
    --vector-size-per-class "${VECTOR_SIZE_PER_CLASS}"
    --eval-size "${EVAL_SIZE}"
    --sample-seed "${SAMPLE_SEED}"
    --level-threshold "${LEVEL_THRESHOLD}"
    --model-name "${model_name}"
    --layers "${layers}"
    --alphas "${ALPHAS_ARR[@]}"
    --batch-size "${BATCH_SIZE}"
    --output-dir "outputs/ex12_filtered_ex"
    --run-name "${run_name}"
  )

  if [[ "${trust_remote_code}" == "true" ]]; then
    cmd+=(--trust-remote-code)
  fi

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  echo "[done] tag=${tag} run_name=${run_name} log=${log_file}" | tee -a "${PIPELINE_LOG}"
}

echo "[start] $(date '+%F %T') run_prefix=${RUN_PREFIX}" | tee "${PIPELINE_LOG}"
echo "[info] dataset=${DATASET_PATH}" | tee -a "${PIPELINE_LOG}"
echo "[info] vector_size_per_class=${VECTOR_SIZE_PER_CLASS} eval_size=${EVAL_SIZE} alphas=${ALPHAS}" | tee -a "${PIPELINE_LOG}"

# Llama (32 layers)
run_one \
  "llama31" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "3,7,11,15,19,23,27,31" \
  "false"

# Qwen2.5 (28 layers => max 27)
run_one \
  "qwen25" \
  "Qwen/Qwen2.5-7B-Instruct" \
  "3,7,11,15,19,23,27" \
  "true"

# Mistral (32 layers)
run_one \
  "mistral7b" \
  "mistralai/Mistral-7B-Instruct-v0.3" \
  "3,7,11,15,19,23,27,31" \
  "false"

echo "[done] $(date '+%F %T') Ex12 pipeline finished" | tee -a "${PIPELINE_LOG}"
echo "[done] pipeline_log=${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
