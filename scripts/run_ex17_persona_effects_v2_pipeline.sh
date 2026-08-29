#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex17_persona_effects_v2.py"

RUN_PREFIX="${RUN_PREFIX:-ex17_persona_effects_v2}"
EVAL_SIZE="${EVAL_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-32}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-15}"
ACTIVATION_POOL="${ACTIVATION_POOL:-mean}"

PERSONAS="${PERSONAS:-person,empathetic person,cynical person,white person,black person,male,female,psychotherapist,engineer,Republican supporter,Democratic supporter}"
BASELINE_PERSONA="${BASELINE_PERSONA:-person}"

OUT_DIR="${ROOT}/outputs/ex17_persona_effects_v2"
LOG_DIR="${ROOT}/logs"
PIPELINE_LOG="${LOG_DIR}/${RUN_PREFIX}_pipeline.log"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

SKIP_PROJECTION_FLAG=""
if [[ "${SKIP_PROJECTION:-0}" == "1" ]]; then
  SKIP_PROJECTION_FLAG="--skip-projection"
fi

cd "${ROOT}"

run_one() {
  local tag="$1"
  local model_name="$2"
  local trust_remote_code="$3"
  local er_vec="$4"
  local ex_vec="$5"
  local ip_vec="$6"

  local run_name="${RUN_PREFIX}_${tag}_n${EVAL_SIZE}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local trust_flag=""
  if [[ "${trust_remote_code}" == "true" ]]; then
    trust_flag="--trust-remote-code"
  fi

  echo "[start] $(date '+%F %T') tag=${tag} model=${model_name}" | tee -a "${PIPELINE_LOG}"
  echo "[info] run_name=${run_name} personas=${PERSONAS}" | tee -a "${PIPELINE_LOG}"

  "${PY}" "${SCRIPT}" \
    --persona-project "${ROOT}" \
    --epitome-project "${EPITOME}" \
    --dataset-path "dataset/filtered/emotional-reactions-reddit-filtered.csv" \
    --eval-seekers-path "outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_eval_seekers.csv" \
    --eval-size "${EVAL_SIZE}" \
    --sample-seed "${SAMPLE_SEED}" \
    --model-name "${model_name}" \
    --batch-size "${BATCH_SIZE}" \
    --classifier-batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --activation-layer "${ACTIVATION_LAYER}" \
    --activation-pool "${ACTIVATION_POOL}" \
    --personas "${PERSONAS}" \
    --baseline-persona "${BASELINE_PERSONA}" \
    --er-vector-path "${er_vec}" \
    --ex-vector-path "${ex_vec}" \
    --ip-vector-path "${ip_vec}" \
    --output-dir "outputs/ex17_persona_effects_v2" \
    --run-name "${run_name}" \
    ${OVERWRITE_FLAG} \
    ${SKIP_PROJECTION_FLAG} \
    ${trust_flag} \
    2>&1 | tee "${log_file}"

  echo "[done] tag=${tag} run_name=${run_name} log=${log_file}" | tee -a "${PIPELINE_LOG}"
}

echo "[start] $(date '+%F %T') Ex17 pipeline" | tee "${PIPELINE_LOG}"
echo "[info] eval_size=${EVAL_SIZE} activation_layer=${ACTIVATION_LAYER}" | tee -a "${PIPELINE_LOG}"
echo "[info] output_dir=${OUT_DIR}" | tee -a "${PIPELINE_LOG}"

run_one \
  "llama31" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "false" \
  "outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_layer_vectors.npz" \
  "outputs/ex12_filtered_ex/ex12_filtered_ex_llama31_n200_layer_vectors.npz" \
  "outputs/ex13_filtered_ip/ex13_filtered_ip_llama31_n200_layer_vectors.npz"

run_one \
  "qwen25" \
  "Qwen/Qwen2.5-7B-Instruct" \
  "true" \
  "outputs/ex11_filtered_er/ex11_filtered_er_qwen25_n200_layer_vectors.npz" \
  "outputs/ex12_filtered_ex/ex12_filtered_ex_qwen25_n200_layer_vectors.npz" \
  "outputs/ex13_filtered_ip/ex13_filtered_ip_qwen25_n200_layer_vectors.npz"

run_one \
  "mistral7b" \
  "mistralai/Mistral-7B-Instruct-v0.3" \
  "false" \
  "outputs/ex11_filtered_er/ex11_filtered_er_mistral7b_n200_layer_vectors.npz" \
  "outputs/ex12_filtered_ex/ex12_filtered_ex_mistral7b_n200_layer_vectors.npz" \
  "outputs/ex13_filtered_ip/ex13_filtered_ip_mistral7b_n200_layer_vectors.npz"

echo "[done] $(date '+%F %T') Ex17 pipeline finished" | tee -a "${PIPELINE_LOG}"
echo "[done] pipeline_log=${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
