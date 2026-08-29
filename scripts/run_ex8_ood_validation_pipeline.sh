#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex8_ood_validation.py"

ED_SOURCE="${ED_SOURCE:-auto}"
ED_PATH="${ED_PATH:-${ROOT}/dataset/empatheticdialogues_test.csv}"
HF_DATASET_NAME="${HF_DATASET_NAME:-empathetic_dialogues}"
HF_SPLIT="${HF_SPLIT:-test}"
CONTEXT_COLUMN="${CONTEXT_COLUMN:-context}"
PROMPT_COLUMN="${PROMPT_COLUMN:-prompt}"
CONV_ID_COLUMN="${CONV_ID_COLUMN:-conv_id}"
NEGATIVE_EMOTIONS="${NEGATIVE_EMOTIONS:-afraid,angry,anxious,sad,devastated,lonely,terrified}"
EVAL_SIZE="${EVAL_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

TARGET_LAYER="${TARGET_LAYER:-15}"
ALPHAS="${ALPHAS:--1 0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_PREFIX="${RUN_PREFIX:-ex8_ood_validation}"
OUT_DIR="${ROOT}/outputs/ex8_ood_validation"
LOG_DIR="${ROOT}/logs"
PIPELINE_LOG="${LOG_DIR}/${RUN_PREFIX}_pipeline.log"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

read -r -a ALPHAS_ARR <<< "${ALPHAS}"
if [[ "${#ALPHAS_ARR[@]}" -eq 0 ]]; then
  echo "[error] ALPHAS is empty. expected example: '-1 0 1 2'" >&2
  exit 1
fi

run_one() {
  local tag="$1"
  local model_name="$2"
  local trust_remote_code="$3"
  local vector_path="$4"

  local run_name="${RUN_PREFIX}_${tag}_n${EVAL_SIZE}"
  local log_file="${LOG_DIR}/${run_name}.log"

  echo "[start] $(date '+%F %T') tag=${tag} model=${model_name}" | tee -a "${PIPELINE_LOG}"

  if [[ ! -f "${vector_path}" ]]; then
    echo "[error] vector not found for tag=${tag}: ${vector_path}" | tee -a "${PIPELINE_LOG}"
    exit 1
  fi

  local -a cmd=(
    "${PY}" "${SCRIPT}"
    --persona-project "${ROOT}"
    --epitome-project "${EPITOME}"
    --ed-source "${ED_SOURCE}"
    --ed-path "${ED_PATH}"
    --hf-dataset-name "${HF_DATASET_NAME}"
    --hf-split "${HF_SPLIT}"
    --context-column "${CONTEXT_COLUMN}"
    --prompt-column "${PROMPT_COLUMN}"
    --conv-id-column "${CONV_ID_COLUMN}"
    --negative-emotions "${NEGATIVE_EMOTIONS}"
    --eval-size "${EVAL_SIZE}"
    --sample-seed "${SAMPLE_SEED}"
    --vector-path "${vector_path}"
    --target-layer "${TARGET_LAYER}"
    --alphas "${ALPHAS_ARR[@]}"
    --model-name "${model_name}"
    --batch-size "${BATCH_SIZE}"
    --output-dir "outputs/ex8_ood_validation"
    --run-name "${run_name}"
  )

  if [[ "${trust_remote_code}" == "true" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  echo "[done] tag=${tag} run_name=${run_name} log=${log_file}" | tee -a "${PIPELINE_LOG}"
}

echo "[start] $(date '+%F %T') run_prefix=${RUN_PREFIX}" | tee "${PIPELINE_LOG}"
echo "[info] ed_source=${ED_SOURCE} ed_path=${ED_PATH} hf_dataset=${HF_DATASET_NAME} split=${HF_SPLIT}" | tee -a "${PIPELINE_LOG}"
echo "[info] target_layer=${TARGET_LAYER} eval_size=${EVAL_SIZE} alphas=${ALPHAS}" | tee -a "${PIPELINE_LOG}"
echo "[info] negative_emotions=${NEGATIVE_EMOTIONS}" | tee -a "${PIPELINE_LOG}"

run_one \
  "llama31" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "false" \
  "${ROOT}/outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_layer_vectors.npz"

run_one \
  "qwen25" \
  "Qwen/Qwen2.5-7B-Instruct" \
  "true" \
  "${ROOT}/outputs/ex11_filtered_er/ex11_filtered_er_qwen25_n200_layer_vectors.npz"

run_one \
  "mistral7b" \
  "mistralai/Mistral-7B-Instruct-v0.3" \
  "false" \
  "${ROOT}/outputs/ex11_filtered_er/ex11_filtered_er_mistral7b_n200_layer_vectors.npz"

echo "[done] $(date '+%F %T') Ex8 pipeline finished" | tee -a "${PIPELINE_LOG}"
echo "[done] pipeline_log=${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
