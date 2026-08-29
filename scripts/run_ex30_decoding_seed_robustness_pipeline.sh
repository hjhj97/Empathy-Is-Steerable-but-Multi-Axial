#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PY="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"

VECTOR_SIZE_PER_CLASS="${VECTOR_SIZE_PER_CLASS:-100}"
EVAL_SIZE="${EVAL_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
DECODING_SEEDS="${DECODING_SEEDS:-22 32}"
ALPHAS="${ALPHAS:--1 0 1}"
LAYER="${LAYER:-15}"
LEVEL_THRESHOLD="${LEVEL_THRESHOLD:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_PREFIX="${RUN_PREFIX:-ex30}"
ANALYZE_AFTER="${ANALYZE_AFTER:-1}"

OUT_ROOT="${ROOT}/outputs/ex30_decoding_seed_robustness"
LOG_DIR="${ROOT}/logs"
PIPELINE_LOG="${LOG_DIR}/${RUN_PREFIX}_decoding_seed_robustness_pipeline.log"

mkdir -p "${OUT_ROOT}/ex11" "${OUT_ROOT}/ex12" "${OUT_ROOT}/ex13" "${LOG_DIR}"
cd "${ROOT}"

read -r -a ALPHAS_ARR <<< "${ALPHAS}"
read -r -a DECODING_SEEDS_ARR <<< "${DECODING_SEEDS}"
if [[ "${#ALPHAS_ARR[@]}" -eq 0 ]]; then
  echo "[error] ALPHAS is empty. expected example: '-1 0 1'" >&2
  exit 1
fi
if [[ "${#DECODING_SEEDS_ARR[@]}" -eq 0 ]]; then
  echo "[error] DECODING_SEEDS is empty. expected example: '22 32'" >&2
  exit 1
fi

run_mechanism() {
  local exp_tag="$1"
  local script_name="$2"
  local dataset_rel="$3"
  local seed="$4"

  local script_path="${ROOT}/src/${script_name}"
  local dataset_path="${ROOT}/${dataset_rel}"
  local out_rel="outputs/ex30_decoding_seed_robustness/${exp_tag}"
  local run_name="${RUN_PREFIX}_${exp_tag}_llama31_l${LAYER}_decseed${seed}_n${EVAL_SIZE}"
  local log_file="${LOG_DIR}/${run_name}.log"

  if [[ ! -f "${script_path}" ]]; then
    echo "[error] script not found: ${script_path}" >&2
    exit 1
  fi
  if [[ ! -f "${dataset_path}" ]]; then
    echo "[error] dataset not found: ${dataset_path}" >&2
    exit 1
  fi

  echo "[start] $(date '+%F %T') ${exp_tag} decoding_seed=${seed} run=${run_name}" | tee -a "${PIPELINE_LOG}"
  "${PY}" "${script_path}" \
    --persona-project "${ROOT}" \
    --epitome-project "${EPITOME}" \
    --dataset-path "${dataset_path}" \
    --model-name "${MODEL_NAME}" \
    --layers "${LAYER}" \
    --alphas "${ALPHAS_ARR[@]}" \
    --vector-size-per-class "${VECTOR_SIZE_PER_CLASS}" \
    --eval-size "${EVAL_SIZE}" \
    --sample-seed "${SAMPLE_SEED}" \
    --seed "${seed}" \
    --level-threshold "${LEVEL_THRESHOLD}" \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "${out_rel}" \
    --run-name "${run_name}" \
    2>&1 | tee "${log_file}"
  echo "[done] $(date '+%F %T') ${exp_tag} decoding_seed=${seed} log=${log_file}" | tee -a "${PIPELINE_LOG}"
}

echo "[start] $(date '+%F %T') Ex30 decoding-seed robustness" | tee "${PIPELINE_LOG}"
echo "[info] layer=${LAYER} alphas=${ALPHAS} decoding_seeds=${DECODING_SEEDS}" | tee -a "${PIPELINE_LOG}"
echo "[info] sample_seed=${SAMPLE_SEED} eval_size=${EVAL_SIZE} vector_size_per_class=${VECTOR_SIZE_PER_CLASS}" | tee -a "${PIPELINE_LOG}"

for seed in "${DECODING_SEEDS_ARR[@]}"; do
  run_mechanism \
    "ex11" \
    "run_ex11_filtered_er_layer_sweep.py" \
    "dataset/filtered/emotional-reactions-reddit-filtered.csv" \
    "${seed}"
  run_mechanism \
    "ex12" \
    "run_ex12_filtered_ex_layer_sweep.py" \
    "dataset/filtered/explorations-reddit-filtered.csv" \
    "${seed}"
  run_mechanism \
    "ex13" \
    "run_ex13_filtered_ip_layer_sweep.py" \
    "dataset/filtered/interpretations-reddit-filtered.csv" \
    "${seed}"
done

if [[ "${ANALYZE_AFTER}" == "1" ]]; then
  echo "[start] $(date '+%F %T') Ex30 analysis" | tee -a "${PIPELINE_LOG}"
  "${PY}" "${ROOT}/src/analyze_ex30_decoding_seed_robustness.py" \
    --persona-project "${ROOT}" \
    --layer "${LAYER}" \
    --required-seeds 12 "${DECODING_SEEDS_ARR[@]}" \
    --output-dir "outputs/ex30_decoding_seed_robustness" \
    --run-name "${RUN_PREFIX}_decoding_seed_robustness_l${LAYER}" \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/${RUN_PREFIX}_decoding_seed_robustness_analysis.log"
fi

echo "[done] $(date '+%F %T') Ex30 pipeline finished" | tee -a "${PIPELINE_LOG}"
echo "[done] pipeline_log=${PIPELINE_LOG}" | tee -a "${PIPELINE_LOG}"
