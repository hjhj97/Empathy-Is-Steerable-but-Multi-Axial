#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex20_language_benchmark_ppl.py"

RUN_PREFIX="${RUN_PREFIX:-ex20_wikitext103_ppl}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
TARGET_LAYER="${TARGET_LAYER:-15}"
ALPHAS="${ALPHAS:--1 1 2}"
MECHANISMS="${MECHANISMS:-ER,EX,IP}"

ER_VECTOR_PATH="${ER_VECTOR_PATH:-outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_layer_vectors.npz}"
EX_VECTOR_PATH="${EX_VECTOR_PATH:-outputs/ex12_filtered_ex/ex12_filtered_ex_llama31_n200_layer_vectors.npz}"
IP_VECTOR_PATH="${IP_VECTOR_PATH:-outputs/ex13_filtered_ip/ex13_filtered_ip_llama31_n200_layer_vectors.npz}"

DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-103-v1}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
LOCAL_TEXT_PATH="${LOCAL_TEXT_PATH:-}"

MAX_TEXT_ROWS="${MAX_TEXT_ROWS:-0}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
STRIDE="${STRIDE:-512}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-0}"

OUT_DIR="${ROOT}/outputs/ex20_language_benchmark"
LOG_DIR="${ROOT}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

RUN_NAME="${RUN_PREFIX}_llama31"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

TRUST_REMOTE_CODE_FLAG=""
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_REMOTE_CODE_FLAG="--trust-remote-code"
fi

NORMALIZE_VECTORS_FLAG=""
if [[ "${NORMALIZE_VECTORS:-0}" == "1" ]]; then
  NORMALIZE_VECTORS_FLAG="--normalize-loaded-vectors"
fi

ADD_SPECIAL_TOKENS_FLAG=""
if [[ "${ADD_SPECIAL_TOKENS:-0}" == "1" ]]; then
  ADD_SPECIAL_TOKENS_FLAG="--add-special-tokens"
fi

read -r -a ALPHAS_ARR <<< "${ALPHAS}"
if [[ "${#ALPHAS_ARR[@]}" -eq 0 ]]; then
  echo "[error] ALPHAS is empty. expected example: '-1 1 2'" >&2
  exit 1
fi

LOCAL_TEXT_FLAG=()
if [[ -n "${LOCAL_TEXT_PATH}" ]]; then
  LOCAL_TEXT_FLAG=(--local-text-path "${LOCAL_TEXT_PATH}")
fi

echo "[start] $(date '+%F %T') Ex20 pipeline"
echo "[info] run_name=${RUN_NAME}"
echo "[info] model=${MODEL_NAME} layer=${TARGET_LAYER} mechanisms=${MECHANISMS} alphas=${ALPHAS}"
echo "[info] dataset=${DATASET_NAME}/${DATASET_CONFIG}:${DATASET_SPLIT} local_text_path=${LOCAL_TEXT_PATH:-<none>}"
echo "[info] output_dir=${OUT_DIR}"
echo "[info] log=${LOG_FILE}"

cd "${ROOT}"

"${PY}" "${SCRIPT}" \
  --persona-project "${ROOT}" \
  --model-name "${MODEL_NAME}" \
  --target-layer "${TARGET_LAYER}" \
  --alphas "${ALPHAS_ARR[@]}" \
  --mechanisms "${MECHANISMS}" \
  --er-vector-path "${ER_VECTOR_PATH}" \
  --ex-vector-path "${EX_VECTOR_PATH}" \
  --ip-vector-path "${IP_VECTOR_PATH}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-config "${DATASET_CONFIG}" \
  --dataset-split "${DATASET_SPLIT}" \
  "${LOCAL_TEXT_FLAG[@]}" \
  --max-text-rows "${MAX_TEXT_ROWS}" \
  --max-length "${MAX_LENGTH}" \
  --stride "${STRIDE}" \
  --max-eval-tokens "${MAX_EVAL_TOKENS}" \
  --output-dir "outputs/ex20_language_benchmark" \
  --run-name "${RUN_NAME}" \
  ${TRUST_REMOTE_CODE_FLAG} \
  ${NORMALIZE_VECTORS_FLAG} \
  ${ADD_SPECIAL_TOKENS_FLAG} \
  2>&1 | tee "${LOG_FILE}"

echo "[done] $(date '+%F %T') Ex20 pipeline finished"
echo "[done] run_name=${RUN_NAME}"
echo "[done] log=${LOG_FILE}"
