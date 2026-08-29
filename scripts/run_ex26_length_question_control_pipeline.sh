#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY="${PYTHON:-python}"
SCRIPT="${ROOT}/src/run_ex26_length_question_control.py"

RUN_NAME="${RUN_NAME:-ex26_length_question_control_n200}"
OUT_DIR="${ROOT}/outputs/ex26_length_question_control"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

LAYER="${LAYER:-15}"
BASELINE_ALPHA="${BASELINE_ALPHA:-0}"
STEERING_ALPHA="${STEERING_ALPHA:-1}"
PROMPT_BASELINE="${PROMPT_BASELINE:-P0}"
PROMPT_TARGETS="${PROMPT_TARGETS:-P1,P2,P3}"
METRICS="${METRICS:-ER_label,IP_label,EX_label}"
LENGTH_FEATURE="${LENGTH_FEATURE:-word_len}"
LENGTH_BINS="${LENGTH_BINS:-5}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
SEED="${SEED:-42}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

"${PY}" "${SCRIPT}" \
  --persona-project "${ROOT}" \
  --layer "${LAYER}" \
  --baseline-alpha "${BASELINE_ALPHA}" \
  --steering-alpha "${STEERING_ALPHA}" \
  --prompt-baseline-condition "${PROMPT_BASELINE}" \
  --prompt-target-conditions "${PROMPT_TARGETS}" \
  --metrics "${METRICS}" \
  --length-feature "${LENGTH_FEATURE}" \
  --length-bins "${LENGTH_BINS}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}" \
  --seed "${SEED}" \
  --output-dir "outputs/ex26_length_question_control" \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "[done] log=${LOG_FILE}"
