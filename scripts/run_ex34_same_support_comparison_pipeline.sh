#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EPITOME="${EPITOME_PROJECT:-${ROOT}/../Empathy-Mental-Health}"
PYTHON="${PYTHON:-python}"

cd "${ROOT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs \
  outputs/rebuttal_priority1_controlled_{er,ex,ip} \
  outputs/rebuttal_priority1_unmatched_{er,ex,ip} \
  outputs/rebuttal_priority1_controlled_er_normmatched \
  outputs/ex34_exact_stratum_control

echo "[pipeline-v3] start=$(date --iso-8601=seconds) gpu=${CUDA_VISIBLE_DEVICES:-default}"
"${PYTHON}" scripts/build_ex34_joint_labels.py
"${PYTHON}" scripts/prepare_ex34_exact_stratum_pools.py

run_arm() {
  local mechanism="$1"
  local experiment="$2"
  local runner="$3"
  local arm="$4"
  local dataset_path="$5"
  local output_dir="$6"
  local run_name="$7"
  local lower="${mechanism,,}"

  echo "[pipeline-v3] ${mechanism} ${arm} start=$(date --iso-8601=seconds)"
  "${PYTHON}" "src/${runner}" \
    --persona-project "${ROOT}" \
    --epitome-project "${EPITOME}" \
    --dataset-path "${dataset_path}" \
    --eval-seekers-path \
      "outputs/${experiment}_filtered_${lower}/${experiment}_filtered_${lower}_llama31_n200_eval_seekers.csv" \
    --run-name "${run_name}" \
    --output-dir "${output_dir}" \
    --layers 15 --alphas -1 0 1 \
    --vector-size-per-class 100 --eval-size 200 \
    --sample-seed 42 --level-threshold 1 --seed 12 \
    --max-new-tokens 128 --temperature 0.8 --top-p 0.9 \
    --repetition-penalty 1.2
  echo "[pipeline-v3] ${mechanism} ${arm} done=$(date --iso-8601=seconds)"
}

for spec in \
  "ER ex11 run_ex11_filtered_er_layer_sweep.py" \
  "EX ex12 run_ex12_filtered_ex_layer_sweep.py" \
  "IP ex13 run_ex13_filtered_ip_layer_sweep.py"; do
  read -r mechanism experiment runner <<< "${spec}"
  lower="${mechanism,,}"

  run_arm \
    "${mechanism}" "${experiment}" "${runner}" "matched" \
    "outputs/rebuttal_priority1_controlled_inputs_v2/rebuttal_p1_filtered_${lower}_ge1_exact_evalexcl_compat.csv" \
    "outputs/rebuttal_priority1_controlled_${lower}" \
    "rebuttal_p1_filtered_${lower}_ge1_exact_l15_a-1_0_1_n200_v2"

  run_arm \
    "${mechanism}" "${experiment}" "${runner}" "same-support-unmatched" \
    "outputs/rebuttal_priority1_controlled_inputs_v2/rebuttal_p1_filtered_${lower}_ge1_unmatched_evalexcl_compat.csv" \
    "outputs/rebuttal_priority1_unmatched_${lower}" \
    "rebuttal_p1_filtered_${lower}_ge1_unmatched_l15_a-1_0_1_n200_v3"
done

ER_NORM_ALPHA=$("${PYTHON}" -c '
import numpy as np
def load(path):
    data = np.load(path)
    index = list(data["layers"].astype(int)).index(15)
    return data["vectors"][index].astype(float)
controlled = load("outputs/rebuttal_priority1_controlled_er/rebuttal_p1_filtered_er_ge1_exact_l15_a-1_0_1_n200_v2_layer_vectors.npz")
unmatched = load("outputs/rebuttal_priority1_unmatched_er/rebuttal_p1_filtered_er_ge1_unmatched_l15_a-1_0_1_n200_v3_layer_vectors.npz")
print(np.linalg.norm(unmatched) / np.linalg.norm(controlled))
')
echo "[pipeline-v3] ER norm-matched alpha=${ER_NORM_ALPHA}"

"${PYTHON}" src/run_ex11_filtered_er_layer_sweep.py \
  --persona-project "${ROOT}" --epitome-project "${EPITOME}" \
  --dataset-path outputs/rebuttal_priority1_controlled_inputs_v2/rebuttal_p1_filtered_er_ge1_exact_evalexcl_compat.csv \
  --eval-seekers-path outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_eval_seekers.csv \
  --run-name rebuttal_p1_filtered_er_ge1_exact_l15_normmatched_n200_v3 \
  --output-dir outputs/rebuttal_priority1_controlled_er_normmatched \
  --layers 15 --alphas 0 "${ER_NORM_ALPHA}" \
  --vector-size-per-class 100 --eval-size 200 \
  --sample-seed 42 --level-threshold 1 --seed 12 \
  --max-new-tokens 128 --temperature 0.8 --top-p 0.9 \
  --repetition-penalty 1.2

"${PYTHON}" scripts/analyze_ex34_exact_stratum_control.py \
  --comparison-arm same_support \
  --bootstrap-iters 10000 \
  --seed 42 \
  --run-name ex34_same_support_comparison_llama31_n200_v3

echo "[pipeline-v3] complete=$(date --iso-8601=seconds)"
