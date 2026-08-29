# Reproducibility

This document maps the main paper experiments to the released scripts. Source
dialogue text, full model generations, activation tensors, and model
checkpoints are not redistributed here.

## Environment

Use Python 3.10 or later and a CUDA-enabled PyTorch build appropriate for the
host GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the package versions in the release
environment, except for PyTorch, whose CUDA build must match the host system.
Install an appropriate PyTorch 2.5.1 build first when using that lock file.

The layer sweeps require access to the three Hugging Face model repositories
named in the paper. The GPT-5-mini evaluation additionally requires an
`OPENAI_API_KEY`.

## EPITOME Data and Classifiers

Clone the authors' EPITOME repository, which provides the Reddit annotation
files and ER/IP/EX classifier checkpoints:

```bash
git clone https://github.com/behavioral-data/Empathy-Mental-Health.git
git -C Empathy-Mental-Health checkout 0e11f98901527550885dbf253bd2af92dbcec43b
export EPITOME_PROJECT=/absolute/path/to/Empathy-Mental-Health
```

The original classifier defines two unused imports that were removed from
recent `transformers` releases. Apply the included compatibility patch when
using a current environment:

```bash
git -C "$EPITOME_PROJECT" apply \
  "$PWD/patches/epitome-modern-transformers.patch"
```

The EPITOME repository does not provide a separate license file at the pinned
revision. Its README provides the Reddit annotations directly and gives
separate access terms for the TalkLife portion. This project uses the Reddit
annotations only and does not redistribute their dialogue text. Users remain
responsible for following the source repository's terms. To prepare local
copies without committing the dialogue text:

```bash
bash scripts/prepare_epitome_data.sh "$EPITOME_PROJECT"
```

This copies the three Reddit annotation CSVs locally and applies the paper's
filter: seeker posts shorter than 100 characters and posts containing the
self-harm keyword list in `scripts/filter_epitome_dataset.py` are removed.
`dataset/` is ignored by Git.

## Split Reconstruction

The main layer-sweep scripts reconstruct their splits from the filtered CSVs.
For each mechanism, the default configuration uses:

- positive extraction pool: 100 rows with level at least 1, sampled with seed 42;
- negative extraction pool: 100 rows with level 0, sampled with seed 43;
- extraction-pool shuffle seed: 44; and
- 200 unique evaluation seekers disjoint from the extraction pool, sampled
  with seed 45.

Each run writes the selected extraction rows, evaluation seekers, metadata,
vectors, generations, classifier scores, and summary tables to its output
directory. These files permit exact split auditing without redistributing them
from this repository.

## Main Layer Sweeps

Set the Python interpreter explicitly if needed, then run the three mechanism
pipelines from the repository root:

```bash
export PYTHON=/absolute/path/to/python
bash scripts/run_ex11_filtered_er_pipeline.sh
bash scripts/run_ex12_filtered_ex_pipeline.sh
bash scripts/run_ex13_filtered_ip_pipeline.sh
```

The scripts reproduce the ER, EX, and IP layer/alpha sweeps across Llama,
Qwen, and Mistral. `ROOT`, `EPITOME_PROJECT`, `PYTHON`, `DATASET_PATH`,
`SAMPLE_SEED`, `EVAL_SIZE`, and other run parameters can be overridden through
environment variables documented at the top of each script.

## Paper Analysis Map

| Paper analysis | Entry point |
|---|---|
| ER/EX/IP layer sweeps | `scripts/run_ex11_filtered_er_pipeline.sh`, `run_ex12_filtered_ex_pipeline.sh`, `run_ex13_filtered_ip_pipeline.sh` |
| Residualized vectors | `scripts/run_ex15_mutual_orthogonal_complement_pipeline.sh` |
| GPT-5-mini cross-evaluator check | `scripts/run_ex16_gpt5mini_layer15_judge_pipeline.sh` |
| EmpatheticDialogues transfer | `scripts/run_ex8_ood_validation_pipeline.sh` |
| WikiText-103 perplexity | `MAX_EVAL_TOKENS=4096 bash scripts/run_ex20_language_benchmark_pipeline.sh` |
| Persona effects | `scripts/run_ex17_persona_effects_v2_pipeline.sh` |
| Paired bootstrap intervals | `scripts/run_ex22_bootstrap_ci_pipeline.sh` |
| Prompt baseline | `scripts/run_ex25_prompt_mechanism_baseline_pipeline.sh` |
| Length/question control | `scripts/run_ex26_length_question_control_pipeline.sh` |
| Decoding-seed check | `scripts/run_ex30_decoding_seed_robustness_pipeline.sh` |
| Persona-shift subspace | `scripts/run_ex32_persona_shift_subspace.sh` |
| Label-composition control | `scripts/run_ex34_same_support_comparison_pipeline.sh` |

The analysis scripts save machine-readable CSV or JSON summaries alongside
the run outputs. Paths can be overridden through their command-line arguments
or environment variables; use `python <entry-point> --help` for Python entry
points.

## Released Aggregate Artifacts

The CSV files under `artifacts/` contain aggregate values only. They do not
contain seeker posts, reference responses, generated responses, or individual
activation vectors. These files support numerical auditing without replacing
the end-to-end reproduction commands above.

`artifacts/wikitext103_ppl_4096_tokens.csv` contains the result reported in the
paper. It evaluates the first 4,096 tokens of the WikiText-103 test split. An
earlier full-split run was discarded because its hook did not apply the
intervention across all teacher-forced token positions; the released
implementation does. The command above reproduces the corrected, reported
setting. Omit `MAX_EVAL_TOKENS=4096` to run the same implementation over the
full test split.

## Reproduction Scope

Sampled generation is not guaranteed to reproduce identical strings across
GPU architectures and library versions. The intended target is procedural
and aggregate-result reproduction: the same data construction, interventions,
evaluation, and summary statistics. Exact strings can be audited from outputs
produced by each local run, but the full generations are not part of the
release artifact.
