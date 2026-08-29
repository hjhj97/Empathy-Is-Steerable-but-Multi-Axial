#!/usr/bin/env python3
"""Export text-free aggregate results for the public artifact release."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"

MODEL_TAGS = ("llama31", "qwen25", "mistral7b")
MECHANISMS = ("ER", "EX", "IP")
EXPERIMENT_BY_MECHANISM = {"ER": "ex11", "EX": "ex12", "IP": "ex13"}


def read(relative_path: str) -> pd.DataFrame:
    path = OUTPUTS / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write(frame: pd.DataFrame, filename: str) -> None:
    forbidden_columns = {
        "seeker_post",
        "response_post",
        "generated_response",
        "prompt_text",
        "classified_csv",
        "summary_csv",
        "vector_path",
        "dataset_path",
    }
    overlap = forbidden_columns.intersection(frame.columns)
    if overlap:
        raise ValueError(f"unsafe columns in {filename}: {sorted(overlap)}")

    rendered = frame.to_csv(index=False)
    for marker in ("/home/", "OPENAI_API_KEY", "sk-"):
        if marker in rendered:
            raise ValueError(f"unsafe marker {marker!r} in {filename}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / filename).write_text(rendered, encoding="utf-8")


def export_main_layer_sweep() -> None:
    frames = []
    for mechanism in MECHANISMS:
        experiment = EXPERIMENT_BY_MECHANISM[mechanism]
        directory = f"{experiment}_filtered_{mechanism.lower()}"
        for model in MODEL_TAGS:
            frame = read(
                f"{directory}/{experiment}_filtered_{mechanism.lower()}_"
                f"{model}_n200_summary.csv"
            )
            frame.insert(0, "model", model)
            frame.insert(1, "steering_mechanism", mechanism)
            frames.append(frame)
    write(pd.concat(frames, ignore_index=True), "main_layer_sweep.csv")


def export_cross_mechanism() -> None:
    frame = read(
        "ex29_cross_mechanism_off_target/"
        "ex29_cross_mechanism_off_target_l15_a-1_a1_off_target_summary.csv"
    )
    write(frame.drop(columns=["classified_csv"]), "layer15_cross_mechanism.csv")


def export_geometry_and_residualization() -> None:
    original = read(
        "vector_similarity_ex11_ex12_ex13/"
        "er_ex_ip_vector_similarity_per_layer.csv"
    )
    original = original.loc[original["layer"] == 15].copy()
    original.insert(2, "condition", "original")

    residual = read(
        "ex15_mutual_orthogonal_complement/"
        "ex15_residualized_pairwise_cosine_per_layer.csv"
    )
    residual = residual.loc[residual["layer"] == 15].copy()
    residual.insert(0, "model", "llama31")
    residual.insert(2, "condition", "residualized")
    write(pd.concat([original, residual], ignore_index=True), "vector_geometry.csv")

    comparison = read(
        "ex15_mutual_orthogonal_complement/"
        "ex15_original_vs_residualized_comparison.csv"
    )
    write(
        comparison.loc[comparison["layer"] == 15].copy(),
        "residualization_layer15.csv",
    )


def export_cross_evaluator() -> None:
    frame = read(
        "ex16_gpt5mini_layer15_judge/"
        "ex16_gpt5mini_layer15_judge_summary_by_task_model_alpha.csv"
    )
    write(frame, "gpt5mini_cross_evaluator.csv")


def export_persona() -> None:
    frames = []
    for model in MODEL_TAGS:
        frame = read(
            "ex17_persona_effects_v2/"
            f"ex17_persona_effects_v2_{model}_n200_summary_vs_person.csv"
        )
        frame.insert(0, "model", model)
        frames.append(frame)
    write(pd.concat(frames, ignore_index=True), "persona_effects.csv")
    write(
        read("ex32_persona_shift_subspace/ex32_model_summary.csv"),
        "persona_subspace.csv",
    )


def export_prompt_and_style_controls() -> None:
    frames = []
    for model in MODEL_TAGS:
        frame = read(
            "ex25_prompt_mechanism_baseline/"
            f"ex25_prompt_mechanism_baseline_{model}_n200_v2_summary_vs_baseline.csv"
        )
        frame.insert(0, "model", model)
        frames.append(frame)
    write(pd.concat(frames, ignore_index=True), "prompt_baseline.csv")
    write(
        read(
            "ex26_length_question_control/"
            "ex26_length_question_control_n200_bootstrap_summary.csv"
        ),
        "length_question_control.csv",
    )


def export_transfer() -> None:
    frames = []
    for model in MODEL_TAGS:
        frame = read(
            f"ex8_ood_validation/ex8_ood_validation_{model}_n200_summary.csv"
        )
        frame.insert(0, "model", model)
        frames.append(frame)
    write(pd.concat(frames, ignore_index=True), "ood_transfer.csv")


def export_language_fluency() -> None:
    frame = read("ex20_language_benchmark/ex20_smoke_fix_results.csv")
    if set(frame["n_eval_tokens"].unique()) != {4096}:
        raise ValueError("paper PPL artifact must contain the verified 4,096-token run")
    write(
        frame.drop(columns=["vector_path", "local_text_path"]),
        "wikitext103_ppl_4096_tokens.csv",
    )


def export_robustness() -> None:
    write(
        read(
            "ex22_bootstrap_ci/"
            "ex22_bootstrap_ci_llama31_n200_v2_bootstrap_summary.csv"
        ),
        "paired_bootstrap.csv",
    )
    decoding = read(
        "ex30_decoding_seed_robustness/"
        "ex30_decoding_seed_robustness_l15_seed_level_summary.csv"
    )
    write(decoding.drop(columns=["summary_csv"]), "decoding_seed_robustness.csv")
    write(
        read(
            "ex34_exact_stratum_control/"
            "ex34_same_support_comparison_llama31_n200_v3_bootstrap_summary.csv"
        ),
        "label_composition_control.csv",
    )


def main() -> None:
    export_main_layer_sweep()
    export_cross_mechanism()
    export_geometry_and_residualization()
    export_cross_evaluator()
    export_persona()
    export_prompt_and_style_controls()
    export_transfer()
    export_language_fluency()
    export_robustness()
    print(f"Exported aggregate artifacts to {ARTIFACTS}")


if __name__ == "__main__":
    main()
