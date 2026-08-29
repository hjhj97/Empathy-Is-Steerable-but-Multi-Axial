#!/usr/bin/env python3
"""Analyze Ex34 exact-stratum controlled extraction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = ("ER", "EX", "IP")
LABEL_COLUMNS = ["ER_label", "EX_label", "IP_label"]
ORIGINAL = {
    "ER": ("outputs/ex11_filtered_er", "ex11_filtered_er_llama31_n200"),
    "EX": ("outputs/ex12_filtered_ex", "ex12_filtered_ex_llama31_n200"),
    "IP": ("outputs/ex13_filtered_ip", "ex13_filtered_ip_llama31_n200"),
}
CONTROLLED = {
    "ER": (
        "outputs/rebuttal_priority1_controlled_er",
        "rebuttal_p1_filtered_er_ge1_exact_l15_a-1_0_1_n200_v2",
    ),
    "EX": (
        "outputs/rebuttal_priority1_controlled_ex",
        "rebuttal_p1_filtered_ex_ge1_exact_l15_a-1_0_1_n200_v2",
    ),
    "IP": (
        "outputs/rebuttal_priority1_controlled_ip",
        "rebuttal_p1_filtered_ip_ge1_exact_l15_a-1_0_1_n200_v2",
    ),
}
SAME_SUPPORT_UNMATCHED = {
    "ER": (
        "outputs/rebuttal_priority1_unmatched_er",
        "rebuttal_p1_filtered_er_ge1_unmatched_l15_a-1_0_1_n200_v3",
    ),
    "EX": (
        "outputs/rebuttal_priority1_unmatched_ex",
        "rebuttal_p1_filtered_ex_ge1_unmatched_l15_a-1_0_1_n200_v3",
    ),
    "IP": (
        "outputs/rebuttal_priority1_unmatched_ip",
        "rebuttal_p1_filtered_ip_ge1_unmatched_l15_a-1_0_1_n200_v3",
    ),
}
NORM_MATCHED = {
    "ER": (
        "outputs/rebuttal_priority1_controlled_er_normmatched",
        "rebuttal_p1_filtered_er_ge1_exact_l15_normmatched_n200_v3",
    ),
    "IP": (
        "outputs/rebuttal_priority1_controlled_ip_normmatched",
        "rebuttal_p1_filtered_ip_ge1_exact_l15_normmatched_n200_v3",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze Ex34 exact-stratum control")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparison-arm",
        choices=["same_support", "historical"],
        default="same_support",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ex34_exact_stratum_control"),
    )
    parser.add_argument("--run-name", default="ex34_same_support_comparison_llama31_n200_v3")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def artifact_path(spec: Tuple[str, str], suffix: str) -> Path:
    directory, run_name = spec
    return resolve(directory) / f"{run_name}_{suffix}"


def load_deltas(path: Path, layer: int, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"source_id": str})
    frame = frame[
        (frame["layer"] == layer)
        & (np.isclose(frame["alpha"], 0.0) | np.isclose(frame["alpha"], alpha))
    ].copy()
    if frame.duplicated(["source_id", "alpha"]).any():
        raise ValueError(f"Expected one response per source_id/alpha in {path}")
    counts = frame.groupby("alpha")["source_id"].nunique().to_dict()
    if counts.get(0.0) != 200 or counts.get(alpha) != 200:
        raise ValueError(f"Expected 200 seekers for alpha 0 and {alpha} in {path}; got {counts}")

    baseline = frame[np.isclose(frame["alpha"], 0.0)].set_index("source_id").sort_index()
    steered = frame[np.isclose(frame["alpha"], alpha)].set_index("source_id").sort_index()
    if not baseline.index.equals(steered.index):
        raise ValueError(f"Seeker mismatch between alpha 0 and {alpha} in {path}")
    deltas = steered[LABEL_COLUMNS].astype(float) - baseline[LABEL_COLUMNS].astype(float)
    deltas.columns = MECHANISMS
    return deltas, baseline


def metrics_from_mean(mean_delta: np.ndarray, target_index: int) -> Dict[str, float]:
    off_indices = [idx for idx in range(3) if idx != target_index]
    target_delta = float(mean_delta[target_index])
    off_target_burden = float(np.abs(mean_delta[off_indices]).sum())
    selectivity = float(abs(target_delta) / (off_target_burden + 1e-12))
    return {
        "delta_ER": float(mean_delta[0]),
        "delta_EX": float(mean_delta[1]),
        "delta_IP": float(mean_delta[2]),
        "target_delta": target_delta,
        "off_target_burden": off_target_burden,
        "selectivity": selectivity,
    }


def bootstrap_metrics(
    values: np.ndarray,
    target_index: int,
    sample_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    boot_means = values[sample_indices].mean(axis=1)
    off_indices = [idx for idx in range(3) if idx != target_index]
    target = boot_means[:, target_index]
    burden = np.abs(boot_means[:, off_indices]).sum(axis=1)
    return {
        "delta_ER": boot_means[:, 0],
        "delta_EX": boot_means[:, 1],
        "delta_IP": boot_means[:, 2],
        "target_delta": target,
        "off_target_burden": burden,
        "selectivity": np.abs(target) / (burden + 1e-12),
    }


def percentile_interval(values: np.ndarray) -> Tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def load_vector(path: Path, layer: int) -> np.ndarray:
    artifact = np.load(path)
    layers = artifact["layers"].astype(int)
    matches = np.where(layers == layer)[0]
    if len(matches) != 1:
        raise ValueError(f"Layer {layer} not found exactly once in {path}: {layers.tolist()}")
    return artifact["vectors"][int(matches[0])].astype(np.float64)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    estimate_rows = []
    bootstrap_rows = []
    geometry_rows = []
    pairwise_geometry_rows = []
    baseline_rows = []
    norm_matched_rows = []
    controlled_vectors: Dict[str, np.ndarray] = {}
    comparison_vectors: Dict[str, np.ndarray] = {}
    comparison_specs = SAME_SUPPORT_UNMATCHED if args.comparison_arm == "same_support" else ORIGINAL
    comparison_name = "same_support_unmatched" if args.comparison_arm == "same_support" else "historical"

    for mechanism in MECHANISMS:
        original_path = artifact_path(comparison_specs[mechanism], "classified.csv")
        controlled_path = artifact_path(CONTROLLED[mechanism], "classified.csv")
        original_delta, original_baseline = load_deltas(original_path, args.layer, args.alpha)
        controlled_delta, controlled_baseline = load_deltas(controlled_path, args.layer, args.alpha)
        if not original_delta.index.equals(controlled_delta.index):
            raise ValueError(f"Controlled/uncontrolled seeker mismatch for {mechanism}")

        target_index = MECHANISMS.index(mechanism)
        sample_indices = rng.integers(
            0,
            len(original_delta),
            size=(args.bootstrap_iters, len(original_delta)),
        )
        condition_values = {
            comparison_name: original_delta.to_numpy(dtype=float),
            "controlled": controlled_delta.to_numpy(dtype=float),
        }
        condition_boot = {}
        for condition, values in condition_values.items():
            estimates = metrics_from_mean(values.mean(axis=0), target_index)
            boot = bootstrap_metrics(values, target_index, sample_indices)
            condition_boot[condition] = boot
            estimate_rows.append({"mechanism": mechanism, "condition": condition, **estimates})
            for metric, samples in boot.items():
                low, high = percentile_interval(samples)
                bootstrap_rows.append({
                    "mechanism": mechanism,
                    "contrast": condition,
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_iters": args.bootstrap_iters,
                })

        original_est = metrics_from_mean(condition_values[comparison_name].mean(axis=0), target_index)
        controlled_est = metrics_from_mean(condition_values["controlled"].mean(axis=0), target_index)
        for metric in original_est:
            samples = condition_boot["controlled"][metric] - condition_boot[comparison_name][metric]
            low, high = percentile_interval(samples)
            bootstrap_rows.append({
                "mechanism": mechanism,
                "contrast": f"controlled_minus_{comparison_name}",
                "metric": metric,
                "estimate": controlled_est[metric] - original_est[metric],
                "ci_low": low,
                "ci_high": high,
                "bootstrap_iters": args.bootstrap_iters,
            })

        original_vector = load_vector(
            artifact_path(comparison_specs[mechanism], "layer_vectors.npz"),
            args.layer,
        )
        controlled_vector = load_vector(artifact_path(CONTROLLED[mechanism], "layer_vectors.npz"), args.layer)
        comparison_vectors[mechanism] = original_vector
        controlled_vectors[mechanism] = controlled_vector
        original_norm = float(np.linalg.norm(original_vector))
        controlled_norm = float(np.linalg.norm(controlled_vector))
        norm_ratio = controlled_norm / original_norm
        cosine = float(
            np.dot(original_vector, controlled_vector) / (original_norm * controlled_norm)
        )
        geometry_rows.append({
            "mechanism": mechanism,
            "comparison_arm": comparison_name,
            "layer": args.layer,
            "uncontrolled_norm": original_norm,
            "controlled_norm": controlled_norm,
            "norm_ratio": norm_ratio,
            "cosine_controlled_vs_uncontrolled": cosine,
            "norm_matched_required": bool(abs(norm_ratio - 1.0) > 0.10),
            "alpha_norm_matched": float(original_norm / controlled_norm),
        })

        common = original_baseline.index.intersection(controlled_baseline.index)
        label_equal = (
            original_baseline.loc[common, LABEL_COLUMNS].to_numpy()
            == controlled_baseline.loc[common, LABEL_COLUMNS].to_numpy()
        ).all(axis=1)
        response_equal = (
            original_baseline.loc[common, "generated_response"].astype(str).to_numpy()
            == controlled_baseline.loc[common, "generated_response"].astype(str).to_numpy()
        )
        baseline_rows.append({
            "mechanism": mechanism,
            "comparison_arm": comparison_name,
            "n": len(common),
            "label_match_rate": float(label_equal.mean()),
            "response_exact_match_rate": float(response_equal.mean()),
        })

    for arm_name, vectors in (
        (comparison_name, comparison_vectors),
        ("controlled", controlled_vectors),
    ):
        for first, second in (("ER", "EX"), ("ER", "IP"), ("EX", "IP")):
            first_vector = vectors[first]
            second_vector = vectors[second]
            cosine = float(
                np.dot(first_vector, second_vector)
                / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
            )
            pairwise_geometry_rows.append({
                "arm": arm_name,
                "pair": f"{first}-{second}",
                "layer": args.layer,
                "cosine": cosine,
            })

    pairwise_frame = pd.DataFrame(pairwise_geometry_rows)
    comparison_cosines = pairwise_frame[pairwise_frame["arm"] == comparison_name].set_index("pair")["cosine"]
    controlled_cosines = pairwise_frame[pairwise_frame["arm"] == "controlled"].set_index("pair")["cosine"]
    for pair in comparison_cosines.index:
        pairwise_geometry_rows.append({
            "arm": f"controlled_minus_{comparison_name}",
            "pair": pair,
            "layer": args.layer,
            "cosine": float(controlled_cosines[pair] - comparison_cosines[pair]),
        })

    for norm_mechanism, norm_spec in NORM_MATCHED.items():
        norm_matched_path = artifact_path(norm_spec, "classified.csv")
        if not norm_matched_path.exists():
            continue
        norm_frame = pd.read_csv(norm_matched_path)
        nonzero_alphas = sorted(
            float(value) for value in norm_frame["alpha"].unique() if not np.isclose(value, 0.0)
        )
        if len(nonzero_alphas) != 1:
            raise ValueError(f"Expected one nonzero norm-matched alpha in {norm_matched_path}")
        norm_alpha = nonzero_alphas[0]
        norm_delta, _ = load_deltas(norm_matched_path, args.layer, norm_alpha)
        sample_indices = rng.integers(
            0,
            len(norm_delta),
            size=(args.bootstrap_iters, len(norm_delta)),
        )
        target_index = MECHANISMS.index(norm_mechanism)
        estimates = metrics_from_mean(
            norm_delta.to_numpy(dtype=float).mean(axis=0),
            target_index,
        )
        boot = bootstrap_metrics(norm_delta.to_numpy(dtype=float), target_index, sample_indices)
        for metric, estimate in estimates.items():
            low, high = percentile_interval(boot[metric])
            norm_matched_rows.append({
                "mechanism": norm_mechanism,
                "alpha": norm_alpha,
                "metric": metric,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_iters": args.bootstrap_iters,
            })

    estimates_path = output_dir / f"{args.run_name}_estimates.csv"
    bootstrap_path = output_dir / f"{args.run_name}_bootstrap_summary.csv"
    geometry_path = output_dir / f"{args.run_name}_vector_geometry.csv"
    pairwise_geometry_path = output_dir / f"{args.run_name}_pairwise_vector_geometry.csv"
    baseline_path = output_dir / f"{args.run_name}_alpha0_reproducibility.csv"
    norm_matched_path_out = output_dir / f"{args.run_name}_norm_matched_summary.csv"
    metadata_path = output_dir / f"{args.run_name}_metadata.json"
    pd.DataFrame(estimate_rows).to_csv(estimates_path, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)
    pd.DataFrame(pairwise_geometry_rows).to_csv(pairwise_geometry_path, index=False)
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    pd.DataFrame(norm_matched_rows).to_csv(norm_matched_path_out, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "task": "ex34_exact_stratum_control",
                "layer": args.layer,
                "alpha": args.alpha,
                "bootstrap_iters": args.bootstrap_iters,
                "seed": args.seed,
                "comparison_arm": args.comparison_arm,
                "mechanisms": list(MECHANISMS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] estimates={estimates_path}")
    print(f"[done] bootstrap={bootstrap_path}")
    print(f"[done] geometry={geometry_path}")
    print(f"[done] pairwise_geometry={pairwise_geometry_path}")
    print(f"[done] alpha0={baseline_path}")
    print(f"[done] norm_matched={norm_matched_path_out}")
    print(f"[done] metadata={metadata_path}")


if __name__ == "__main__":
    main()
