#!/usr/bin/env python3
"""Compare ER/EX/IP steering-vector similarity across Ex11, Ex12, and Ex13.

This script loads the layer-wise steering vectors saved by:
  - Ex11: filtered ER steering
  - Ex12: filtered EX steering
  - Ex13: filtered IP steering

For each model and each overlapping layer, it computes pairwise cosine
similarity among the three vectors:
  - ER vs EX
  - ER vs IP
  - EX vs IP

Outputs:
  - per-layer long table
  - per-layer wide table
  - per-model/pair summary table
  - a small text report printed to stdout
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str


EXPERIMENTS: Sequence[ExperimentSpec] = (
    ExperimentSpec("ex11_filtered_er", "ER"),
    ExperimentSpec("ex12_filtered_ex", "EX"),
    ExperimentSpec("ex13_filtered_ip", "IP"),
)

DEFAULT_MODELS: Sequence[str] = ("llama31", "qwen25", "mistral7b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare Ex11/Ex12/Ex13 steering-vector similarity")
    p.add_argument("--persona-project", type=Path, default=ROOT)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model tags to compare (default: llama31 qwen25 mistral7b).",
    )
    p.add_argument(
        "--experiments",
        nargs="+",
        default=[spec.name for spec in EXPERIMENTS],
        help="Experiment directories to scan (default: ex11_filtered_er ex12_filtered_ex ex13_filtered_ip).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vector_similarity_ex11_ex12_ex13"),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def find_latest_npz(exp_dir: Path, exp_name: str, model_tag: str) -> Path:
    pattern = f"{exp_name}_{model_tag}_*_layer_vectors.npz"
    matches = sorted(exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No vector file found for {exp_name}/{model_tag}: {exp_dir / pattern}")
    return matches[0]


def load_layer_vectors(npz_path: Path) -> Tuple[List[int], Dict[int, np.ndarray]]:
    data = np.load(npz_path)
    if "layers" not in data or "vectors" not in data:
        raise ValueError(f"Invalid npz format: {npz_path}")

    layers = [int(x) for x in np.asarray(data["layers"]).tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors array in {npz_path}, got {vectors.shape}")
    if len(layers) != vectors.shape[0]:
        raise ValueError(
            f"Layer count mismatch in {npz_path}: layers={len(layers)} vectors={vectors.shape[0]}"
        )

    out: Dict[int, np.ndarray] = {}
    for layer, vec in zip(layers, vectors):
        out[int(layer)] = np.asarray(vec, dtype=np.float32)
    return sorted(out.keys()), out


def pair_name(a: str, b: str) -> str:
    return f"{a}_vs_{b}"


def build_tables(
    model_tag: str,
    exp_vectors: Dict[str, Dict[int, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = {spec.name: spec.label for spec in EXPERIMENTS}
    available = [exp for exp in labels if exp in exp_vectors]
    if len(available) < 3:
        raise ValueError(f"Need all three experiments for comparison, got: {available}")

    common_layers = sorted(set.intersection(*(set(exp_vectors[exp].keys()) for exp in available)))
    if not common_layers:
        raise ValueError(f"No overlapping layers found for model={model_tag}")

    pairs = [("ER", "EX"), ("ER", "IP"), ("EX", "IP")]

    long_rows: List[dict] = []
    wide_rows: List[dict] = []

    for layer in common_layers:
        layer_vecs = {labels[exp]: exp_vectors[exp][layer] for exp in available}
        wide_row = {"model": model_tag, "layer": int(layer)}
        for a, b in pairs:
            cos = cosine(layer_vecs[a], layer_vecs[b])
            long_rows.append(
                {
                    "model": model_tag,
                    "layer": int(layer),
                    "pair": pair_name(a, b),
                    "vector_a": a,
                    "vector_b": b,
                    "cosine": cos,
                    "norm_a": float(np.linalg.norm(layer_vecs[a])),
                    "norm_b": float(np.linalg.norm(layer_vecs[b])),
                }
            )
            wide_row[f"cos_{a.lower()}_{b.lower()}"] = cos
        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows).sort_values(["model", "pair", "layer"]).reset_index(drop=True)
    wide_df = pd.DataFrame(wide_rows).sort_values(["model", "layer"]).reset_index(drop=True)

    summary_df = (
        long_df.groupby(["model", "pair"], as_index=False)
        .agg(
            n_layers=("layer", "count"),
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            cosine_min=("cosine", "min"),
            cosine_max=("cosine", "max"),
        )
        .sort_values(["model", "pair"])
        .reset_index(drop=True)
    )
    summary_df["cosine_std"] = summary_df["cosine_std"].fillna(0.0)

    return long_df, wide_df, summary_df


def main() -> None:
    args = parse_args()
    project_root = args.persona_project.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_lookup = {spec.name: spec for spec in EXPERIMENTS}
    selected_experiments = []
    for exp_name in args.experiments:
        if exp_name not in exp_lookup:
            raise ValueError(f"Unsupported experiment: {exp_name}")
        selected_experiments.append(exp_lookup[exp_name])

    long_frames: List[pd.DataFrame] = []
    wide_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    report_lines: List[str] = []

    for model_tag in args.models:
        exp_vectors: Dict[str, Dict[int, np.ndarray]] = {}
        loaded_paths: Dict[str, Path] = {}
        for spec in selected_experiments:
            exp_dir = project_root / "outputs" / spec.name
            npz_path = find_latest_npz(exp_dir, spec.name, model_tag)
            _, vectors = load_layer_vectors(npz_path)
            exp_vectors[spec.name] = vectors
            loaded_paths[spec.name] = npz_path

        long_df, wide_df, summary_df = build_tables(model_tag, exp_vectors)
        long_frames.append(long_df)
        wide_frames.append(wide_df)
        summary_frames.append(summary_df)

        report_lines.append(f"Model: {model_tag}")
        for spec in selected_experiments:
            report_lines.append(f"  {spec.label}: {loaded_paths[spec.name]}")
        report_lines.append("  Pairwise mean cosine:")
        for row in summary_df.itertuples(index=False):
            report_lines.append(
                f"    {row.pair}: mean={row.cosine_mean:+.4f}, std={row.cosine_std:.4f}, "
                f"min={row.cosine_min:+.4f}, max={row.cosine_max:+.4f}, n={int(row.n_layers)}"
            )
        report_lines.append("")

    all_long = pd.concat(long_frames, ignore_index=True)
    all_wide = pd.concat(wide_frames, ignore_index=True)
    all_summary = pd.concat(summary_frames, ignore_index=True)

    out_long = output_dir / "er_ex_ip_vector_similarity_per_layer.csv"
    out_wide = output_dir / "er_ex_ip_vector_similarity_wide.csv"
    out_summary = output_dir / "er_ex_ip_vector_similarity_summary.csv"
    out_report = output_dir / "er_ex_ip_vector_similarity_report.txt"

    if not args.overwrite:
        for p in [out_long, out_wide, out_summary, out_report]:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")

    all_long.to_csv(out_long, index=False)
    all_wide.to_csv(out_wide, index=False)
    all_summary.to_csv(out_summary, index=False)
    out_report.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    print("\n".join(report_lines).rstrip())
    print(f"\n[done] long table: {out_long}")
    print(f"[done] wide table: {out_wide}")
    print(f"[done] summary: {out_summary}")
    print(f"[done] report: {out_report}")


if __name__ == "__main__":
    main()
