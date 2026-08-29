#!/usr/bin/env python3
"""Ex32: persona-shift mechanism-subspace and residual PCA analysis.

This is an analysis-only script. It reuses Ex17 persona-conditioned layer-15
activations and Ex11/12/13 mechanism vectors; it does not run generation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/persona_empathy_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "Llama": {
        "tag": "llama31",
        "hidden": 4096,
        "er": "outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_layer_vectors.npz",
        "ex": "outputs/ex12_filtered_ex/ex12_filtered_ex_llama31_n200_layer_vectors.npz",
        "ip": "outputs/ex13_filtered_ip/ex13_filtered_ip_llama31_n200_layer_vectors.npz",
    },
    "Qwen": {
        "tag": "qwen25",
        "hidden": 3584,
        "er": "outputs/ex11_filtered_er/ex11_filtered_er_qwen25_n200_layer_vectors.npz",
        "ex": "outputs/ex12_filtered_ex/ex12_filtered_ex_qwen25_n200_layer_vectors.npz",
        "ip": "outputs/ex13_filtered_ip/ex13_filtered_ip_qwen25_n200_layer_vectors.npz",
    },
    "Mistral": {
        "tag": "mistral7b",
        "hidden": 4096,
        "er": "outputs/ex11_filtered_er/ex11_filtered_er_mistral7b_n200_layer_vectors.npz",
        "ex": "outputs/ex12_filtered_ex/ex12_filtered_ex_mistral7b_n200_layer_vectors.npz",
        "ip": "outputs/ex13_filtered_ip/ex13_filtered_ip_mistral7b_n200_layer_vectors.npz",
    },
}
MECHANISMS = ["ER", "IP", "EX"]
MODEL_COLORS = {
    "Llama": "#2563eb",
    "Qwen": "#d97706",
    "Mistral": "#059669",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex32 persona-shift subspace analysis")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--ex17-dir", type=Path, default=Path("outputs/ex17_persona_effects_v2"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex32_persona_shift_subspace"))
    p.add_argument("--figure-dir", type=Path, default=Path("latex/figures"))
    p.add_argument("--models", nargs="+", default=["Llama", "Qwen", "Mistral"], choices=sorted(MODEL_SPECS))
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--baseline-persona", type=str, default="person")
    p.add_argument("--n-pcs", type=int, default=5)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def safe_norm(x: np.ndarray, axis=None) -> np.ndarray:
    return np.linalg.norm(x, axis=axis)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def model_prefix(model: str) -> str:
    return f"ex17_persona_effects_v2_{MODEL_SPECS[model]['tag']}_n200"


def load_layer_vector(path: Path, layer: int) -> np.ndarray:
    require_file(path)
    data = np.load(path)
    layers = data["layers"].astype(int)
    vectors = data["vectors"].astype(np.float32)
    idx = np.where(layers == int(layer))[0]
    if len(idx) != 1:
        raise ValueError(f"Layer {layer} not found exactly once in {path}: {layers.tolist()}")
    return vectors[int(idx[0])].astype(np.float32)


def orthonormal_basis(vectors: Iterable[np.ndarray]) -> np.ndarray:
    mat = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors], axis=1)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    tol = max(mat.shape) * np.finfo(np.float64).eps * float(s[0])
    rank = int((s > tol).sum())
    if rank == 0:
        raise ValueError("Mechanism vector matrix has zero rank")
    return u[:, :rank].astype(np.float32)


def load_model_inputs(root: Path, ex17_dir: Path, model: str, layer: int) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    prefix = model_prefix(model)
    act_rows_path = ex17_dir / f"{prefix}_activation_rows.csv"
    activations_path = ex17_dir / f"{prefix}_layer15_activations.npy"
    classified_path = ex17_dir / f"{prefix}_classified.csv"
    for path in [act_rows_path, activations_path, classified_path]:
        require_file(path)

    act_rows = pd.read_csv(act_rows_path)
    acts = np.load(activations_path, mmap_mode="r")
    classified = pd.read_csv(classified_path)
    vectors = {
        "ER": load_layer_vector(resolve_path(root, Path(MODEL_SPECS[model]["er"])), layer),
        "IP": load_layer_vector(resolve_path(root, Path(MODEL_SPECS[model]["ip"])), layer),
        "EX": load_layer_vector(resolve_path(root, Path(MODEL_SPECS[model]["ex"])), layer),
    }
    q_basis = orthonormal_basis([vectors["ER"], vectors["IP"], vectors["EX"]])
    return act_rows, acts, classified, vectors, q_basis


def build_paired_shift_rows(
    model: str,
    act_rows: pd.DataFrame,
    acts: np.ndarray,
    classified: pd.DataFrame,
    vectors: Dict[str, np.ndarray],
    q_basis: np.ndarray,
    baseline_persona: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    needed = {"activation_index", "used_persona", "eval_index", "source_id"}
    missing = sorted(needed - set(act_rows.columns))
    if missing:
        raise ValueError(f"activation rows missing columns: {missing}")

    class_needed = {"activation_index", "used_persona", "eval_index", "ER_label", "IP_label", "EX_label"}
    class_missing = sorted(class_needed - set(classified.columns))
    if class_missing:
        raise ValueError(f"classified rows missing columns: {class_missing}")

    meta = act_rows.merge(
        classified[
            [
                "activation_index",
                "ER_label",
                "IP_label",
                "EX_label",
                "response_token_count",
            ]
        ],
        on="activation_index",
        how="left",
        validate="one_to_one",
    )
    meta["used_persona"] = meta["used_persona"].astype(str)
    meta["eval_index"] = meta["eval_index"].astype(int)
    baseline = meta.loc[meta["used_persona"] == baseline_persona].copy()
    if baseline.empty:
        raise ValueError(f"Baseline persona '{baseline_persona}' not found for {model}")

    base_by_eval = baseline.set_index("eval_index")
    vector_units = {}
    for mech, vec in vectors.items():
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise ValueError(f"{model} {mech} vector has zero norm")
        vector_units[mech] = (vec / norm).astype(np.float32)

    records: List[dict] = []
    shift_vectors: List[np.ndarray] = []
    residual_vectors: List[np.ndarray] = []

    for row in meta.loc[meta["used_persona"] != baseline_persona].itertuples(index=False):
        eval_index = int(row.eval_index)
        if eval_index not in base_by_eval.index:
            continue
        base = base_by_eval.loc[eval_index]
        act_idx = int(row.activation_index)
        base_idx = int(base["activation_index"])
        d = np.asarray(acts[act_idx], dtype=np.float32) - np.asarray(acts[base_idx], dtype=np.float32)
        d64 = d.astype(np.float64, copy=False)
        q64 = q_basis.astype(np.float64, copy=False)
        d_parallel = q64 @ (q64.T @ d64)
        d_resid = d64 - d_parallel
        norm_sq = float(np.dot(d64, d64))
        parallel_norm_sq = float(np.dot(d_parallel, d_parallel))
        residual_norm_sq = float(np.dot(d_resid, d_resid))
        if norm_sq <= 0.0:
            r_mech = np.nan
            r_resid = np.nan
        else:
            r_mech = parallel_norm_sq / norm_sq
            r_resid = residual_norm_sq / norm_sq

        rec = {
            "model": model,
            "persona": str(row.used_persona),
            "eval_index": eval_index,
            "source_id": str(row.source_id),
            "activation_index": act_idx,
            "baseline_activation_index": base_idx,
            "shift_norm": float(np.sqrt(norm_sq)),
            "mechanism_projection_norm": float(np.sqrt(max(parallel_norm_sq, 0.0))),
            "residual_norm": float(np.sqrt(max(residual_norm_sq, 0.0))),
            "mechanism_fraction": float(r_mech),
            "residual_fraction": float(r_resid),
            "delta_ER": float(row.ER_label) - float(base["ER_label"]),
            "delta_IP": float(row.IP_label) - float(base["IP_label"]),
            "delta_EX": float(row.EX_label) - float(base["EX_label"]),
            "response_token_count": float(row.response_token_count),
            "baseline_response_token_count": float(base["response_token_count"]),
            "delta_response_token_count": float(row.response_token_count) - float(base["response_token_count"]),
        }
        for mech in MECHANISMS:
            unit = vector_units[mech].astype(np.float64, copy=False)
            rec[f"shift_dot_unit_{mech}"] = float(np.dot(d64, unit))
            rec[f"shift_cos_{mech}"] = cosine(d64, unit)
        records.append(rec)
        shift_vectors.append(d.astype(np.float32))
        residual_vectors.append(d_resid.astype(np.float32))

    if not records:
        raise RuntimeError(f"No paired persona shifts built for {model}")
    return pd.DataFrame(records), np.stack(shift_vectors, axis=0), np.stack(residual_vectors, axis=0)


def run_pca(x: np.ndarray, n_pcs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    n = x.shape[0]
    eig = (s**2) / max(n - 1, 1)
    total = float(eig.sum())
    explained = eig / total if total > 0 else np.zeros_like(eig)
    k = min(n_pcs, vt.shape[0])
    scores = centered @ vt[:k].T
    return vt[:k].astype(np.float32), explained[:k].astype(np.float64), scores.astype(np.float32), mean.astype(np.float32)


def summarize_persona_means(
    shift_df: pd.DataFrame,
    shift_vectors: np.ndarray,
    vectors: Dict[str, np.ndarray],
    q_basis: np.ndarray,
) -> pd.DataFrame:
    rows: List[dict] = []
    q64 = q_basis.astype(np.float64, copy=False)
    vector_units = {
        mech: (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float64)
        for mech, vec in vectors.items()
    }
    for persona, idx in shift_df.groupby("persona").groups.items():
        indices = np.asarray(list(idx), dtype=int)
        mean_shift = shift_vectors[indices].astype(np.float64).mean(axis=0)
        parallel = q64 @ (q64.T @ mean_shift)
        resid = mean_shift - parallel
        norm_sq = float(np.dot(mean_shift, mean_shift))
        par_sq = float(np.dot(parallel, parallel))
        resid_sq = float(np.dot(resid, resid))
        rec = {
            "model": str(shift_df.iloc[indices[0]]["model"]),
            "persona": persona,
            "n": int(len(indices)),
            "delta_ER_mean": float(shift_df.iloc[indices]["delta_ER"].mean()),
            "delta_IP_mean": float(shift_df.iloc[indices]["delta_IP"].mean()),
            "delta_EX_mean": float(shift_df.iloc[indices]["delta_EX"].mean()),
            "shift_norm": float(np.sqrt(norm_sq)),
            "mechanism_projection_norm": float(np.sqrt(max(par_sq, 0.0))),
            "residual_norm": float(np.sqrt(max(resid_sq, 0.0))),
            "mechanism_fraction": float(par_sq / norm_sq) if norm_sq > 0 else np.nan,
            "residual_fraction": float(resid_sq / norm_sq) if norm_sq > 0 else np.nan,
            "per_sample_mechanism_fraction_mean": float(shift_df.iloc[indices]["mechanism_fraction"].mean()),
            "per_sample_mechanism_fraction_median": float(shift_df.iloc[indices]["mechanism_fraction"].median()),
        }
        for mech in MECHANISMS:
            rec[f"mean_shift_dot_unit_{mech}"] = float(np.dot(mean_shift, vector_units[mech]))
            rec[f"mean_shift_cos_{mech}"] = cosine(mean_shift, vector_units[mech])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", "persona"]).reset_index(drop=True)


def add_pc_scores_to_shift_df(
    shift_df: pd.DataFrame,
    residual_vectors: np.ndarray,
    pcs: np.ndarray,
    pca_mean: np.ndarray,
) -> pd.DataFrame:
    centered = residual_vectors.astype(np.float64) - pca_mean.astype(np.float64)
    scores = centered @ pcs.astype(np.float64).T
    out = shift_df.copy()
    for j in range(scores.shape[1]):
        out[f"resid_pc{j + 1}"] = scores[:, j].astype(float)
    return out


def build_pca_summaries(
    model: str,
    pcs: np.ndarray,
    explained: np.ndarray,
    shift_df: pd.DataFrame,
    vectors: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pca_rows = []
    for j, var in enumerate(explained, start=1):
        pca_rows.append(
            {
                "model": model,
                "pc": j,
                "explained_variance_ratio": float(var),
                "cumulative_explained_variance_ratio": float(explained[:j].sum()),
            }
        )
    pca_df = pd.DataFrame(pca_rows)

    cosine_rows = []
    for j, pc in enumerate(pcs, start=1):
        row = {"model": model, "pc": j}
        for mech in MECHANISMS:
            row[f"cos_{mech}"] = cosine(pc, vectors[mech])
        cosine_rows.append(row)
    cosine_df = pd.DataFrame(cosine_rows)
    return pca_df, cosine_df


def summarize_behavioral_association(shift_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    feature_sets = {
        "shift_cos_ER": ["shift_cos_ER"],
        "shift_dot_unit_ER": ["shift_dot_unit_ER"],
        "resid_pc1": ["resid_pc1"],
        "resid_pc1_pc2": ["resid_pc1", "resid_pc2"],
    }
    outcomes = ["delta_ER", "delta_IP", "delta_EX"]
    for model, model_df in shift_df.groupby("model"):
        for outcome in outcomes:
            y = model_df[outcome].astype(float)
            for feature_name, features in feature_sets.items():
                if any(f not in model_df.columns for f in features):
                    continue
                valid = y.notna()
                for f in features:
                    valid &= model_df[f].notna()
                sub = model_df.loc[valid].copy()
                if sub.empty:
                    continue
                pred_cols = sub[features].astype(float)
                rows.append(
                    {
                        "model": model,
                        "level": "per_sample",
                        "outcome": outcome,
                        "feature_set": feature_name,
                        "n": int(len(sub)),
                        "pearson_first_feature": float(sub[features[0]].astype(float).corr(sub[outcome].astype(float))),
                        "spearman_first_feature": float(
                            sub[features[0]].astype(float).corr(sub[outcome].astype(float), method="spearman")
                        ),
                        "r2_in_sample": fit_r2(sub[outcome].astype(float), pred_cols),
                    }
                )

                loo = leave_one_persona_out(sub, outcome, features)
                rows.extend(loo)
    return pd.DataFrame(rows)


def fit_r2(y: pd.Series, x: pd.DataFrame) -> float:
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(xv)), xv])
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def leave_one_persona_out(df: pd.DataFrame, outcome: str, features: List[str]) -> List[dict]:
    rows: List[dict] = []
    personas = sorted(df["persona"].dropna().unique().tolist())
    preds_all: List[float] = []
    y_all: List[float] = []
    for persona in personas:
        train = df.loc[df["persona"] != persona]
        test = df.loc[df["persona"] == persona]
        if train.empty or test.empty:
            continue
        x_train = train[features].to_numpy(dtype=float)
        y_train = train[outcome].to_numpy(dtype=float)
        x_test = test[features].to_numpy(dtype=float)
        y_test = test[outcome].to_numpy(dtype=float)
        design_train = np.column_stack([np.ones(len(x_train)), x_train])
        design_test = np.column_stack([np.ones(len(x_test)), x_test])
        coef, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
        pred = design_test @ coef
        preds_all.extend(pred.tolist())
        y_all.extend(y_test.tolist())
    if not y_all:
        return rows
    y = np.asarray(y_all, dtype=float)
    pred = np.asarray(preds_all, dtype=float)
    mse = float(np.mean((y - pred) ** 2))
    mae = float(np.mean(np.abs(y - pred)))
    corr = float(pd.Series(pred).corr(pd.Series(y))) if len(np.unique(pred)) > 1 else np.nan
    rows.append(
        {
            "model": str(df["model"].iloc[0]),
            "level": "leave_one_persona_out",
            "outcome": outcome,
            "feature_set": "_".join(features),
            "n": int(len(y)),
            "pearson_first_feature": np.nan,
            "spearman_first_feature": np.nan,
            "r2_in_sample": np.nan,
            "mse": mse,
            "mae": mae,
            "pred_y_corr": corr,
        }
    )
    return rows


def summarize_model(shift_df: pd.DataFrame, persona_df: pd.DataFrame, pca_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for model, g in shift_df.groupby("model"):
        pg = persona_df.loc[persona_df["model"] == model]
        pca_g = pca_df.loc[pca_df["model"] == model]
        top2 = float(pca_g.loc[pca_g["pc"].isin([1, 2]), "explained_variance_ratio"].sum())
        top5 = float(pca_g.loc[pca_g["pc"].isin([1, 2, 3, 4, 5]), "explained_variance_ratio"].sum())
        rows.append(
            {
                "model": model,
                "n_shift_rows": int(len(g)),
                "n_personas": int(g["persona"].nunique()),
                "per_sample_mechanism_fraction_mean": float(g["mechanism_fraction"].mean()),
                "per_sample_mechanism_fraction_median": float(g["mechanism_fraction"].median()),
                "persona_mean_mechanism_fraction_mean": float(pg["mechanism_fraction"].mean()),
                "persona_mean_mechanism_fraction_median": float(pg["mechanism_fraction"].median()),
                "persona_mean_residual_fraction_mean": float(pg["residual_fraction"].mean()),
                "residual_pca_top2_variance": top2,
                "residual_pca_top5_variance": top5,
            }
        )
    return pd.DataFrame(rows)


def make_figures(persona_df: pd.DataFrame, shift_df: pd.DataFrame, model_summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    order = ["Llama", "Qwen", "Mistral"]
    summary = model_summary.set_index("model").reindex(order)
    axes[0].bar(
        np.arange(len(order)),
        summary["persona_mean_mechanism_fraction_mean"].astype(float),
        color=[MODEL_COLORS[m] for m in order],
        alpha=0.9,
    )
    axes[0].set_xticks(np.arange(len(order)), order, rotation=0)
    axes[0].set_ylabel("Mean fraction in ER/IP/EX subspace")
    axes[0].set_ylim(0, max(0.05, float(summary["persona_mean_mechanism_fraction_mean"].max()) * 1.25))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    for model in order:
        sub = persona_df.loc[persona_df["model"] == model]
        axes[1].scatter(
            sub["resid_pc1_centroid"],
            sub["resid_pc2_centroid"],
            label=model,
            color=MODEL_COLORS[model],
            s=28,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )
    axes[1].axhline(0, color="#9ca3af", linewidth=0.7, linestyle="--")
    axes[1].axvline(0, color="#9ca3af", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("Residual PC1 centroid")
    axes[1].set_ylabel("Residual PC2 centroid")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    out_base = figure_dir / "figure_persona_shift_subspace"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] figure: {out_base.with_suffix('.pdf')}", flush=True)
    print(f"[done] figure: {out_base.with_suffix('.png')}", flush=True)


def main() -> None:
    args = parse_args()
    root = args.persona_project.resolve()
    ex17_dir = resolve_path(root, args.ex17_dir).resolve()
    output_dir = resolve_path(root, args.output_dir).resolve()
    figure_dir = resolve_path(root, args.figure_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_shift_rows: List[pd.DataFrame] = []
    all_persona_rows: List[pd.DataFrame] = []
    all_pca_rows: List[pd.DataFrame] = []
    all_cosine_rows: List[pd.DataFrame] = []

    for model in args.models:
        print(f"[model] {model}", flush=True)
        act_rows, acts, classified, vectors, q_basis = load_model_inputs(root, ex17_dir, model, args.layer)
        shift_df, shift_vectors, residual_vectors = build_paired_shift_rows(
            model=model,
            act_rows=act_rows,
            acts=acts,
            classified=classified,
            vectors=vectors,
            q_basis=q_basis,
            baseline_persona=args.baseline_persona,
        )
        pcs, explained, scores, pca_mean = run_pca(residual_vectors, n_pcs=args.n_pcs)
        shift_df = add_pc_scores_to_shift_df(shift_df, residual_vectors, pcs, pca_mean)
        persona_df = summarize_persona_means(shift_df, shift_vectors, vectors, q_basis)
        for pc in range(1, args.n_pcs + 1):
            col = f"resid_pc{pc}"
            if col in shift_df.columns:
                centroids = shift_df.groupby("persona")[col].mean().rename(f"{col}_centroid")
                persona_df = persona_df.merge(centroids, on="persona", how="left")
        pca_df, cosine_df = build_pca_summaries(model, pcs, explained, shift_df, vectors)

        all_shift_rows.append(shift_df)
        all_persona_rows.append(persona_df)
        all_pca_rows.append(pca_df)
        all_cosine_rows.append(cosine_df)
        print(
            f"  shifts={len(shift_df)} personas={shift_df['persona'].nunique()} "
            f"mean_mech_fraction={shift_df['mechanism_fraction'].mean():.4f} "
            f"top2_resid_pca={explained[:2].sum():.4f}",
            flush=True,
        )

    shift_all = pd.concat(all_shift_rows, ignore_index=True)
    persona_all = pd.concat(all_persona_rows, ignore_index=True)
    pca_all = pd.concat(all_pca_rows, ignore_index=True)
    cosine_all = pd.concat(all_cosine_rows, ignore_index=True)
    model_summary = summarize_model(shift_all, persona_all, pca_all)
    association = summarize_behavioral_association(shift_all)

    paths = {
        "persona_shift_rows_csv": output_dir / "ex32_persona_shift_rows.csv",
        "persona_mean_decomposition_csv": output_dir / "ex32_persona_mean_decomposition.csv",
        "model_summary_csv": output_dir / "ex32_model_summary.csv",
        "residual_pca_summary_csv": output_dir / "ex32_residual_pca_summary.csv",
        "pc_mechanism_cosines_csv": output_dir / "ex32_pc_mechanism_cosines.csv",
        "behavioral_association_csv": output_dir / "ex32_behavioral_association.csv",
        "metadata_json": output_dir / "ex32_metadata.json",
    }
    shift_all.to_csv(paths["persona_shift_rows_csv"], index=False)
    persona_all.to_csv(paths["persona_mean_decomposition_csv"], index=False)
    model_summary.to_csv(paths["model_summary_csv"], index=False)
    pca_all.to_csv(paths["residual_pca_summary_csv"], index=False)
    cosine_all.to_csv(paths["pc_mechanism_cosines_csv"], index=False)
    association.to_csv(paths["behavioral_association_csv"], index=False)

    metadata = {
        "task": "ex32_persona_shift_subspace",
        "models": args.models,
        "layer": int(args.layer),
        "baseline_persona": args.baseline_persona,
        "n_pcs": int(args.n_pcs),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["metadata_json"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_figures:
        make_figures(persona_all, shift_all, model_summary, figure_dir)

    for name, path in paths.items():
        print(f"[output] {name}: {path}", flush=True)
    print("[summary]")
    print(model_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
