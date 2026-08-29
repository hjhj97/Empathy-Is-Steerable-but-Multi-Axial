#!/usr/bin/env python3
"""Ex22: bootstrap resampling for core steering/persona deltas.

Scope (fast path):
1) Steering significance on layer-15 only:
   - Ex11 ER: alpha(+/-) vs alpha=0
   - Ex12 EX: alpha(+/-) vs alpha=0
   - Ex13 IP: alpha(+/-) vs alpha=0
2) Persona gap significance:
   - Ex17 target personas vs baseline persona (default: black/white vs person)

This script reuses existing classified CSV outputs. No new generation is performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/ex22_bootstrap_ci")

DEFAULT_EX11_CLASSIFIED = Path("outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_classified.csv")
DEFAULT_EX12_CLASSIFIED = Path("outputs/ex12_filtered_ex/ex12_filtered_ex_llama31_n200_classified.csv")
DEFAULT_EX13_CLASSIFIED = Path("outputs/ex13_filtered_ip/ex13_filtered_ip_llama31_n200_classified.csv")
DEFAULT_EX17_CLASSIFIED = Path("outputs/ex17_persona_effects_v2/ex17_persona_effects_v2_llama31_n200_classified.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex22 bootstrap CI for steering/persona deltas")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)

    p.add_argument("--ex11-classified", type=Path, default=DEFAULT_EX11_CLASSIFIED)
    p.add_argument("--ex12-classified", type=Path, default=DEFAULT_EX12_CLASSIFIED)
    p.add_argument("--ex13-classified", type=Path, default=DEFAULT_EX13_CLASSIFIED)
    p.add_argument("--ex17-classified", type=Path, default=DEFAULT_EX17_CLASSIFIED)

    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--alphas", type=str, default="-1,1", help="Comma-separated steered alphas.")
    p.add_argument("--baseline-alpha", type=float, default=0.0)

    p.add_argument("--persona-baseline", type=str, default="person")
    p.add_argument("--persona-targets", type=str, default="black person,white person")
    p.add_argument("--persona-metrics", type=str, default="ER_label")

    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, default="ex22_bootstrap_ci_llama31_n200")
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in (raw or "").split(","):
        val = tok.strip()
        if not val:
            continue
        low = val.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(val)
    return out


def parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for tok in (raw or "").split(","):
        s = tok.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("No values parsed from list.")
    return out


def normalize_alpha(v) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return float("nan")


def format_alpha_tag(alpha: float) -> str:
    if float(alpha).is_integer():
        return f"{int(alpha):+d}"
    return f"{alpha:+g}"


def pick_present_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def choose_pairing_keys(df: pd.DataFrame, priority: Sequence[str]) -> List[str]:
    """Choose stable pairing keys by priority, stopping before volatile ids.

    Example: if both eval_index and source_id exist, use both.
    If neither exists, fall back to first available key from priority.
    """
    present = pick_present_cols(df, priority)
    if not present:
        return []

    # Prefer stable seeker-level ids only.
    stable = [c for c in present if c in {"eval_index", "source_id", "base_sample_index"}]
    if stable:
        return stable
    return [present[0]]


def make_unit_id(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.Series:
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return pd.Series([str(i) for i in range(len(df))], index=df.index)
    parts = [df[c].astype(str) for c in key_cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "::" + p
    return out


def build_steering_paired_deltas(
    df: pd.DataFrame,
    metric_col: str,
    layer: int,
    baseline_alpha: float,
    steered_alpha: float,
    experiment: str,
) -> pd.DataFrame:
    if metric_col not in df.columns:
        raise ValueError(f"Metric column not found: {metric_col}")
    if "layer" not in df.columns or "alpha" not in df.columns:
        raise ValueError("Input must contain 'layer' and 'alpha' columns.")

    work = df.copy()
    work["alpha_norm"] = work["alpha"].map(normalize_alpha)
    work["layer_norm"] = pd.to_numeric(work["layer"], errors="coerce")
    work = work[work["layer_norm"] == int(layer)].copy()
    work = work.dropna(subset=["alpha_norm"])

    key_cols = choose_pairing_keys(work, ["base_sample_index", "source_id", "response_rank"])
    if not key_cols:
        raise ValueError(
            "Could not infer pairing keys for steering. Need one of: base_sample_index, source_id, response_rank."
        )

    base = work[np.isclose(work["alpha_norm"], float(baseline_alpha))].copy()
    steer = work[np.isclose(work["alpha_norm"], float(steered_alpha))].copy()
    if base.empty or steer.empty:
        return pd.DataFrame()

    base = (
        base.groupby(key_cols, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "value_base"})
    )
    steer = (
        steer.groupby(key_cols, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "value_steer"})
    )

    merged = steer.merge(base, on=key_cols, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["delta"] = merged["value_steer"] - merged["value_base"]
    merged["unit_id"] = make_unit_id(merged, key_cols)
    merged["analysis_type"] = "steering"
    merged["experiment"] = experiment
    merged["metric"] = metric_col
    merged["layer"] = int(layer)
    merged["condition_a"] = f"alpha={steered_alpha:g}"
    merged["condition_b"] = f"alpha={baseline_alpha:g}"
    merged["analysis_id"] = (
        f"{experiment}_{metric_col}_layer{layer}_alpha{format_alpha_tag(steered_alpha)}_vs_{format_alpha_tag(baseline_alpha)}"
    )
    return merged


def build_persona_paired_deltas(
    df: pd.DataFrame,
    metric_col: str,
    baseline_persona: str,
    target_persona: str,
    experiment: str = "ex17",
) -> pd.DataFrame:
    required = {"used_persona", metric_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Persona input missing columns: {missing}")

    work = df.copy()
    work["used_persona"] = work["used_persona"].fillna("").astype(str)

    key_cols = choose_pairing_keys(work, ["eval_index", "source_id", "activation_index"])
    if not key_cols:
        raise ValueError("Could not infer pairing keys for persona. Need one of: eval_index, source_id, activation_index.")

    base = work[work["used_persona"] == baseline_persona].copy()
    tgt = work[work["used_persona"] == target_persona].copy()
    if base.empty or tgt.empty:
        return pd.DataFrame()

    base = (
        base.groupby(key_cols, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "value_base"})
    )
    tgt = (
        tgt.groupby(key_cols, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "value_steer"})
    )

    merged = tgt.merge(base, on=key_cols, how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["delta"] = merged["value_steer"] - merged["value_base"]
    merged["unit_id"] = make_unit_id(merged, key_cols)
    merged["analysis_type"] = "persona"
    merged["experiment"] = experiment
    merged["metric"] = metric_col
    merged["layer"] = np.nan
    merged["condition_a"] = f"persona={target_persona}"
    merged["condition_b"] = f"persona={baseline_persona}"
    merged["analysis_id"] = f"{experiment}_{metric_col}_{target_persona.replace(' ', '_')}_vs_{baseline_persona.replace(' ', '_')}"
    return merged


def bootstrap_mean_delta(
    deltas: np.ndarray,
    iters: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    if deltas.ndim != 1:
        deltas = deltas.reshape(-1)
    deltas = deltas.astype(np.float64)
    n = int(deltas.shape[0])
    if n <= 0:
        raise ValueError("No deltas to bootstrap.")

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, n, size=(iters, n))
    boot_means = deltas[sample_idx].mean(axis=1)

    observed = float(deltas.mean())
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    p_lo = float(np.mean(boot_means <= 0.0))
    p_hi = float(np.mean(boot_means >= 0.0))
    p_two = min(1.0, 2.0 * min(p_lo, p_hi))

    stats = {
        "n_units": n,
        "observed_delta_mean": observed,
        "bootstrap_mean": float(boot_means.mean()),
        "bootstrap_std": float(boot_means.std(ddof=1)) if iters > 1 else 0.0,
        "ci_low_95": float(ci_low),
        "ci_high_95": float(ci_high),
        "p_two_sided_vs0": float(p_two),
        "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
    }
    return stats, boot_means


def save_metadata(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    persona_project = args.persona_project.resolve()
    out_dir = resolve_path(persona_project, args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ex11_path = resolve_path(persona_project, args.ex11_classified).resolve()
    ex12_path = resolve_path(persona_project, args.ex12_classified).resolve()
    ex13_path = resolve_path(persona_project, args.ex13_classified).resolve()
    ex17_path = resolve_path(persona_project, args.ex17_classified).resolve()

    for p in [ex11_path, ex12_path, ex13_path, ex17_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    alphas = parse_float_list(args.alphas)
    persona_targets = parse_csv_list(args.persona_targets)
    persona_metrics = parse_csv_list(args.persona_metrics)
    if not persona_targets:
        raise ValueError("No persona targets provided.")
    if not persona_metrics:
        raise ValueError("No persona metrics provided.")
    if int(args.bootstrap_iters) <= 0:
        raise ValueError("--bootstrap-iters must be positive.")

    df11 = pd.read_csv(ex11_path)
    df12 = pd.read_csv(ex12_path)
    df13 = pd.read_csv(ex13_path)
    df17 = pd.read_csv(ex17_path)

    all_pairs: List[pd.DataFrame] = []
    for a in alphas:
        all_pairs.append(
            build_steering_paired_deltas(
                df=df11,
                metric_col="ER_label",
                layer=int(args.layer),
                baseline_alpha=float(args.baseline_alpha),
                steered_alpha=float(a),
                experiment="ex11",
            )
        )
        all_pairs.append(
            build_steering_paired_deltas(
                df=df12,
                metric_col="EX_label",
                layer=int(args.layer),
                baseline_alpha=float(args.baseline_alpha),
                steered_alpha=float(a),
                experiment="ex12",
            )
        )
        all_pairs.append(
            build_steering_paired_deltas(
                df=df13,
                metric_col="IP_label",
                layer=int(args.layer),
                baseline_alpha=float(args.baseline_alpha),
                steered_alpha=float(a),
                experiment="ex13",
            )
        )

    for metric in persona_metrics:
        for persona in persona_targets:
            all_pairs.append(
                build_persona_paired_deltas(
                    df=df17,
                    metric_col=metric,
                    baseline_persona=args.persona_baseline,
                    target_persona=persona,
                    experiment="ex17",
                )
            )

    all_pairs = [d for d in all_pairs if d is not None and not d.empty]
    if not all_pairs:
        raise RuntimeError("No paired deltas produced. Check filters and input files.")

    paired_df = pd.concat(all_pairs, ignore_index=True)

    summary_rows: List[Dict] = []
    sample_rows: List[Dict] = []

    # Different seed per analysis for deterministic but independent bootstrap draws.
    base_seed = int(args.seed)
    for idx, (analysis_id, g) in enumerate(paired_df.groupby("analysis_id", sort=True)):
        deltas = g["delta"].to_numpy(dtype=np.float64)
        run_seed = base_seed + idx
        stats, boot_means = bootstrap_mean_delta(
            deltas=deltas,
            iters=int(args.bootstrap_iters),
            seed=run_seed,
        )

        row = {
            "analysis_id": analysis_id,
            "analysis_type": str(g["analysis_type"].iloc[0]),
            "experiment": str(g["experiment"].iloc[0]),
            "metric": str(g["metric"].iloc[0]),
            "layer": g["layer"].iloc[0],
            "condition_a": str(g["condition_a"].iloc[0]),
            "condition_b": str(g["condition_b"].iloc[0]),
            "mean_a": float(g["value_steer"].mean()),
            "mean_b": float(g["value_base"].mean()),
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_seed": run_seed,
        }
        row.update(stats)
        summary_rows.append(row)

        sample_rows.extend(
            {
                "analysis_id": analysis_id,
                "bootstrap_iter": int(i),
                "mean_delta": float(v),
            }
            for i, v in enumerate(boot_means)
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["analysis_type", "experiment", "metric", "analysis_id"])
    samples_df = pd.DataFrame(sample_rows)
    paired_keep = [
        "analysis_id",
        "analysis_type",
        "experiment",
        "metric",
        "condition_a",
        "condition_b",
        "unit_id",
        "value_steer",
        "value_base",
        "delta",
    ]
    paired_out = paired_df[[c for c in paired_keep if c in paired_df.columns]].copy()

    summary_path = out_dir / f"{args.run_name}_bootstrap_summary.csv"
    samples_path = out_dir / f"{args.run_name}_bootstrap_samples.csv"
    paired_path = out_dir / f"{args.run_name}_paired_unit_deltas.csv"
    meta_path = out_dir / f"{args.run_name}_metadata.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")
    paired_out.to_csv(paired_path, index=False, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "ex11_classified": str(ex11_path),
            "ex12_classified": str(ex12_path),
            "ex13_classified": str(ex13_path),
            "ex17_classified": str(ex17_path),
        },
        "config": {
            "layer": int(args.layer),
            "alphas": [float(a) for a in alphas],
            "baseline_alpha": float(args.baseline_alpha),
            "persona_baseline": args.persona_baseline,
            "persona_targets": persona_targets,
            "persona_metrics": persona_metrics,
            "bootstrap_iters": int(args.bootstrap_iters),
            "seed": int(args.seed),
        },
        "outputs": {
            "summary_csv": str(summary_path),
            "samples_csv": str(samples_path),
            "paired_unit_deltas_csv": str(paired_path),
        },
        "n_analyses": int(summary_df.shape[0]),
    }
    save_metadata(meta_path, metadata)

    print(f"[done] summary={summary_path}")
    print(f"[done] samples={samples_path}")
    print(f"[done] paired={paired_path}")
    print(f"[done] metadata={meta_path}")


if __name__ == "__main__":
    main()
