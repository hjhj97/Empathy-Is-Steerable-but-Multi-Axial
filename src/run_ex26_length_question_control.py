#!/usr/bin/env python3
"""Ex26: length/question-controlled post-hoc analysis for Ex25 and Ex11/12/13.

This script reuses existing classified CSV outputs:
1) Prompt baseline (Ex25): P1/P2/P3 vs P0
2) Steering runs (Ex11/Ex12/Ex13): alpha=+1 vs alpha=0 at layer 15

Outputs:
- surface features
- descriptive summaries
- regression-controlled deltas
- length-matched deltas (with per-bin rows)
- paired bootstrap summaries (raw and controlled)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/ex26_length_question_control")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex26 length/question-controlled analysis")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)

    p.add_argument(
        "--ex25-pattern",
        type=str,
        default="outputs/ex25_prompt_mechanism_baseline/ex25_prompt_mechanism_baseline_*_n200_v2_classified.csv",
    )
    p.add_argument(
        "--ex25-fallback-pattern",
        type=str,
        default="outputs/ex25_prompt_mechanism_baseline/ex25_prompt_mechanism_baseline_*_n200_classified.csv",
    )

    p.add_argument("--ex11-pattern", type=str, default="outputs/ex11_filtered_er/ex11_filtered_er_*_n200_classified.csv")
    p.add_argument("--ex12-pattern", type=str, default="outputs/ex12_filtered_ex/ex12_filtered_ex_*_n200_classified.csv")
    p.add_argument("--ex13-pattern", type=str, default="outputs/ex13_filtered_ip/ex13_filtered_ip_*_n200_classified.csv")

    p.add_argument("--prompt-baseline-condition", type=str, default="P0")
    p.add_argument("--prompt-target-conditions", type=str, default="P1,P2,P3")
    p.add_argument("--metrics", type=str, default="ER_label,IP_label,EX_label")

    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--baseline-alpha", type=float, default=0.0)
    p.add_argument("--steering-alpha", type=float, default=1.0)

    p.add_argument("--length-feature", type=str, default="word_len", choices=["char_len", "word_len", "token_len"])
    p.add_argument("--length-bins", type=int, default=5)

    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", type=str, default="ex26_length_question_control_n200")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in (raw or "").split(","):
        v = tok.strip()
        if not v:
            continue
        low = v.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(v)
    return out


def model_key_from_name(model_name: str) -> str:
    m = str(model_name or "").lower()
    if "llama-3.1-8b-instruct" in m:
        return "llama31"
    if "qwen2.5-7b-instruct" in m or "qwen2.5" in m:
        return "qwen25"
    if "mistral-7b-instruct-v0.3" in m or "mistral-7b" in m:
        return "mistral7b"
    return re.sub(r"[^a-z0-9]+", "", m)[:24] or "model"


def detect_response_col(df: pd.DataFrame) -> str:
    for c in ["generated_response", "response_post", "response"]:
        if c in df.columns:
            return c
    raise ValueError("No response text column found. Need one of: generated_response, response_post, response")


def pick_present_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def choose_pairing_keys(df: pd.DataFrame, priority: Sequence[str]) -> List[str]:
    present = pick_present_cols(df, priority)
    if not present:
        return []
    return present


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_col = detect_response_col(out)
    text = out[text_col].fillna("").astype(str)

    out["response_text"] = text
    out["char_len"] = text.str.len().astype(float)
    out["word_len"] = text.str.split().str.len().fillna(0).astype(float)

    if "response_token_count" in out.columns:
        out["token_len"] = pd.to_numeric(out["response_token_count"], errors="coerce").fillna(out["word_len"]).astype(float)
    else:
        out["token_len"] = out["word_len"].astype(float)

    out["question_count"] = text.str.count(r"\?").astype(float)
    sentence_marks = text.str.count(r"[.!?]+").astype(float)
    non_empty = text.str.strip().ne("")
    # If no sentence punctuation exists but text is non-empty, treat as one sentence.
    out["sentence_count"] = np.where(non_empty, np.maximum(1.0, sentence_marks), 0.0).astype(float)
    out["question_ratio"] = np.where(out["sentence_count"] > 0, out["question_count"] / out["sentence_count"], 0.0).astype(float)
    out["ends_with_question"] = text.str.rstrip().str.endswith("?").astype(int)
    out["has_question"] = (out["question_count"] >= 1).astype(int)
    return out


def infer_model_key_from_df(path: Path) -> str:
    head = pd.read_csv(path, nrows=1)
    if "model_name" in head.columns and len(head) > 0:
        return model_key_from_name(str(head["model_name"].iloc[0]))
    return model_key_from_name(path.stem)


def newest_paths_by_model(persona_project: Path, pattern: str, fallback_pattern: Optional[str] = None) -> Dict[str, Path]:
    base_matches = sorted((persona_project).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not base_matches and fallback_pattern:
        base_matches = sorted((persona_project).glob(fallback_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    out: Dict[str, Path] = {}
    for p in base_matches:
        key = infer_model_key_from_df(p)
        if key not in out:
            out[key] = p
    return out


def load_prompt_df(paths_by_model: Dict[str, Path]) -> pd.DataFrame:
    if not paths_by_model:
        raise FileNotFoundError("No Ex25 classified files found.")
    frames: List[pd.DataFrame] = []
    for model_key, path in sorted(paths_by_model.items()):
        df = pd.read_csv(path)
        if "condition_id" not in df.columns:
            raise ValueError(f"Ex25 input missing condition_id: {path}")
        df["model_key"] = model_key
        df["analysis_type"] = "prompt"
        df["experiment"] = "ex25"
        df["input_path"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return add_surface_features(out)


def load_steering_df(paths_by_model: Dict[str, Path], experiment: str) -> pd.DataFrame:
    if not paths_by_model:
        raise FileNotFoundError(f"No {experiment} classified files found.")
    frames: List[pd.DataFrame] = []
    for model_key, path in sorted(paths_by_model.items()):
        df = pd.read_csv(path)
        for col in ["layer", "alpha"]:
            if col not in df.columns:
                raise ValueError(f"{experiment} input missing '{col}': {path}")
        df["model_key"] = model_key
        df["analysis_type"] = "steering"
        df["experiment"] = experiment
        df["input_path"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return add_surface_features(out)


def summarize_prompt(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    grp = ["analysis_type", "experiment", "model_key", "condition_id"]
    rows: List[dict] = []
    for key, g in df.groupby(grp, sort=True):
        rec = {k: v for k, v in zip(grp, key)}
        rec["n"] = int(len(g))
        for m in metrics:
            rec[f"{m}_mean"] = float(g[m].mean())
        rec["char_len_mean"] = float(g["char_len"].mean())
        rec["word_len_mean"] = float(g["word_len"].mean())
        rec["token_len_mean"] = float(g["token_len"].mean())
        rec["question_count_mean"] = float(g["question_count"].mean())
        rec["question_ratio_mean"] = float(g["question_ratio"].mean())
        rec["ends_with_question_rate"] = float(g["ends_with_question"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(grp).reset_index(drop=True)


def summarize_steering(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    grp = ["analysis_type", "experiment", "model_key", "layer", "alpha"]
    rows: List[dict] = []
    for key, g in df.groupby(grp, sort=True):
        rec = {k: v for k, v in zip(grp, key)}
        rec["n"] = int(len(g))
        for m in metrics:
            rec[f"{m}_mean"] = float(g[m].mean())
        rec["char_len_mean"] = float(g["char_len"].mean())
        rec["word_len_mean"] = float(g["word_len"].mean())
        rec["token_len_mean"] = float(g["token_len"].mean())
        rec["question_count_mean"] = float(g["question_count"].mean())
        rec["question_ratio_mean"] = float(g["question_ratio"].mean())
        rec["ends_with_question_rate"] = float(g["ends_with_question"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(grp).reset_index(drop=True)


def cond_tag_from_alpha(alpha: float) -> str:
    if float(alpha).is_integer():
        return f"alpha={int(alpha):+d}"
    return f"alpha={alpha:+g}"


def build_pairs(
    df: pd.DataFrame,
    *,
    condition_col: str,
    baseline_value,
    target_value,
    metrics: Sequence[str],
    key_priority: Sequence[str],
    add_model_key_in_pair: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    work = df[df[condition_col].isin([baseline_value, target_value])].copy()
    key_cols = choose_pairing_keys(work, key_priority)
    if add_model_key_in_pair and "model_key" in work.columns and "model_key" not in key_cols:
        key_cols = ["model_key"] + key_cols
    if not key_cols:
        raise ValueError("Could not infer pairing keys for comparison.")

    agg_cols = list(metrics) + ["char_len", "word_len", "token_len", "question_count", "question_ratio"]

    base = (
        work[work[condition_col] == baseline_value]
        .groupby(key_cols, as_index=False)[agg_cols]
        .mean()
        .rename(columns={c: f"{c}_base" for c in agg_cols})
    )
    tgt = (
        work[work[condition_col] == target_value]
        .groupby(key_cols, as_index=False)[agg_cols]
        .mean()
        .rename(columns={c: f"{c}_tgt" for c in agg_cols})
    )
    pair = tgt.merge(base, on=key_cols, how="inner")
    return pair, key_cols


def safe_std(x: np.ndarray) -> float:
    s = float(np.nanstd(x))
    return s if s > 0 else 1.0


def fit_controlled_ols(
    y: np.ndarray,
    cond: np.ndarray,
    length: np.ndarray,
    qcount: np.ndarray,
    model_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = y.astype(float)
    cond = cond.astype(float)
    length = length.astype(float)
    qcount = qcount.astype(float)

    length_z = (length - float(np.nanmean(length))) / safe_std(length)
    qcount_z = (qcount - float(np.nanmean(qcount))) / safe_std(qcount)

    cols = [
        np.ones_like(cond),
        cond,
        length_z,
        qcount_z,
    ]
    names = ["intercept", "cond", "length_z", "question_count_z"]

    if model_labels is not None:
        labels = pd.Series(model_labels.astype(str))
        if labels.nunique(dropna=False) > 1:
            dummies = pd.get_dummies(labels, drop_first=True)
            for c in dummies.columns:
                cols.append(dummies[c].to_numpy(dtype=float))
                names.append(f"model_{c}")

    X = np.column_stack(cols).astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = {n: float(v) for n, v in zip(names, beta)}
    return out


def paired_bootstrap_stats(samples: np.ndarray) -> Dict[str, float]:
    samples = samples.astype(float)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    p_lo = float(np.mean(samples <= 0.0))
    p_hi = float(np.mean(samples >= 0.0))
    p_two = min(1.0, 2.0 * min(p_lo, p_hi))
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
        "ci_low_95": float(lo),
        "ci_high_95": float(hi),
        "p_two_sided_vs0": float(p_two),
        "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
    }


def bootstrap_raw_and_controlled(
    pair_df: pd.DataFrame,
    metric: str,
    length_feature: str,
    *,
    include_model_fe: bool,
    iters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pair_df)
    if n <= 0:
        raise ValueError("Empty pair_df for bootstrap.")
    rng = np.random.default_rng(seed)

    raw_out = np.zeros(iters, dtype=float)
    ctl_out = np.zeros(iters, dtype=float)

    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        sample = pair_df.iloc[idx].reset_index(drop=True)

        d = sample[f"{metric}_tgt"].to_numpy(float) - sample[f"{metric}_base"].to_numpy(float)
        raw_out[i] = float(d.mean())

        y = np.concatenate([sample[f"{metric}_base"].to_numpy(float), sample[f"{metric}_tgt"].to_numpy(float)])
        cond = np.concatenate([np.zeros(n, dtype=float), np.ones(n, dtype=float)])
        length = np.concatenate([sample[f"{length_feature}_base"].to_numpy(float), sample[f"{length_feature}_tgt"].to_numpy(float)])
        qcount = np.concatenate([sample["question_count_base"].to_numpy(float), sample["question_count_tgt"].to_numpy(float)])

        model_labels = None
        if include_model_fe and "model_key" in sample.columns:
            mk = sample["model_key"].astype(str).to_numpy()
            model_labels = np.concatenate([mk, mk])

        coef = fit_controlled_ols(y, cond, length, qcount, model_labels=model_labels)
        ctl_out[i] = float(coef["cond"])
    return raw_out, ctl_out


def length_match_bins(
    pair_df: pd.DataFrame,
    metric: str,
    length_feature: str,
    n_bins: int,
) -> Tuple[float, pd.DataFrame]:
    if len(pair_df) == 0:
        return float("nan"), pd.DataFrame()

    work = pair_df.copy()
    work["delta"] = work[f"{metric}_tgt"] - work[f"{metric}_base"]
    work["pair_avg_len"] = (work[f"{length_feature}_base"] + work[f"{length_feature}_tgt"]) / 2.0

    unique_len = np.unique(work["pair_avg_len"].to_numpy(float))
    q = int(max(1, min(int(n_bins), len(unique_len))))
    if q == 1:
        work["length_bin"] = "bin_0"
    else:
        work["length_bin"] = pd.qcut(work["pair_avg_len"], q=q, duplicates="drop")
        work["length_bin"] = work["length_bin"].astype(str)

    rows: List[dict] = []
    for b, g in work.groupby("length_bin", sort=True):
        rows.append(
            {
                "length_bin": str(b),
                "n_pairs": int(len(g)),
                "delta_mean": float(g["delta"].mean()),
                "pair_avg_len_mean": float(g["pair_avg_len"].mean()),
                "q0_pair_rate": float(((g["question_count_base"] == 0) & (g["question_count_tgt"] == 0)).mean()),
                "q1plus_pair_rate": float(((g["question_count_base"] >= 1) | (g["question_count_tgt"] >= 1)).mean()),
            }
        )
    bins_df = pd.DataFrame(rows)
    if bins_df.empty:
        return float("nan"), bins_df

    weighted = float(np.average(bins_df["delta_mean"].to_numpy(float), weights=bins_df["n_pairs"].to_numpy(float)))
    return weighted, bins_df


def run_comparison(
    pair_df: pd.DataFrame,
    *,
    analysis_type: str,
    experiment: str,
    model_scope: str,
    comparison: str,
    metrics: Sequence[str],
    length_feature: str,
    length_bins: int,
    include_model_fe: bool,
    bootstrap_iters: int,
    seed_base: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    reg_rows: List[dict] = []
    len_rows: List[dict] = []
    boot_rows: List[dict] = []

    for mi, metric in enumerate(metrics):
        if f"{metric}_base" not in pair_df.columns or f"{metric}_tgt" not in pair_df.columns:
            continue
        n_pairs = len(pair_df)
        if n_pairs == 0:
            continue

        raw_delta = pair_df[f"{metric}_tgt"].to_numpy(float) - pair_df[f"{metric}_base"].to_numpy(float)
        raw_delta_mean = float(raw_delta.mean())

        y = np.concatenate([pair_df[f"{metric}_base"].to_numpy(float), pair_df[f"{metric}_tgt"].to_numpy(float)])
        cond = np.concatenate([np.zeros(n_pairs, dtype=float), np.ones(n_pairs, dtype=float)])
        length = np.concatenate([pair_df[f"{length_feature}_base"].to_numpy(float), pair_df[f"{length_feature}_tgt"].to_numpy(float)])
        qcount = np.concatenate([pair_df["question_count_base"].to_numpy(float), pair_df["question_count_tgt"].to_numpy(float)])

        model_labels = None
        if include_model_fe and "model_key" in pair_df.columns:
            mk = pair_df["model_key"].astype(str).to_numpy()
            model_labels = np.concatenate([mk, mk])

        coef = fit_controlled_ols(y, cond, length, qcount, model_labels=model_labels)
        controlled_delta = float(coef["cond"])

        lm_delta, lm_bins = length_match_bins(pair_df, metric=metric, length_feature=length_feature, n_bins=length_bins)
        if not lm_bins.empty:
            lm_bins = lm_bins.copy()
            lm_bins["analysis_type"] = analysis_type
            lm_bins["experiment"] = experiment
            lm_bins["model_scope"] = model_scope
            lm_bins["comparison"] = comparison
            lm_bins["metric"] = metric
            len_rows.extend(lm_bins.to_dict(orient="records"))

        raw_boot, ctl_boot = bootstrap_raw_and_controlled(
            pair_df=pair_df,
            metric=metric,
            length_feature=length_feature,
            include_model_fe=include_model_fe,
            iters=bootstrap_iters,
            seed=seed_base + (mi * 1000),
        )
        raw_stats = paired_bootstrap_stats(raw_boot)
        ctl_stats = paired_bootstrap_stats(ctl_boot)

        reg_row = {
            "analysis_type": analysis_type,
            "experiment": experiment,
            "model_scope": model_scope,
            "comparison": comparison,
            "metric": metric,
            "n_pairs": int(n_pairs),
            "raw_delta_mean": raw_delta_mean,
            "controlled_delta": controlled_delta,
            "length_matched_delta": float(lm_delta),
            "delta_reduction_ratio": float((raw_delta_mean - controlled_delta) / raw_delta_mean)
            if abs(raw_delta_mean) > 1e-12
            else np.nan,
            "coef_length_z": float(coef.get("length_z", np.nan)),
            "coef_question_count_z": float(coef.get("question_count_z", np.nan)),
            "raw_ci_low_95": float(raw_stats["ci_low_95"]),
            "raw_ci_high_95": float(raw_stats["ci_high_95"]),
            "raw_ci_excludes_zero": bool(raw_stats["ci_excludes_zero"]),
            "controlled_ci_low_95": float(ctl_stats["ci_low_95"]),
            "controlled_ci_high_95": float(ctl_stats["ci_high_95"]),
            "controlled_ci_excludes_zero": bool(ctl_stats["ci_excludes_zero"]),
        }
        reg_rows.append(reg_row)

        boot_rows.append(
            {
                "analysis_type": analysis_type,
                "experiment": experiment,
                "model_scope": model_scope,
                "comparison": comparison,
                "metric": metric,
                "n_pairs": int(n_pairs),
                "bootstrap_iters": int(bootstrap_iters),
                "raw_mean": float(raw_stats["mean"]),
                "raw_std": float(raw_stats["std"]),
                "raw_ci_low_95": float(raw_stats["ci_low_95"]),
                "raw_ci_high_95": float(raw_stats["ci_high_95"]),
                "raw_p_two_sided_vs0": float(raw_stats["p_two_sided_vs0"]),
                "raw_ci_excludes_zero": bool(raw_stats["ci_excludes_zero"]),
                "controlled_mean": float(ctl_stats["mean"]),
                "controlled_std": float(ctl_stats["std"]),
                "controlled_ci_low_95": float(ctl_stats["ci_low_95"]),
                "controlled_ci_high_95": float(ctl_stats["ci_high_95"]),
                "controlled_p_two_sided_vs0": float(ctl_stats["p_two_sided_vs0"]),
                "controlled_ci_excludes_zero": bool(ctl_stats["ci_excludes_zero"]),
            }
        )

    return reg_rows, len_rows, boot_rows


def main() -> None:
    args = parse_args()
    persona_project = args.persona_project.resolve()
    out_dir = resolve_path(persona_project, args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_csv_list(args.metrics)
    prompt_targets = parse_csv_list(args.prompt_target_conditions)
    if not metrics:
        raise ValueError("No metrics parsed from --metrics")
    if not prompt_targets:
        raise ValueError("No prompt targets parsed from --prompt-target-conditions")
    if int(args.bootstrap_iters) <= 0:
        raise ValueError("--bootstrap-iters must be positive")
    if int(args.length_bins) <= 0:
        raise ValueError("--length-bins must be positive")

    run_name = args.run_name
    surface_path = out_dir / f"{run_name}_surface_features.csv"
    desc_path = out_dir / f"{run_name}_descriptive_summary.csv"
    reg_path = out_dir / f"{run_name}_regression_summary.csv"
    lm_path = out_dir / f"{run_name}_length_matched_summary.csv"
    boot_path = out_dir / f"{run_name}_bootstrap_summary.csv"
    meta_path = out_dir / f"{run_name}_metadata.json"
    outputs = [surface_path, desc_path, reg_path, lm_path, boot_path, meta_path]
    for p in outputs:
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")

    ex25_paths = newest_paths_by_model(persona_project, args.ex25_pattern, args.ex25_fallback_pattern)
    ex11_paths = newest_paths_by_model(persona_project, args.ex11_pattern)
    ex12_paths = newest_paths_by_model(persona_project, args.ex12_pattern)
    ex13_paths = newest_paths_by_model(persona_project, args.ex13_pattern)
    if not ex11_paths or not ex12_paths or not ex13_paths:
        raise FileNotFoundError("Missing Ex11/Ex12/Ex13 classified inputs.")

    prompt_df = load_prompt_df(ex25_paths)
    ex11_df = load_steering_df(ex11_paths, "ex11")
    ex12_df = load_steering_df(ex12_paths, "ex12")
    ex13_df = load_steering_df(ex13_paths, "ex13")

    steer_all = pd.concat([ex11_df, ex12_df, ex13_df], ignore_index=True)
    steer_all["alpha"] = pd.to_numeric(steer_all["alpha"], errors="coerce")
    steer_all["layer"] = pd.to_numeric(steer_all["layer"], errors="coerce")
    steer_all = steer_all[
        (steer_all["layer"] == int(args.layer))
        & (np.isclose(steer_all["alpha"], float(args.baseline_alpha)) | np.isclose(steer_all["alpha"], float(args.steering_alpha)))
    ].copy()
    steer_all["condition_id"] = steer_all["alpha"].map(cond_tag_from_alpha)

    prompt_surface = prompt_df.copy()
    prompt_surface["condition_kind"] = "prompt_condition"
    prompt_surface["condition_value"] = prompt_surface["condition_id"].astype(str)
    prompt_surface["layer"] = np.nan
    prompt_surface["alpha"] = np.nan

    steer_surface = steer_all.copy()
    steer_surface["condition_kind"] = "alpha"
    steer_surface["condition_value"] = steer_surface["condition_id"].astype(str)

    keep_cols = [
        "analysis_type",
        "experiment",
        "model_key",
        "condition_kind",
        "condition_value",
        "layer",
        "alpha",
        "source_id",
        "eval_index",
        "base_sample_index",
        "response_rank",
        "response_text",
        "char_len",
        "word_len",
        "token_len",
        "question_count",
        "sentence_count",
        "question_ratio",
        "ends_with_question",
        "ER_label",
        "IP_label",
        "EX_label",
        "input_path",
    ]
    surface_df = pd.concat([prompt_surface, steer_surface], ignore_index=True)
    surface_df = surface_df[[c for c in keep_cols if c in surface_df.columns]].copy()
    surface_df.to_csv(surface_path, index=False, encoding="utf-8-sig")

    desc_prompt = summarize_prompt(prompt_df, metrics)
    desc_steer = summarize_steering(steer_all, metrics)
    desc_df = pd.concat([desc_prompt, desc_steer], ignore_index=True, sort=False)
    desc_df.to_csv(desc_path, index=False, encoding="utf-8-sig")

    regression_rows: List[dict] = []
    length_rows: List[dict] = []
    bootstrap_rows: List[dict] = []
    seed_cursor = int(args.seed)

    # Prompt comparisons: per model + pooled.
    for model_scope in sorted(prompt_df["model_key"].dropna().astype(str).unique().tolist()) + ["all_models"]:
        if model_scope == "all_models":
            sub = prompt_df.copy()
            add_model_fe = True
            add_model_key_in_pair = True
        else:
            sub = prompt_df[prompt_df["model_key"] == model_scope].copy()
            add_model_fe = False
            add_model_key_in_pair = False

        for tgt in prompt_targets:
            pair_df, _ = build_pairs(
                sub,
                condition_col="condition_id",
                baseline_value=args.prompt_baseline_condition,
                target_value=tgt,
                metrics=metrics,
                key_priority=["source_id", "eval_index", "base_sample_index"],
                add_model_key_in_pair=add_model_key_in_pair,
            )
            if pair_df.empty:
                continue
            reg, lm, bt = run_comparison(
                pair_df,
                analysis_type="prompt",
                experiment="ex25",
                model_scope=model_scope,
                comparison=f"{tgt}_vs_{args.prompt_baseline_condition}",
                metrics=metrics,
                length_feature=args.length_feature,
                length_bins=int(args.length_bins),
                include_model_fe=add_model_fe,
                bootstrap_iters=int(args.bootstrap_iters),
                seed_base=seed_cursor,
            )
            seed_cursor += 100_000
            regression_rows.extend(reg)
            length_rows.extend(lm)
            bootstrap_rows.extend(bt)

    # Steering comparisons: ex11/ex12/ex13, per model + pooled.
    for exp in ["ex11", "ex12", "ex13"]:
        exp_df = steer_all[steer_all["experiment"] == exp].copy()
        for model_scope in sorted(exp_df["model_key"].dropna().astype(str).unique().tolist()) + ["all_models"]:
            if model_scope == "all_models":
                sub = exp_df.copy()
                add_model_fe = True
                add_model_key_in_pair = True
            else:
                sub = exp_df[exp_df["model_key"] == model_scope].copy()
                add_model_fe = False
                add_model_key_in_pair = False

            pair_df, _ = build_pairs(
                sub,
                condition_col="alpha",
                baseline_value=float(args.baseline_alpha),
                target_value=float(args.steering_alpha),
                metrics=metrics,
                key_priority=["base_sample_index", "source_id", "response_rank", "eval_index"],
                add_model_key_in_pair=add_model_key_in_pair,
            )
            if pair_df.empty:
                continue
            reg, lm, bt = run_comparison(
                pair_df,
                analysis_type="steering",
                experiment=exp,
                model_scope=model_scope,
                comparison=f"{cond_tag_from_alpha(float(args.steering_alpha))}_vs_{cond_tag_from_alpha(float(args.baseline_alpha))}",
                metrics=metrics,
                length_feature=args.length_feature,
                length_bins=int(args.length_bins),
                include_model_fe=add_model_fe,
                bootstrap_iters=int(args.bootstrap_iters),
                seed_base=seed_cursor,
            )
            seed_cursor += 100_000
            regression_rows.extend(reg)
            length_rows.extend(lm)
            bootstrap_rows.extend(bt)

    reg_df = pd.DataFrame(regression_rows)
    lm_df = pd.DataFrame(length_rows)
    boot_df = pd.DataFrame(bootstrap_rows)

    if not reg_df.empty:
        reg_df = reg_df.sort_values(["analysis_type", "experiment", "model_scope", "comparison", "metric"]).reset_index(drop=True)
    if not lm_df.empty:
        lm_df = lm_df.sort_values(["analysis_type", "experiment", "model_scope", "comparison", "metric", "length_bin"]).reset_index(drop=True)
    if not boot_df.empty:
        boot_df = boot_df.sort_values(["analysis_type", "experiment", "model_scope", "comparison", "metric"]).reset_index(drop=True)

    reg_df.to_csv(reg_path, index=False, encoding="utf-8-sig")
    lm_df.to_csv(lm_path, index=False, encoding="utf-8-sig")
    boot_df.to_csv(boot_path, index=False, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "ex26_length_question_control",
        "inputs": {
            "ex25_paths_by_model": {k: str(v) for k, v in ex25_paths.items()},
            "ex11_paths_by_model": {k: str(v) for k, v in ex11_paths.items()},
            "ex12_paths_by_model": {k: str(v) for k, v in ex12_paths.items()},
            "ex13_paths_by_model": {k: str(v) for k, v in ex13_paths.items()},
        },
        "config": {
            "prompt_baseline_condition": args.prompt_baseline_condition,
            "prompt_target_conditions": prompt_targets,
            "metrics": metrics,
            "layer": int(args.layer),
            "baseline_alpha": float(args.baseline_alpha),
            "steering_alpha": float(args.steering_alpha),
            "length_feature": args.length_feature,
            "length_bins": int(args.length_bins),
            "bootstrap_iters": int(args.bootstrap_iters),
            "seed": int(args.seed),
        },
        "outputs": {
            "surface_features_csv": str(surface_path),
            "descriptive_summary_csv": str(desc_path),
            "regression_summary_csv": str(reg_path),
            "length_matched_summary_csv": str(lm_path),
            "bootstrap_summary_csv": str(boot_path),
        },
        "counts": {
            "surface_rows": int(len(surface_df)),
            "descriptive_rows": int(len(desc_df)),
            "regression_rows": int(len(reg_df)),
            "length_matched_rows": int(len(lm_df)),
            "bootstrap_rows": int(len(boot_df)),
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] surface={surface_path}")
    print(f"[done] descriptive={desc_path}")
    print(f"[done] regression={reg_path}")
    print(f"[done] length_matched={lm_path}")
    print(f"[done] bootstrap={boot_path}")
    print(f"[done] metadata={meta_path}")


if __name__ == "__main__":
    main()
