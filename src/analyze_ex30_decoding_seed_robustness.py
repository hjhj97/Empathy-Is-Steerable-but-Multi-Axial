#!/usr/bin/env python3
"""Ex30 decoding-seed robustness analysis for canonical layer-15 steering.

The script performs no generation or classifier scoring. It combines the
existing Ex11/Ex12/Ex13 Llama canonical summaries (decoding seed 12) with new
Ex30 Llama summaries, verifies that Ex30 runs keep the canonical eval seeker
split, and reports target-mechanism deltas across decoding seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MECHANISM_ORDER = ["ER", "EX", "IP"]
MECHANISM_CONFIG = {
    "ER": {
        "exp_tag": "ex11",
        "canonical_dir": Path("outputs/ex11_filtered_er"),
        "canonical_run": "ex11_filtered_er_llama31_n200",
        "mean_column": "ER_mean",
        "delta_column": "delta_vs_alpha0",
    },
    "EX": {
        "exp_tag": "ex12",
        "canonical_dir": Path("outputs/ex12_filtered_ex"),
        "canonical_run": "ex12_filtered_ex_llama31_n200",
        "mean_column": "EX_mean",
        "delta_column": "delta_vs_alpha0",
    },
    "IP": {
        "exp_tag": "ex13",
        "canonical_dir": Path("outputs/ex13_filtered_ip"),
        "canonical_run": "ex13_filtered_ip_llama31_n200",
        "mean_column": "IP_mean",
        "delta_column": "delta_vs_alpha0",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex30 decoding-seed robustness analysis")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument(
        "--ex30-dir",
        type=Path,
        default=Path("outputs/ex30_decoding_seed_robustness"),
    )
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 1.0])
    p.add_argument("--baseline-alpha", type=float, default=0.0)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--required-seeds", type=int, nargs="+", default=[12, 22, 32])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ex30_decoding_seed_robustness"),
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="ex30_decoding_seed_robustness_l15",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def read_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sibling_path(metadata_path: Path, suffix: str) -> Path:
    suffix_token = "_metadata.json"
    if not metadata_path.name.endswith(suffix_token):
        raise ValueError(f"Unexpected metadata filename: {metadata_path}")
    run_prefix = metadata_path.name[: -len(suffix_token)]
    return metadata_path.with_name(f"{run_prefix}_{suffix}")


def seed_from_metadata(metadata: Dict[str, object], path: Path) -> int:
    args = metadata.get("args")
    if not isinstance(args, dict) or "seed" not in args:
        raise ValueError(f"Metadata lacks args.seed: {path}")
    return int(args["seed"])


def sample_seed_from_metadata(metadata: Dict[str, object], path: Path) -> int:
    args = metadata.get("args")
    if not isinstance(args, dict) or "sample_seed" not in args:
        raise ValueError(f"Metadata lacks args.sample_seed: {path}")
    return int(args["sample_seed"])


def model_from_metadata(metadata: Dict[str, object], path: Path) -> str:
    model_name = metadata.get("model_name")
    if not isinstance(model_name, str):
        raise ValueError(f"Metadata lacks model_name: {path}")
    return model_name


def is_llama31(model_name: str) -> bool:
    lowered = model_name.lower()
    return "llama-3.1-8b-instruct" in lowered or "llama31" in lowered


def canonical_input(project: Path, mechanism: str) -> Dict[str, object]:
    config = MECHANISM_CONFIG[mechanism]
    run_dir = resolve_path(project, config["canonical_dir"]).resolve()
    run_name = str(config["canonical_run"])
    metadata_path = run_dir / f"{run_name}_metadata.json"
    metadata = read_metadata(metadata_path)
    return make_input_row(
        mechanism=mechanism,
        summary_path=run_dir / f"{run_name}_summary.csv",
        metadata_path=metadata_path,
        eval_seekers_path=run_dir / f"{run_name}_eval_seekers.csv",
        metadata=metadata,
        source="canonical",
    )


def make_input_row(
    mechanism: str,
    summary_path: Path,
    metadata_path: Path,
    eval_seekers_path: Path,
    metadata: Dict[str, object],
    source: str,
) -> Dict[str, object]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not eval_seekers_path.exists():
        raise FileNotFoundError(f"Missing eval seekers CSV: {eval_seekers_path}")
    model_name = model_from_metadata(metadata, metadata_path)
    if not is_llama31(model_name):
        raise ValueError(f"Ex30 only expects Llama-3.1 inputs, got {model_name} in {metadata_path}")
    return {
        "mechanism": mechanism,
        "source": source,
        "decoding_seed": seed_from_metadata(metadata, metadata_path),
        "sample_seed": sample_seed_from_metadata(metadata, metadata_path),
        "model_name": model_name,
        "summary_csv": str(summary_path.resolve()),
        "metadata_json": str(metadata_path.resolve()),
        "eval_seekers_csv": str(eval_seekers_path.resolve()),
    }


def ex30_inputs(project: Path, ex30_dir: Path, mechanism: str) -> List[Dict[str, object]]:
    exp_tag = str(MECHANISM_CONFIG[mechanism]["exp_tag"])
    run_dir = resolve_path(project, ex30_dir / exp_tag).resolve()
    if not run_dir.exists():
        return []
    rows: List[Dict[str, object]] = []
    for metadata_path in sorted(run_dir.glob("*_metadata.json")):
        metadata = read_metadata(metadata_path)
        rows.append(
            make_input_row(
                mechanism=mechanism,
                summary_path=sibling_path(metadata_path, "summary.csv"),
                metadata_path=metadata_path,
                eval_seekers_path=sibling_path(metadata_path, "eval_seekers.csv"),
                metadata=metadata,
                source="ex30",
            )
        )
    return rows


def discover_inputs(args: argparse.Namespace, project: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for mechanism in MECHANISM_ORDER:
        rows.append(canonical_input(project, mechanism))
        rows.extend(ex30_inputs(project, resolve_path(project, args.ex30_dir), mechanism))

    inputs = pd.DataFrame(rows)
    duplicates = inputs.duplicated(["mechanism", "decoding_seed"], keep=False)
    if duplicates.any():
        cols = ["mechanism", "decoding_seed", "summary_csv"]
        raise ValueError(
            "Duplicate mechanism/decoding-seed inputs found:\n"
            + inputs.loc[duplicates, cols].to_string(index=False)
        )

    bad_sample_seeds = inputs[inputs["sample_seed"] != int(args.sample_seed)]
    if not bad_sample_seeds.empty:
        cols = ["mechanism", "decoding_seed", "sample_seed", "metadata_json"]
        raise ValueError(
            f"Ex30 requires sample_seed={args.sample_seed}; mismatched runs:\n"
            + bad_sample_seeds[cols].to_string(index=False)
        )

    required = {(mechanism, int(seed)) for mechanism in MECHANISM_ORDER for seed in args.required_seeds}
    observed = set(zip(inputs["mechanism"], inputs["decoding_seed"]))
    missing = sorted(required - observed, key=lambda item: (MECHANISM_ORDER.index(item[0]), item[1]))
    if missing:
        raise FileNotFoundError(f"Missing required Ex30 mechanism/decoding-seed inputs: {missing}")

    keep = inputs["decoding_seed"].isin([int(seed) for seed in args.required_seeds])
    inputs = inputs[keep].copy()
    inputs["_mechanism_order"] = inputs["mechanism"].map(MECHANISM_ORDER.index)
    return inputs.sort_values(["_mechanism_order", "decoding_seed"]).drop(
        columns="_mechanism_order"
    ).reset_index(drop=True)


def source_ids(path: Path) -> List[str]:
    seekers = pd.read_csv(path)
    require_columns(seekers, ["source_id"], path)
    return seekers["source_id"].astype(str).tolist()


def split_checks(inputs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for mechanism in MECHANISM_ORDER:
        mechanism_inputs = inputs[inputs["mechanism"] == mechanism]
        canonical = mechanism_inputs[mechanism_inputs["source"] == "canonical"]
        if len(canonical) != 1:
            raise ValueError(f"Expected exactly one canonical row for {mechanism}")
        canonical_row = canonical.iloc[0]
        canonical_ids = source_ids(Path(str(canonical_row["eval_seekers_csv"])))
        for _, input_row in mechanism_inputs.iterrows():
            ids = source_ids(Path(str(input_row["eval_seekers_csv"])))
            matches = ids == canonical_ids
            rows.append(
                {
                    "mechanism": mechanism,
                    "source": input_row["source"],
                    "decoding_seed": int(input_row["decoding_seed"]),
                    "sample_seed": int(input_row["sample_seed"]),
                    "n_eval_seekers": int(len(ids)),
                    "eval_split_matches_canonical": bool(matches),
                    "eval_seekers_csv": input_row["eval_seekers_csv"],
                }
            )
    checks = pd.DataFrame(rows)
    mismatched = checks[~checks["eval_split_matches_canonical"]]
    if not mismatched.empty:
        cols = ["mechanism", "decoding_seed", "eval_seekers_csv"]
        raise ValueError(
            "Ex30 eval seeker split differs from canonical split:\n"
            + mismatched[cols].to_string(index=False)
        )
    return checks


def summary_rows(input_row: pd.Series, layer: int, alphas: List[float], baseline_alpha: float) -> List[Dict[str, object]]:
    mechanism = str(input_row["mechanism"])
    config = MECHANISM_CONFIG[mechanism]
    path = Path(str(input_row["summary_csv"]))
    summary = pd.read_csv(path)
    mean_col = str(config["mean_column"])
    delta_col = str(config["delta_column"])
    require_columns(summary, ["layer", "alpha", "n", mean_col, delta_col], path)

    numeric_layer = pd.to_numeric(summary["layer"], errors="raise").astype(int)
    numeric_alpha = pd.to_numeric(summary["alpha"], errors="raise").astype(float)
    layer_rows = summary[numeric_layer == int(layer)].copy()
    baseline = layer_rows[np.isclose(pd.to_numeric(layer_rows["alpha"]), float(baseline_alpha))]
    if len(baseline) != 1:
        raise ValueError(f"Expected one layer={layer}, alpha={baseline_alpha:g} row in {path}")
    baseline_row = baseline.iloc[0]

    rows: List[Dict[str, object]] = []
    for alpha in alphas:
        current = layer_rows[np.isclose(pd.to_numeric(layer_rows["alpha"]), float(alpha))]
        if len(current) != 1:
            raise ValueError(f"Expected one layer={layer}, alpha={alpha:g} row in {path}")
        row = current.iloc[0]
        rows.append(
            {
                "mechanism": mechanism,
                "source": input_row["source"],
                "model_name": input_row["model_name"],
                "decoding_seed": int(input_row["decoding_seed"]),
                "sample_seed": int(input_row["sample_seed"]),
                "layer": int(layer),
                "alpha": float(alpha),
                "expected_direction": "positive" if alpha > baseline_alpha else "negative",
                "n": int(row["n"]),
                "baseline_n": int(baseline_row["n"]),
                "target_mean": float(row[mean_col]),
                "baseline_target_mean": float(baseline_row[mean_col]),
                "delta_vs_alpha0": float(row[delta_col]),
                "summary_csv": str(path),
            }
        )
    return rows


def seed_level_summary(inputs: pd.DataFrame, layer: int, alphas: List[float], baseline_alpha: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, input_row in inputs.iterrows():
        rows.extend(summary_rows(input_row, layer, alphas, baseline_alpha))
    out = pd.DataFrame(rows)
    out["_mechanism_order"] = out["mechanism"].map(MECHANISM_ORDER.index)
    return out.sort_values(["_mechanism_order", "alpha", "decoding_seed"]).drop(
        columns="_mechanism_order"
    ).reset_index(drop=True)


def direction_ok(delta: float, alpha: float, baseline_alpha: float) -> bool:
    if alpha > baseline_alpha:
        return delta > 0.0
    if alpha < baseline_alpha:
        return delta < 0.0
    raise ValueError("Direction check requires a non-baseline alpha")


def direction_summary(seed_level: pd.DataFrame, baseline_alpha: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (mechanism, alpha), group in seed_level.groupby(["mechanism", "alpha"], sort=False):
        deltas = group["delta_vs_alpha0"].astype(float)
        ok = [direction_ok(float(delta), float(alpha), baseline_alpha) for delta in deltas]
        rows.append(
            {
                "mechanism": mechanism,
                "layer": int(group["layer"].iloc[0]),
                "alpha": float(alpha),
                "expected_direction": group["expected_direction"].iloc[0],
                "n_decoding_seeds": int(group["decoding_seed"].nunique()),
                "decoding_seeds": ",".join(str(int(seed)) for seed in group["decoding_seed"]),
                "delta_mean": float(deltas.mean()),
                "delta_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "delta_min": float(deltas.min()),
                "delta_max": float(deltas.max()),
                "direction_match_count": int(sum(ok)),
                "all_directions_match": bool(all(ok)),
            }
        )
    out = pd.DataFrame(rows)
    out["_mechanism_order"] = out["mechanism"].map(MECHANISM_ORDER.index)
    return out.sort_values(["_mechanism_order", "alpha"]).drop(
        columns="_mechanism_order"
    ).reset_index(drop=True)


def write_outputs(
    args: argparse.Namespace,
    out_dir: Path,
    inputs: pd.DataFrame,
    checks: pd.DataFrame,
    seed_level: pd.DataFrame,
    directions: pd.DataFrame,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_manifest_csv": out_dir / f"{args.run_name}_input_manifest.csv",
        "split_check_csv": out_dir / f"{args.run_name}_split_check.csv",
        "seed_level_summary_csv": out_dir / f"{args.run_name}_seed_level_summary.csv",
        "direction_summary_csv": out_dir / f"{args.run_name}_direction_summary.csv",
        "metadata_json": out_dir / f"{args.run_name}_metadata.json",
    }
    if not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}. Use --overwrite.")

    inputs.to_csv(paths["input_manifest_csv"], index=False)
    checks.to_csv(paths["split_check_csv"], index=False)
    seed_level.to_csv(paths["seed_level_summary_csv"], index=False)
    directions.to_csv(paths["direction_summary_csv"], index=False)

    metadata = {
        "task": "ex30_decoding_seed_robustness_analysis",
        "run_name": args.run_name,
        "layer": int(args.layer),
        "alphas": [float(alpha) for alpha in args.alphas],
        "baseline_alpha": float(args.baseline_alpha),
        "sample_seed": int(args.sample_seed),
        "required_decoding_seeds": [int(seed) for seed in args.required_seeds],
        "input_count": int(inputs.shape[0]),
        "seed_level_row_count": int(seed_level.shape[0]),
        "direction_row_count": int(directions.shape[0]),
        "all_eval_splits_match_canonical": bool(checks["eval_split_matches_canonical"].all()),
        "all_directions_match": bool(directions["all_directions_match"].all()),
        "outputs": {name: str(path) for name, path in paths.items()},
        "notes": [
            "No model inference or classifier re-scoring is performed.",
            "Seed 12 rows come from the canonical Ex11/Ex12/Ex13 Llama summaries.",
            "Deltas are same-seed target-mechanism deltas against alpha=0.",
            "This analysis does not vary temperature or top-p.",
        ],
    }
    paths["metadata_json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    if any(np.isclose(float(alpha), float(args.baseline_alpha)) for alpha in args.alphas):
        raise ValueError("--alphas must contain non-baseline intervention alphas only")
    if len(set(int(seed) for seed in args.required_seeds)) != len(args.required_seeds):
        raise ValueError("--required-seeds contains duplicates")

    project = args.persona_project.resolve()
    out_dir = resolve_path(project, args.output_dir).resolve()
    alphas = [float(alpha) for alpha in args.alphas]

    inputs = discover_inputs(args, project)
    checks = split_checks(inputs)
    seed_level = seed_level_summary(inputs, int(args.layer), alphas, float(args.baseline_alpha))
    directions = direction_summary(seed_level, float(args.baseline_alpha))
    outputs = write_outputs(args, out_dir, inputs, checks, seed_level, directions)

    print(f"[done] Ex30 input runs: {inputs.shape[0]}")
    print(f"[done] Ex30 seed-level rows: {seed_level.shape[0]}")
    print(f"[done] Ex30 direction rows: {directions.shape[0]}")
    print(directions.to_string(index=False))
    for name, path in outputs.items():
        print(f"[output] {name}: {path}")


if __name__ == "__main__":
    main()
