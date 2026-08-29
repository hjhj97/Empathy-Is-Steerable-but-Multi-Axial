#!/usr/bin/env python3
"""Prepare Ex34 exact-stratum controlled vector pools.

For each EPITOME target, positive and negative response rows are matched on
the exact 0/1/2 levels of the other two labels after excluding the original
evaluation seekers. The output contains vector rows only; evaluation seekers
are supplied separately to the steering runners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("ER", "EX", "IP")
FILTERED_FILES = {
    "ER": "emotional-reactions-reddit-filtered.csv",
    "EX": "explorations-reddit-filtered.csv",
    "IP": "interpretations-reddit-filtered.csv",
}
EXPERIMENT_NUMBERS = {"ER": 11, "EX": 12, "IP": 13}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare Ex34 exact-stratum controlled pools")
    parser.add_argument(
        "--joint-labels-path",
        type=Path,
        default=Path("outputs/rebuttal_priority1/joint_labels.csv"),
    )
    parser.add_argument("--filtered-dir", type=Path, default=Path("dataset/filtered"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rebuttal_priority1_controlled_inputs_v2"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def eval_path(target: str) -> Path:
    number = EXPERIMENT_NUMBERS[target]
    lower = target.lower()
    return PROJECT_ROOT / (
        f"outputs/ex{number}_filtered_{lower}/"
        f"ex{number}_filtered_{lower}_llama31_n200_eval_seekers.csv"
    )


def largest_remainder_allocation(capacities: Dict[str, int], total: int) -> Dict[str, int]:
    capacity_total = int(sum(capacities.values()))
    if capacity_total < total:
        raise RuntimeError(f"Matched capacity {capacity_total} is smaller than requested {total}")

    ideals = {key: total * value / capacity_total for key, value in capacities.items()}
    allocation = {key: int(np.floor(value)) for key, value in ideals.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        capacities,
        key=lambda key: (-(ideals[key] - allocation[key]), key),
    )
    for key in order:
        if remaining == 0:
            break
        if allocation[key] < capacities[key]:
            allocation[key] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError(f"Could not allocate {total} matched rows; remaining={remaining}")
    return allocation


def remove_ambiguous_keys(
    frame: pd.DataFrame,
    value_columns: List[str],
    frame_name: str,
) -> Tuple[pd.DataFrame, int]:
    """Collapse exact duplicates and drop identifiers with conflicting values."""
    key_columns = ["sp_id", "rp_id"]
    variants = frame.groupby(key_columns, dropna=False)[value_columns].nunique(dropna=False)
    ambiguous_index = variants.index[variants.max(axis=1) > 1]
    keyed = pd.MultiIndex.from_frame(frame[key_columns])
    ambiguous_mask = keyed.isin(ambiguous_index)
    ambiguous_count = int(len(ambiguous_index))
    cleaned = frame.loc[~ambiguous_mask].drop_duplicates(key_columns, keep="first").copy()
    if cleaned.duplicated(key_columns).any():
        raise AssertionError(f"{frame_name}: duplicate keys remain after cleaning")
    print(
        f"[clean] {frame_name}: rows={len(frame)} -> {len(cleaned)}, "
        f"ambiguous_keys_dropped={ambiguous_count}",
        flush=True,
    )
    return cleaned, ambiguous_count


def load_candidates(
    target: str,
    joint: pd.DataFrame,
    filtered_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    eval_file = eval_path(target)
    if not eval_file.exists():
        raise FileNotFoundError(f"Missing eval seekers for {target}: {eval_file}")
    eval_seekers = pd.read_csv(eval_file, dtype={"source_id": str})
    required_eval = {"source_id", "seeker_post"}
    if not required_eval.issubset(eval_seekers.columns):
        raise ValueError(f"{eval_file} must contain {sorted(required_eval)}")
    if len(eval_seekers) != 200 or eval_seekers["source_id"].duplicated().any():
        raise ValueError(f"Expected 200 unique eval seekers in {eval_file}")

    target_file = filtered_dir / FILTERED_FILES[target]
    source = pd.read_csv(target_file, dtype={"sp_id": str, "rp_id": str})
    required_source = {"sp_id", "rp_id", "seeker_post", "response_post", "level"}
    if not required_source.issubset(source.columns):
        raise ValueError(f"{target_file} missing columns {sorted(required_source - set(source.columns))}")

    source, ambiguous_source_keys = remove_ambiguous_keys(
        source,
        ["seeker_post", "response_post", "level"],
        f"{target} filtered source",
    )
    label_columns = ["y_ER", "y_EX", "y_IP"]
    candidates = joint.merge(
        source[["sp_id", "rp_id", "seeker_post", "response_post", "level"]],
        on=["sp_id", "rp_id"],
        how="inner",
        validate="one_to_one",
    )
    candidates = candidates[~candidates["sp_id"].isin(set(eval_seekers["source_id"]))].copy()
    candidates[label_columns] = candidates[label_columns].astype(int)
    candidates["level"] = candidates["level"].astype(int)
    if not (candidates["level"] == candidates[f"y_{target}"]).all():
        raise ValueError(f"Target label mismatch after joining {target_file}")

    other_targets = [name for name in TARGETS if name != target]
    candidates["stratum"] = (
        candidates[f"y_{other_targets[0]}"].astype(str)
        + "|"
        + candidates[f"y_{other_targets[1]}"].astype(str)
    )
    candidates.attrs["ambiguous_source_keys_dropped"] = ambiguous_source_keys
    return candidates, eval_seekers, other_targets


def sample_target_pool(
    target: str,
    candidates: pd.DataFrame,
    eval_seekers: pd.DataFrame,
    other_targets: List[str],
    sample_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    positive = candidates[candidates[f"y_{target}"] >= 1].copy()
    negative = candidates[candidates[f"y_{target}"] == 0].copy()
    strata = sorted(set(positive["stratum"]) & set(negative["stratum"]))
    capacities = {
        stratum: min(
            int((positive["stratum"] == stratum).sum()),
            int((negative["stratum"] == stratum).sum()),
        )
        for stratum in strata
    }
    capacities = {key: value for key, value in capacities.items() if value > 0}
    allocation = largest_remainder_allocation(capacities, sample_size)

    rng = np.random.default_rng(seed)
    records: List[dict] = []
    pair_index = 0
    for stratum in sorted(allocation):
        count = allocation[stratum]
        if count == 0:
            continue
        pos_group = positive[positive["stratum"] == stratum]
        neg_group = negative[negative["stratum"] == stratum]
        pos_indices = rng.choice(pos_group.index.to_numpy(), size=count, replace=False)
        neg_indices = rng.choice(neg_group.index.to_numpy(), size=count, replace=False)
        for pos_index, neg_index in zip(pos_indices, neg_indices):
            for condition, row_index in (("pos", pos_index), ("neg", neg_index)):
                row = candidates.loc[row_index]
                records.append({
                    "sp_id": str(row["sp_id"]),
                    "rp_id": str(row["rp_id"]),
                    "seeker_post": str(row["seeker_post"]),
                    "response_post": str(row["response_post"]),
                    "level": int(row[f"y_{target}"]),
                    "split_role": "controlled_vector",
                    "target": target,
                    "condition": condition,
                    "ER": int(row["y_ER"]),
                    "IP": int(row["y_IP"]),
                    "EX": int(row["y_EX"]),
                    "matched_pair_id": pair_index,
                    "stratum": stratum,
                })
            pair_index += 1

    output = pd.DataFrame(records)
    expected_columns = [
        "sp_id", "rp_id", "seeker_post", "response_post", "level", "split_role",
        "target", "condition", "ER", "IP", "EX", "matched_pair_id", "stratum",
    ]
    output = output[expected_columns]

    counts = output["condition"].value_counts().to_dict()
    if counts != {"pos": sample_size, "neg": sample_size}:
        raise AssertionError(f"{target}: unexpected condition counts {counts}")
    hist = pd.crosstab(output["stratum"], output["condition"])
    if not (hist["pos"] == hist["neg"]).all():
        raise AssertionError(f"{target}: stratum histograms are not exactly matched\n{hist}")
    if output.duplicated(["sp_id", "rp_id"]).any():
        raise AssertionError(f"{target}: duplicate (sp_id, rp_id) rows in vector pool")
    overlap = set(output["sp_id"]) & set(eval_seekers["source_id"].astype(str))
    if overlap:
        raise AssertionError(f"{target}: vector/eval seeker overlap: {sorted(overlap)[:5]}")

    metadata = {
        "target": target,
        "sample_size_per_class": sample_size,
        "sample_seed": seed,
        "positive_rule": f"{target}>=1",
        "negative_rule": f"{target}=0",
        "non_target_labels": other_targets,
        "matched_capacity": int(sum(capacities.values())),
        "stratum_capacities": capacities,
        "stratum_allocation": allocation,
        "condition_counts": counts,
        "unique_vector_seekers": int(output["sp_id"].nunique()),
        "unique_eval_seekers": int(eval_seekers["source_id"].nunique()),
        "vector_eval_overlap": 0,
        "ambiguous_source_keys_dropped": int(
            candidates.attrs.get("ambiguous_source_keys_dropped", 0)
        ),
    }
    return output, metadata


def sample_unmatched_comparator(
    target: str,
    candidates: pd.DataFrame,
    eval_seekers: pd.DataFrame,
    reference_pool: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    """Sample an unmatched arm from the same support and target-level mix."""
    rng = np.random.default_rng(seed)
    reference_positive = reference_pool[reference_pool["condition"] == "pos"]
    target_level_counts = {
        int(level): int(count)
        for level, count in reference_positive["level"].value_counts().sort_index().items()
    }

    selected_parts: List[pd.DataFrame] = []
    for level, count in target_level_counts.items():
        pool = candidates[candidates[f"y_{target}"] == level]
        if len(pool) < count:
            raise RuntimeError(
                f"{target}: need {count} unmatched positive rows at level {level}, got {len(pool)}"
            )
        indices = rng.choice(pool.index.to_numpy(), size=count, replace=False)
        part = candidates.loc[indices].copy()
        part["condition"] = "pos"
        selected_parts.append(part)

    negative = candidates[candidates[f"y_{target}"] == 0]
    negative_count = int((reference_pool["condition"] == "neg").sum())
    if len(negative) < negative_count:
        raise RuntimeError(f"{target}: need {negative_count} unmatched negative rows, got {len(negative)}")
    negative_indices = rng.choice(negative.index.to_numpy(), size=negative_count, replace=False)
    negative_part = candidates.loc[negative_indices].copy()
    negative_part["condition"] = "neg"
    selected_parts.append(negative_part)

    selected = pd.concat(selected_parts, ignore_index=True)
    records: List[dict] = []
    for row in selected.itertuples(index=False):
        records.append({
            "sp_id": str(row.sp_id),
            "rp_id": str(row.rp_id),
            "seeker_post": str(row.seeker_post),
            "response_post": str(row.response_post),
            "level": int(getattr(row, f"y_{target}")),
            "split_role": "unmatched_vector",
            "target": target,
            "condition": str(row.condition),
            "ER": int(row.y_ER),
            "IP": int(row.y_IP),
            "EX": int(row.y_EX),
            "matched_pair_id": -1,
            "stratum": str(row.stratum),
        })
    output = pd.DataFrame(records)
    expected_columns = [
        "sp_id", "rp_id", "seeker_post", "response_post", "level", "split_role",
        "target", "condition", "ER", "IP", "EX", "matched_pair_id", "stratum",
    ]
    output = output[expected_columns]

    expected_counts = reference_pool["condition"].value_counts().to_dict()
    actual_counts = output["condition"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise AssertionError(f"{target}: unmatched condition counts {actual_counts} != {expected_counts}")
    actual_target_levels = {
        int(level): int(count)
        for level, count in output[output["condition"] == "pos"]["level"]
        .value_counts().sort_index().items()
    }
    if actual_target_levels != target_level_counts:
        raise AssertionError(
            f"{target}: unmatched target levels {actual_target_levels} != {target_level_counts}"
        )
    if output.duplicated(["sp_id", "rp_id"]).any():
        raise AssertionError(f"{target}: duplicate (sp_id, rp_id) rows in unmatched pool")
    overlap = set(output["sp_id"]) & set(eval_seekers["source_id"].astype(str))
    if overlap:
        raise AssertionError(f"{target}: unmatched vector/eval overlap: {sorted(overlap)[:5]}")

    stratum_hist = pd.crosstab(output["stratum"], output["condition"])
    metadata = {
        "target": target,
        "arm": "same_support_unmatched",
        "sample_seed": seed,
        "target_level_counts": target_level_counts,
        "condition_counts": actual_counts,
        "stratum_histogram": {
            stratum: {condition: int(value) for condition, value in row.items()}
            for stratum, row in stratum_hist.to_dict(orient="index").items()
        },
        "unique_vector_seekers": int(output["sp_id"].nunique()),
        "unique_eval_seekers": int(eval_seekers["source_id"].nunique()),
        "vector_eval_overlap": 0,
    }
    return output, metadata


def main() -> None:
    args = parse_args()
    joint_path = resolve(args.joint_labels_path)
    filtered_dir = resolve(args.filtered_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joint = pd.read_csv(joint_path, dtype={"sp_id": str, "rp_id": str})
    required_joint = {"sp_id", "rp_id", "y_ER", "y_EX", "y_IP"}
    if not required_joint.issubset(joint.columns):
        raise ValueError(f"{joint_path} missing columns {sorted(required_joint - set(joint.columns))}")
    joint, ambiguous_joint_keys = remove_ambiguous_keys(
        joint,
        ["y_ER", "y_EX", "y_IP"],
        "joint labels",
    )

    summary_rows: List[dict] = []
    for target in TARGETS:
        candidates, eval_seekers, others = load_candidates(target, joint, filtered_dir)
        output, metadata = sample_target_pool(
            target=target,
            candidates=candidates,
            eval_seekers=eval_seekers,
            other_targets=others,
            sample_size=args.sample_size,
            seed=args.sample_seed,
        )
        lower = target.lower()
        stem = f"rebuttal_p1_filtered_{lower}_ge1_exact_evalexcl_compat"
        csv_path = output_dir / f"{stem}.csv"
        meta_path = output_dir / f"{stem}_metadata.json"
        output.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append({
            "target": target,
            "arm": "exact_stratum",
            "rows": len(output),
            "matched_capacity": metadata["matched_capacity"],
            "unique_vector_seekers": metadata["unique_vector_seekers"],
            "vector_eval_overlap": metadata["vector_eval_overlap"],
            "ambiguous_joint_keys_dropped": ambiguous_joint_keys,
            "ambiguous_source_keys_dropped": metadata["ambiguous_source_keys_dropped"],
            "csv_path": str(csv_path),
        })
        print(
            f"[done] {target}: rows={len(output)} capacity={metadata['matched_capacity']} "
            f"unique_seekers={metadata['unique_vector_seekers']} path={csv_path}",
            flush=True,
        )

        unmatched, unmatched_metadata = sample_unmatched_comparator(
            target=target,
            candidates=candidates,
            eval_seekers=eval_seekers,
            reference_pool=output,
            seed=args.sample_seed,
        )
        unmatched_stem = f"rebuttal_p1_filtered_{lower}_ge1_unmatched_evalexcl_compat"
        unmatched_csv = output_dir / f"{unmatched_stem}.csv"
        unmatched_meta = output_dir / f"{unmatched_stem}_metadata.json"
        unmatched.to_csv(unmatched_csv, index=False)
        unmatched_meta.write_text(
            json.dumps(unmatched_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_rows.append({
            "target": target,
            "arm": "same_support_unmatched",
            "rows": len(unmatched),
            "matched_capacity": np.nan,
            "unique_vector_seekers": unmatched_metadata["unique_vector_seekers"],
            "vector_eval_overlap": unmatched_metadata["vector_eval_overlap"],
            "ambiguous_joint_keys_dropped": ambiguous_joint_keys,
            "ambiguous_source_keys_dropped": metadata["ambiguous_source_keys_dropped"],
            "csv_path": str(unmatched_csv),
        })
        print(
            f"[done] {target} unmatched: rows={len(unmatched)} "
            f"unique_seekers={unmatched_metadata['unique_vector_seekers']} path={unmatched_csv}",
            flush=True,
        )

    summary_path = output_dir / "ex34_exact_stratum_pool_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
