#!/usr/bin/env python3
"""Build the ER/IP/EX joint-label table used by the Ex34 pool control."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "ER": "emotional-reactions-reddit-filtered.csv",
    "EX": "explorations-reddit-filtered.csv",
    "IP": "interpretations-reddit-filtered.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build Ex34 joint EPITOME labels")
    parser.add_argument(
        "--filtered-dir", type=Path, default=Path("dataset/filtered")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/rebuttal_priority1/joint_labels.csv"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_labels(filtered_dir: Path, mechanism: str) -> pd.DataFrame:
    path = filtered_dir / FILES[mechanism]
    frame = pd.read_csv(path, dtype={"sp_id": str, "rp_id": str})
    required = {"sp_id", "rp_id", "level"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} missing columns {sorted(required - set(frame.columns))}")
    labels = frame[["sp_id", "rp_id", "level"]].copy()
    labels["level"] = pd.to_numeric(labels["level"], errors="raise").astype(int)
    return labels.rename(columns={"level": f"y_{mechanism}"})


def main() -> None:
    args = parse_args()
    filtered_dir = resolve(args.filtered_dir)
    output_path = resolve(args.output_path)

    joint = load_labels(filtered_dir, "ER")
    for mechanism in ("EX", "IP"):
        joint = joint.merge(
            load_labels(filtered_dir, mechanism),
            on=["sp_id", "rp_id"],
            how="inner",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joint.to_csv(output_path, index=False)
    print(f"Saved {len(joint)} joint-label rows to {output_path}")


if __name__ == "__main__":
    main()
