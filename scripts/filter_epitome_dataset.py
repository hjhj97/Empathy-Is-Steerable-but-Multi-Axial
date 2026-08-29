"""
Filter EPITOME dataset samples by:
1. Remove seeker posts shorter than 100 characters
2. Remove posts containing suicide/self-harm related keywords
"""

import pandas as pd
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "dataset"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "filtered"

SUICIDE_KEYWORDS = [
    "suicid",
    "kill myself",
    "end my life",
    "want to die",
    "wanna die",
    "self-harm",
    "self harm",
    "cutting myself",
    "overdose",
    "hang myself",
    "shoot myself",
    "jump off",
    "slit my",
    "pills to die",
    "end it all",
    "not be alive",
    "take my life",
    "no reason to live",
    "better off dead",
    "wish i was dead",
    "wish i were dead",
]

FILENAMES = [
    "emotional-reactions-reddit.csv",
    "explorations-reddit.csv",
    "interpretations-reddit.csv",
]

MIN_POST_LENGTH = 100


def get_valid_sp_ids(df: pd.DataFrame) -> set:
    """Return sp_ids that pass all filters."""
    too_short = df["seeker_post"].str.len() < MIN_POST_LENGTH

    pattern = "|".join(SUICIDE_KEYWORDS)
    has_sensitive = df["seeker_post"].str.lower().str.contains(pattern, na=False)

    valid_mask = ~too_short & ~has_sensitive

    n_too_short = too_short.sum()
    n_sensitive = has_sensitive.sum()
    n_both = (too_short & has_sensitive).sum()
    n_total = len(df)
    n_valid = valid_mask.sum()

    print(f"  Total samples       : {n_total}")
    print(f"  Too short (< 100)   : {n_too_short}")
    print(f"  Sensitive content   : {n_sensitive}")
    print(f"  Both (overlap)      : {n_both}")
    print(f"  Kept after filter   : {n_valid} ({n_valid/n_total*100:.1f}%)")

    return set(df.loc[valid_mask, "sp_id"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the first file to determine valid sp_ids (all files share the same seeker_post)
    ref_df = pd.read_csv(DATASET_DIR / FILENAMES[0])
    print(f"Computing valid sp_ids from: {FILENAMES[0]}")
    valid_ids = get_valid_sp_ids(ref_df)
    print()

    for fname in FILENAMES:
        df = pd.read_csv(DATASET_DIR / fname)
        filtered = df[df["sp_id"].isin(valid_ids)].reset_index(drop=True)
        out_name = fname.replace("-reddit.csv", "-reddit-filtered.csv")
        out_path = OUTPUT_DIR / out_name
        filtered.to_csv(out_path, index=False)
        print(f"{fname}: {len(df)} -> {len(filtered)} rows saved to {out_path}")


if __name__ == "__main__":
    main()
