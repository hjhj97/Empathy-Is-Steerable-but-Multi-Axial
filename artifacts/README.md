# Aggregate Results

These files contain aggregate results used to audit the paper's reported
values. They contain no source dialogue, generated response, per-example
record, activation tensor, model checkpoint, API key, or local absolute path.

Regenerate them from local experiment outputs with:

```bash
python scripts/export_release_artifacts.py
```

`wikitext103_ppl_4096_tokens.csv` reports the paper's language-fluency check on
the first 4,096 tokens of the WikiText-103 test split. See
`REPRODUCIBILITY.md` for the corrected hook behavior and reproduction command.
