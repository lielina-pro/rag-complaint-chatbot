"""Stratified sampling utilities for Task 2.

Building embeddings for the full ~464K-complaint dataset takes several hours
on standard hardware, so Task 2 works on a stratified sample of ~10K-15K
complaints that preserves the product-category proportions of the cleaned
dataset produced in Task 1.
"""

from __future__ import annotations

import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    strat_col: str,
    target_n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample ``target_n`` rows from ``df``, preserving the proportion of each
    category in ``strat_col`` as closely as possible.

    If a category has fewer rows available than its proportional share, all
    of that category's rows are kept (no replacement/upsampling), and the
    final sample size may be slightly under ``target_n`` as a result.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe (e.g. the Task 1 cleaned/filtered dataset).
    strat_col : str
        Column to stratify by (e.g. ``"product_category"``).
    target_n : int
        Desired total sample size.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The sampled rows, shuffled, with the original index preserved as a
        column is not added (caller can ``reset_index`` if desired).
    """
    if strat_col not in df.columns:
        raise ValueError(f"'{strat_col}' not found in dataframe columns: {list(df.columns)}")

    proportions = df[strat_col].value_counts(normalize=True)
    category_counts = df[strat_col].value_counts()

    # Proportional target per category, rounded.
    targets = (proportions * target_n).round().astype(int)

    # Cap each category's target at the number of rows actually available.
    targets = targets.clip(upper=category_counts)

    sampled_parts = []
    for category, n in targets.items():
        if n <= 0:
            continue
        subset = df[df[strat_col] == category]
        sampled_parts.append(subset.sample(n=int(n), random_state=random_state))

    sample = pd.concat(sampled_parts, ignore_index=False)
    sample = sample.sample(frac=1, random_state=random_state)  # shuffle
    return sample


def summarize_sampling(
    original_df: pd.DataFrame, sample_df: pd.DataFrame, strat_col: str
) -> pd.DataFrame:
    """Build a side-by-side table comparing category proportions in the
    original dataset vs. the sample, for inclusion in the report."""
    orig_pct = original_df[strat_col].value_counts(normalize=True).rename("original_pct")
    sample_pct = sample_df[strat_col].value_counts(normalize=True).rename("sample_pct")
    orig_n = original_df[strat_col].value_counts().rename("original_n")
    sample_n = sample_df[strat_col].value_counts().rename("sample_n")

    summary = pd.concat([orig_n, sample_n, orig_pct, sample_pct], axis=1)
    summary["original_pct"] = (summary["original_pct"] * 100).round(2)
    summary["sample_pct"] = (summary["sample_pct"] * 100).round(2)
    return summary.sort_values("original_n", ascending=False)
