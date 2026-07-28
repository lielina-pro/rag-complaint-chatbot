"""Stratified sampling utilities for Task 2.

Builds a representative sample of the cleaned complaint dataset that
preserves the product-category proportions of the full dataset.
"""

from __future__ import annotations

import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    strat_col: str,
    target_n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample ``target_n`` rows from ``df``, preserving category proportions.

    If a category has fewer rows than its proportional share, all of that
    category's rows are kept (no upsampling), so the final count may be
    slightly under ``target_n``.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    strat_col : str
        Column to stratify by (e.g. ``"product_category"``).
    target_n : int
        Desired total sample size.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled rows, shuffled, with the original index preserved.
    """
    if strat_col not in df.columns:
        raise ValueError(
            f"'{strat_col}' not found in columns: {list(df.columns)}"
        )

    proportions: pd.Series = df[strat_col].value_counts(normalize=True)
    category_counts: pd.Series = df[strat_col].value_counts()
    targets: pd.Series = (proportions * target_n).round().astype(int)
    targets = targets.clip(upper=category_counts)

    parts: list[pd.DataFrame] = []
    for category, n in targets.items():
        if n <= 0:
            continue
        subset = df[df[strat_col] == category]
        parts.append(subset.sample(n=int(n), random_state=random_state))

    sample = pd.concat(parts, ignore_index=False)
    return sample.sample(frac=1, random_state=random_state)


def summarize_sampling(
    original_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    strat_col: str,
) -> pd.DataFrame:
    """Return a side-by-side table comparing category proportions.

    Parameters
    ----------
    original_df : pd.DataFrame
        The full cleaned dataset.
    sample_df : pd.DataFrame
        The stratified sample.
    strat_col : str
        The column used for stratification.

    Returns
    -------
    pd.DataFrame
        Columns: original_n, sample_n, original_pct, sample_pct.
    """
    orig_pct: pd.Series = original_df[strat_col].value_counts(normalize=True).rename("original_pct")
    sample_pct: pd.Series = sample_df[strat_col].value_counts(normalize=True).rename("sample_pct")
    orig_n: pd.Series = original_df[strat_col].value_counts().rename("original_n")
    sample_n: pd.Series = sample_df[strat_col].value_counts().rename("sample_n")

    summary = pd.concat([orig_n, sample_n, orig_pct, sample_pct], axis=1)
    summary["original_pct"] = (summary["original_pct"] * 100).round(2)
    summary["sample_pct"] = (summary["sample_pct"] * 100).round(2)
    return summary.sort_values("original_n", ascending=False)
