import pandas as pd

from src.sampling import stratified_sample, summarize_sampling


def _make_df():
    # 800 rows: 400 A, 300 B, 100 C  -> proportions 0.5 / 0.375 / 0.125
    categories = ["A"] * 400 + ["B"] * 300 + ["C"] * 100
    return pd.DataFrame({"category": categories, "value": range(800)})


def test_stratified_sample_preserves_proportions():
    df = _make_df()
    sample = stratified_sample(df, strat_col="category", target_n=80, random_state=0)

    counts = sample["category"].value_counts(normalize=True)
    assert abs(counts["A"] - 0.5) < 0.05
    assert abs(counts["B"] - 0.375) < 0.05
    assert abs(counts["C"] - 0.125) < 0.05


def test_stratified_sample_total_size_close_to_target():
    df = _make_df()
    target = 80
    sample = stratified_sample(df, strat_col="category", target_n=target, random_state=0)
    assert abs(len(sample) - target) <= 2


def test_stratified_sample_caps_at_available_rows():
    df = _make_df()
    # Ask for more than exists; should not error, should not exceed totals.
    sample = stratified_sample(df, strat_col="category", target_n=10_000, random_state=0)
    assert len(sample) <= len(df)
    assert (sample["category"].value_counts() <= df["category"].value_counts()).all()


def test_stratified_sample_no_duplicate_rows():
    df = _make_df()
    sample = stratified_sample(df, strat_col="category", target_n=80, random_state=0)
    assert sample.index.is_unique


def test_summarize_sampling_returns_expected_columns():
    df = _make_df()
    sample = stratified_sample(df, strat_col="category", target_n=80, random_state=0)
    summary = summarize_sampling(df, sample, "category")
    assert set(summary.columns) == {"original_n", "sample_n", "original_pct", "sample_pct"}
    assert len(summary) == 3
