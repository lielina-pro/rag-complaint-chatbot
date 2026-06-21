"""Sanity-check test so the CI pipeline has something to run from day one.
Replace/extend with real tests as src/ modules are implemented (e.g. chunking,
retriever, prompt building).
"""


def test_sanity():
    assert 1 + 1 == 2
