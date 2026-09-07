"""
tests/conftest.py
====================
Shared pytest fixtures. Ensures the repo root is importable regardless of
where pytest is invoked from, and provides small, fast synthetic corpora
so tests don't depend on the real (larger, slower) data files existing
or being unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.loader import Corpus, Inscription


@pytest.fixture
def tiny_corpus() -> Corpus:
    """A small, hand-built corpus with clear structure, fast enough for
    every test that needs a real Corpus object but doesn't need real data."""
    inscriptions = [
        Inscription(inscription_id="T1", signs=["A", "B", "C"], site="SiteX",
                    object_type="seal", motif="MotifA", reading_direction="R-L"),
        Inscription(inscription_id="T2", signs=["A", "B", "D"], site="SiteX",
                    object_type="seal", motif="MotifA", reading_direction="R-L"),
        Inscription(inscription_id="T3", signs=["A", "B", "C"], site="SiteY",
                    object_type="tablet", motif="MotifB", reading_direction="R-L"),
        Inscription(inscription_id="T4", signs=["X", "Y"], site="SiteY",
                    object_type="tablet", motif="MotifB", reading_direction="R-L"),
        Inscription(inscription_id="T5", signs=["A", "B", "C", "D"], site="SiteX",
                    object_type="seal", motif="MotifA", reading_direction="R-L"),
    ]
    return Corpus(inscriptions)


@pytest.fixture
def larger_synthetic_corpus() -> Corpus:
    """Bigger synthetic corpus (from data/synthetic_corpus.py) for tests
    that need enough data for n-gram/entropy statistics to be non-degenerate."""
    from data.synthetic_corpus import generate_synthetic_corpus
    return generate_synthetic_corpus(n_inscriptions=300, seed=1)
