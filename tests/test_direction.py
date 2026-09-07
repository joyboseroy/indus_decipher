"""
tests/test_direction.py
==========================
Tests for analysis/direction_test.py, the diagnostic that caught the
real reading-direction bug in both real corpora's original converters.
Builds a small synthetic corpus with a KNOWN correct direction (final
position deliberately made more constrained than initial, mirroring the
M77 fingerprint) and checks the diagnostic correctly identifies it.
"""
from data.loader import Corpus, Inscription
from analysis.direction_test import test_reading_direction as run_direction_test


def _make_corpus_with_known_direction(n=200):
    """Builds a corpus where, in the CORRECT reading order, initial signs
    are highly varied (many distinct choices) and final signs are highly
    constrained (almost always one specific sign) -- the M77 fingerprint
    this diagnostic is designed to detect."""
    import random
    rng = random.Random(0)
    initial_choices = [f"INIT{i}" for i in range(20)]  # many options: high entropy
    final_sign = "END_SIGN"  # almost always the same: low entropy

    inscriptions = []
    for i in range(n):
        middle = [rng.choice(["M1", "M2", "M3"]) for _ in range(2)]
        # correct reading order: [varied initial, ..., constrained final]
        correct_order = [rng.choice(initial_choices)] + middle + [final_sign]
        inscriptions.append(Inscription(
            inscription_id=f"c{i}", signs=correct_order, reading_direction="R-L"))
    return Corpus(inscriptions)


def test_direction_diagnostic_identifies_correct_orientation():
    corpus = _make_corpus_with_known_direction()
    result = run_direction_test(corpus)
    assert result.likely_direction == "as-stored", (
        f"Diagnostic failed to identify the correct orientation on a corpus "
        f"built with a known, deliberate direction fingerprint. Gap as-stored: "
        f"{result.as_stored['final_minus_initial_gap']:.3f}, "
        f"gap reversed: {result.reversed_['final_minus_initial_gap']:.3f}"
    )


def test_direction_diagnostic_flips_when_corpus_is_reversed():
    """If the SAME corpus is stored backwards, the diagnostic should
    now say 'reversed' is correct, not 'as-stored'."""
    correct_corpus = _make_corpus_with_known_direction()
    backwards_inscriptions = [
        Inscription(inscription_id=ins.inscription_id, signs=list(reversed(ins.signs)),
                    reading_direction="R-L")
        for ins in correct_corpus.inscriptions
    ]
    backwards_corpus = Corpus(backwards_inscriptions)
    result = run_direction_test(backwards_corpus)
    assert result.likely_direction == "reversed"
