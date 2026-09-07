"""
tests/test_ngram.py
======================
Regression tests for analysis/ngram.py, anchored on the two real bugs
found in this module over the course of this project:

  1. KneserNeyModel's context-length off-by-one, whose symptom was
     IDENTICAL perplexity at every order from 1 to 6 -- every order was
     silently collapsing to the same base-case computation. The direct
     regression test for this is test_kneser_ney_perplexity_varies_by_order.
  2. bigram_markov_null (in data/permutation_nulls.py, tested here via
     its effect on ngram statistics) originally fixed each generated
     sequence's length externally rather than learning an END token,
     which distorted top_sign_final_share. Covered in test_null_models.py.
"""
import math
from analysis.ngram import (train_ngram, perplexity, cross_validated_perplexity,
                             train_kneser_ney, kn_perplexity, kn_cross_validated_perplexity,
                             bigram_counts, log_likelihood_ratio)


def test_bigram_counts_basic():
    sequences = [["A", "B", "C"], ["A", "B", "D"]]
    uni, bi = bigram_counts(sequences)
    assert uni["A"] == 2
    assert uni["B"] == 2
    assert bi[("A", "B")] == 2
    assert bi[("B", "C")] == 1
    assert bi[("B", "D")] == 1


def test_ngram_model_add_alpha_smoothing_never_zero(larger_synthetic_corpus):
    sequences = larger_synthetic_corpus.filter(exclude_damaged=True).sequences(normalized=True)
    model = train_ngram(sequences, n=2)
    vocab = sorted(set(s for seq in sequences for s in seq))
    # an unseen (context, word) pair should still get nonzero probability
    # under add-alpha smoothing, not zero
    p = model.prob(("__never_seen_context__",), vocab[0], alpha=0.5, vocab_size=len(vocab))
    assert p > 0


def test_kneser_ney_perplexity_varies_by_order(larger_synthetic_corpus):
    """Basic sanity check: training at different orders on the same data
    should not be a complete no-op. See
    test_kneser_ney_backoff_uses_correct_context_length below for the
    precise, deterministic regression test of the actual bug."""
    sequences = larger_synthetic_corpus.filter(exclude_damaged=True).sequences(normalized=True)
    perplexities = []
    for n in [1, 2, 3, 4]:
        ppl = kn_cross_validated_perplexity(sequences, n=n, k_folds=3)["mean_perplexity"]
        perplexities.append(ppl)
    distinct_values = len(set(round(p, 3) for p in perplexities))
    assert distinct_values > 1, f"Kneser-Ney gave identical perplexity at every order: {perplexities}"


def test_kneser_ney_backoff_uses_correct_context_length():
    """THE precise regression test for the context-length off-by-one bug.
    Earlier versions of KneserNeyModel silently used the wrong-length
    context slice specifically during recursive BACKOFF (when the exact
    higher-order context was never seen in training), which only shows
    up as a problem when backoff is actually exercised -- a corpus dense
    enough that most queries hit their exact context directly (as in
    test_kneser_ney_perplexity_varies_by_order's fixture) can mask this
    entirely, which is exactly why the bug wasn't caught by a looser test
    first. This test forces backoff deliberately: B is followed by C 100%
    of the time in training (a strong, learnable bigram signal), the
    specific trigram context (A, B) never occurs in training at all
    (forcing a backoff query), and W never follows B under any context.
    A correct model must still recover the strong bigram signal via
    backoff and clearly prefer C over W; the buggy version collapsed
    both to IDENTICAL probability (ratio exactly 1.0) because the wrong
    context length at the backoff step failed to match ANY trained
    counts, falling through to an undifferentiated base measure."""
    sequences = ([["X", "B", "C"]] * 30 + [["Y", "B", "C"]] * 30 + [["Z", "D", "W"]] * 5)
    model = train_kneser_ney(sequences, max_order=3)
    p_c = model.prob(("A", "B"), "C")  # (A,B) never seen; must back off to strong P(C|B)
    p_w = model.prob(("A", "B"), "W")  # W never follows B under any context
    assert p_c > p_w * 10, (
        f"P(C|A,B)={p_c:.4f} is not clearly greater than P(W|A,B)={p_w:.4f}. "
        f"A correctly implemented backoff should strongly prefer C (which reliably "
        f"follows B) over W (which never does). A ratio near 1.0 here is the exact "
        f"symptom of the context-length off-by-one bug."
    )


def test_kneser_ney_matches_ngram_perplexity_order_of_magnitude(larger_synthetic_corpus):
    """Sanity check, not a precise equivalence: Kneser-Ney and add-alpha
    perplexity at the same order should be in a broadly similar range for
    a well-behaved corpus; wildly different orders of magnitude would
    suggest one of the two implementations is doing something wrong."""
    sequences = larger_synthetic_corpus.filter(exclude_damaged=True).sequences(normalized=True)
    alpha_ppl = cross_validated_perplexity(sequences, n=1, k_folds=3)["mean_perplexity"]
    kn_ppl = kn_cross_validated_perplexity(sequences, n=1, k_folds=3)["mean_perplexity"]
    assert alpha_ppl > 0 and kn_ppl > 0
    ratio = max(alpha_ppl, kn_ppl) / min(alpha_ppl, kn_ppl)
    assert ratio < 10, f"add-alpha ppl={alpha_ppl:.1f} vs Kneser-Ney ppl={kn_ppl:.1f} at n=1: too far apart"


def test_kneser_ney_probabilities_are_valid():
    sequences = [["A", "B", "C"], ["A", "B", "D"], ["A", "C", "B"]]
    model = train_kneser_ney(sequences, max_order=2)
    p = model.prob(("A",), "B")
    assert 0.0 <= p <= 1.0


def test_log_likelihood_ratio_finds_significant_pairs():
    # A -> B always co-occurs; C is independent noise
    sequences = [["A", "B"]] * 20 + [["C", "A"]] * 2 + [["B", "C"]] * 2
    results = log_likelihood_ratio(sequences, min_count=1)
    top_pair = (results[0]["sign_a"], results[0]["sign_b"])
    assert top_pair == ("A", "B")
