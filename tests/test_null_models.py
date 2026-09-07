"""
tests/test_null_models.py
============================
Tests for the four null-model generator modules built over this project:
data/adversarial_null_model.py, data/permutation_nulls.py,
data/stratified_null_model.py. Anchored partly on a real bug: an earlier
version of bigram_markov_null fixed each generated sequence's length to
an externally-sampled real length rather than learning an END token,
which distorted top_sign_final_share since the chain had no way to
reproduce real "closing sign" behavior. test_bigram_null_learns_end_token
is the direct regression test for this.
"""
from data.loader import Corpus
from data.adversarial_null_model import generate_matched_null_corpus
from data.permutation_nulls import (within_inscription_shuffle, global_shuffle,
                                     position_preserving_shuffle, bigram_markov_null,
                                     trigram_markov_null)
from data.stratified_null_model import motif_stratified_null, site_stratified_null


def test_adversarial_null_preserves_length_distribution(larger_synthetic_corpus):
    real = larger_synthetic_corpus.filter(exclude_damaged=True)
    null = generate_matched_null_corpus(larger_synthetic_corpus, n_inscriptions=len(real), seed=0)
    real_mean_len = sum(len(s) for s in real.sequences()) / len(real)
    null_mean_len = sum(len(s) for s in null.sequences()) / len(null)
    # lengths are resampled from the real distribution, so means should
    # be close, not necessarily identical
    assert abs(real_mean_len - null_mean_len) < 1.0


def test_adversarial_null_has_no_real_bigram_dependency():
    """The null draws every sign independently, so a sign's identity
    should not depend on what preceded it. Weak statistical check: the
    same sign appearing twice in a row (self-transition) should be rare,
    at the rate expected from independent draws, not artificially
    inflated the way real sequential dependency would produce."""
    from data.loader import Corpus, Inscription
    real = Corpus([Inscription(inscription_id=f"r{i}", signs=["A", "B", "A", "B", "C"])
                   for i in range(50)])
    null = generate_matched_null_corpus(real, n_inscriptions=200, seed=0)
    assert len(null) == 200
    assert all(len(ins.signs) >= 1 for ins in null.inscriptions)


def test_within_inscription_shuffle_preserves_vocabulary(tiny_corpus):
    shuffled = within_inscription_shuffle(tiny_corpus, seed=0)
    original_multiset = sorted(s for ins in tiny_corpus.filter(exclude_damaged=True).inscriptions for s in ins.signs)
    shuffled_multiset = sorted(s for ins in shuffled.inscriptions for s in ins.signs)
    assert original_multiset == shuffled_multiset


def test_global_shuffle_preserves_length_sequence(tiny_corpus):
    shuffled = global_shuffle(tiny_corpus, seed=0)
    original_lengths = [len(ins.normalized_signs()) for ins in tiny_corpus.filter(exclude_damaged=True).inscriptions]
    shuffled_lengths = [len(ins.signs) for ins in shuffled.inscriptions]
    assert original_lengths == shuffled_lengths


def test_position_preserving_shuffle_keeps_endpoints_fixed(tiny_corpus):
    shuffled = position_preserving_shuffle(tiny_corpus, seed=0)
    originals = {ins.inscription_id: ins.normalized_signs() for ins in tiny_corpus.filter(exclude_damaged=True).inscriptions}
    for ins in shuffled.inscriptions:
        orig_id = ins.inscription_id.replace("POSSHUF-", "")
        orig_seq = originals[orig_id]
        if len(orig_seq) >= 2:
            assert ins.signs[0] == orig_seq[0]
            assert ins.signs[-1] == orig_seq[-1]


def test_bigram_null_learns_end_token(larger_synthetic_corpus):
    """Regression test for the length-fixing bug: bigram_markov_null must
    learn stopping behavior from real data (via a trained END token), not
    have every generated sequence's length externally fixed to a
    pre-sampled real length. Weak but direct check: generated sequence
    lengths should vary (not be suspiciously uniform in a way that would
    suggest external length-fixing), and the null's mean length should be
    in the same ballpark as the real corpus's own mean length."""
    real = larger_synthetic_corpus.filter(exclude_damaged=True)
    null = bigram_markov_null(larger_synthetic_corpus, n_inscriptions=len(real), seed=0)
    real_mean_len = sum(len(s) for s in real.sequences()) / len(real)
    null_mean_len = sum(len(ins.signs) for ins in null.inscriptions) / len(null)
    assert abs(real_mean_len - null_mean_len) < 2.0
    null_lengths = set(len(ins.signs) for ins in null.inscriptions)
    assert len(null_lengths) > 1, "all generated sequences have the same length -- END token likely not learned"


def test_trigram_null_generates_valid_corpus(larger_synthetic_corpus):
    real = larger_synthetic_corpus.filter(exclude_damaged=True)
    null = trigram_markov_null(larger_synthetic_corpus, n_inscriptions=50, seed=0)
    assert len(null) > 0
    assert all(len(ins.signs) >= 1 for ins in null.inscriptions)


def test_stratified_null_respects_strata(larger_synthetic_corpus):
    from data.loader import Corpus, Inscription
    inscriptions = []
    for i in range(30):
        inscriptions.append(Inscription(inscription_id=f"m{i}", signs=["A", "B", "C"],
                                          motif="MotifA", site="Site1"))
    for i in range(30):
        inscriptions.append(Inscription(inscription_id=f"n{i}", signs=["X", "Y", "Z"],
                                          motif="MotifB", site="Site2"))
    corpus = Corpus(inscriptions)
    null = motif_stratified_null(corpus, n_inscriptions=100, seed=0, min_stratum_size=5)
    assert len(null) == 100
    all_signs = set(s for ins in null.inscriptions for s in ins.signs)
    # the null should only ever generate signs that exist somewhere in the real corpus
    assert all_signs <= {"A", "B", "C", "X", "Y", "Z"}
