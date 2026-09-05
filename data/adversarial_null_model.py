"""
data/adversarial_null_model.py
=================================
The hardest test the falsification harness can be given: instead of
comparing the real corpus to a synthetic system built from scratch with
its own vocabulary and rules (as data/synthetic_civilizations.py does),
build a non-linguistic generator that reuses the REAL corpus's own
surface statistics directly, then check whether the classifier can still
tell the two apart.

`generate_matched_null_corpus()` preserves, by construction, three
things drawn straight from the real data:
  - the empirical length distribution (resampled with replacement)
  - the empirical INITIAL-position sign distribution
  - the empirical FINAL-position sign distribution
  - the empirical overall (unigram) sign distribution, used for every
    non-edge position

Each synthetic inscription is built by drawing its length, then drawing
its initial sign, final sign, and every middle sign INDEPENDENTLY from
these marginals. There is no bigram or higher-order dependency at all:
whatever sign comes before a given position has no influence on what
comes next. That is the deliberate difference from the real corpus,
which (per Rao/Yadav's own entropic argument) shows real sequential
structure beyond independent draws.

Consequence, worth stating plainly before looking at any result: three of
the six falsification-harness features (`zipf_gamma`, `zipf_r2`,
`top_sign_final_share`) and `mean_length` are matched almost exactly BY
CONSTRUCTION, since they are direct functions of the marginals this
generator reuses. `conditional_entropy` and `perplexity_ratio_n2_n1` are
NOT matched by construction, since they are the two features that
actually depend on sequential order rather than position-marginal
identity. So this is really a targeted test of those two features: if
the real corpus's conditional dependency is doing real classification
work, the matched null (having none) should be distinguishable from the
real corpus using those two features even though the other four look
identical. If the classifier still can't tell them apart, that is a
genuinely important negative result about how much this six-feature
classifier is actually detecting.
"""
from __future__ import annotations
import random
from collections import Counter

from data.loader import Corpus, Inscription


def generate_matched_null_corpus(real_corpus: Corpus, n_inscriptions: int,
                                   seed: int = 0, id_prefix: str = "NULL") -> Corpus:
    filtered = real_corpus.filter(exclude_damaged=True)
    sequences = filtered.sequences(normalized=True)
    sequences = [s for s in sequences if len(s) >= 1]
    if not sequences:
        raise ValueError("real_corpus has no usable (non-damaged, non-empty) inscriptions")

    rng = random.Random(seed)

    lengths = [len(s) for s in sequences]

    overall_counts = Counter(sign for seq in sequences for sign in seq)
    overall_signs, overall_weights = zip(*overall_counts.items())

    initial_counts = Counter(seq[0] for seq in sequences)
    initial_signs, initial_weights = zip(*initial_counts.items())

    multi_sign_seqs = [s for s in sequences if len(s) >= 2]
    if multi_sign_seqs:
        final_counts = Counter(seq[-1] for seq in multi_sign_seqs)
        final_signs, final_weights = zip(*final_counts.items())
    else:
        # degenerate fallback if the corpus has no length->=2 inscriptions at all
        final_signs, final_weights = overall_signs, overall_weights

    def draw(signs, weights):
        return rng.choices(signs, weights=weights, k=1)[0]

    inscriptions = []
    for i in range(n_inscriptions):
        length = rng.choice(lengths)
        if length <= 1:
            seq = [draw(initial_signs, initial_weights)]
        else:
            seq = [draw(initial_signs, initial_weights)]
            seq += [draw(overall_signs, overall_weights) for _ in range(length - 2)]
            seq.append(draw(final_signs, final_weights))

        inscriptions.append(Inscription(
            inscription_id=f"{id_prefix}-{i:05d}",
            signs=seq,
            site="null_model",
            object_type="unknown",
            reading_direction="R-L",  # already constructed in canonical reading order
            motif="unknown",
        ))

    return Corpus(inscriptions)
