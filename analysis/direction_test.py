"""
analysis/direction_test.py
============================
Diagnostic for reading direction, prompted by an observation on real data:
the most frequent sign in both real corpora tested so far (ICIT G-codes and
CISI P-codes) sits at the INITIAL position, opposite to M77's famous
"jar sign" (#342), which is final-heavy. Three explanations are possible:
genuinely different sign schemes behave differently, a real convention
difference between digitizations, or a direction bug in how a corpus was
converted into this toolkit's CSV schema. This module gives evidence
toward telling those apart -- it does not settle the question by itself.

The diagnostic used here follows the logic in Rao et al. (2009, PNAS) and
Yadav et al. (2010, PLOS ONE): they report that TEXT ENDERS are more
strictly defined than TEXT BEGINNERS in the correctly-oriented M77 corpus
-- i.e. the final-position sign distribution has LOWER entropy (fewer
signs account for most of the mass) than the initial-position distribution.
This is presented in the literature as internal evidence for which end of
the text is more grammatically constrained, which is part of the basis for
the established right-to-left reading convention.

We compute this same asymmetry for a corpus BOTH as given and reversed,
and report which orientation matches the published fingerprint (final
more constrained than initial) more strongly. This is a heuristic
consistency check, not a proof of correct direction -- a corpus could
legitimately have different grammatical conventions than M77, and the
fingerprint itself (established on one corpus, in one sign scheme) is not
guaranteed to generalize. Treat a mismatch as "worth investigating further
in the conversion step or the data's own direction metadata," not as
proof the toolkit's converter is wrong.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math

from data.loader import Corpus
from analysis.entropy import conditional_entropy


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _positional_entropies(sequences: list[list[str]]) -> dict:
    initial = Counter(seq[0] for seq in sequences if seq)
    final = Counter(seq[-1] for seq in sequences if seq)
    h_initial = _shannon_entropy(initial)
    h_final = _shannon_entropy(final)
    return {
        "h_initial": h_initial,
        "h_final": h_final,
        "final_minus_initial_gap": h_initial - h_final,  # positive = final more constrained
        "n_distinct_initial": len(initial),
        "n_distinct_final": len(final),
    }


@dataclass
class DirectionTestResult:
    as_stored: dict
    reversed_: dict
    likely_direction: str
    note: str


def test_reading_direction(corpus: Corpus) -> DirectionTestResult:
    filtered = corpus.filter(exclude_damaged=True)
    as_stored_seqs = filtered.sequences(normalized=True)
    reversed_seqs = [list(reversed(s)) for s in as_stored_seqs]

    as_stored = _positional_entropies(as_stored_seqs)
    as_stored["conditional_entropy"] = conditional_entropy(as_stored_seqs)

    reversed_ = _positional_entropies(reversed_seqs)
    reversed_["conditional_entropy"] = conditional_entropy(reversed_seqs)

    gap_as_stored = as_stored["final_minus_initial_gap"]
    gap_reversed = reversed_["final_minus_initial_gap"]

    if gap_as_stored > gap_reversed and gap_as_stored > 0:
        likely = "as-stored"
        note = ("As-stored orientation shows the published fingerprint "
                "(final position more constrained than initial) more strongly.")
    elif gap_reversed > gap_as_stored and gap_reversed > 0:
        likely = "reversed"
        note = ("REVERSED orientation shows the published fingerprint more "
                "strongly than as-stored -- worth checking whether this "
                "corpus's reading_direction metadata or this toolkit's "
                "conversion step has the direction backwards.")
    else:
        likely = "inconclusive"
        note = ("Neither orientation shows a clear final-more-constrained "
                "asymmetry (or both are equally weak/strong). This could mean "
                "the corpus is too small, this sign scheme doesn't share "
                "M77's grammatical convention, or direction isn't "
                "recoverable from this signal alone.")

    return DirectionTestResult(
        as_stored=as_stored, reversed_=reversed_,
        likely_direction=likely, note=note,
    )
