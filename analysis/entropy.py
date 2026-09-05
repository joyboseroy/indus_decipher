"""
analysis/entropy.py
=====================
Conditional entropy H(X|Y) analysis, following Rao et al. (2009, PNAS) and
the reply to Farmer/Sproat/Witzel (2004) in Rao et al. (2010, Computational
Linguistics). The logic:

  - A rigid, formulaic non-linguistic system (heraldic emblems, fixed
    administrative codes) has LOW conditional entropy: knowing one sign
    almost fully determines the next.
  - Fully random sequences have HIGH conditional entropy: knowing one sign
    tells you nothing about the next.
  - Natural spoken languages sit in between: there is real constraint
    (grammar) but also real freedom (choice of words).

Where the actual script's conditional entropy falls, relative to control
corpora, is evidence (not proof) about which regime it belongs to.

CONTROL CORPORA
----------------
This module ships two synthetic, generator-based controls (`fully_random`
and `rigid_fixed`) built directly from the script's own vocabulary size, so
they always give a fair, calibrated comparison regardless of alphabet size.

For the natural-language and other non-linguistic comparisons used in the
literature (Sumerian, Old Tamil, Vedic Sanskrit, English, DNA/protein,
computer code), supply your own tokenized reference sequences via
`external_control_entropy()` -- e.g. character n-grams of a plain text
corpus you already have rights to use. This module does not bundle or
fetch any third-party text.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random


def conditional_entropy(sequences: list[list[str]]) -> float:
    """H(X_i | X_{i-1}) estimated directly from bigram counts (order-1)."""
    joint = Counter()
    marg = Counter()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            joint[(a, b)] += 1
            marg[a] += 1
    total = sum(joint.values())
    if total == 0:
        return float("nan")
    h = 0.0
    for (a, b), c_ab in joint.items():
        p_ab = c_ab / total
        p_b_given_a = c_ab / marg[a]
        h -= p_ab * math.log2(p_b_given_a)
    return h


def entropy_by_sequence_length(sequences: list[list[str]], max_len: int = 6) -> dict[int, float]:
    """Conditional entropy computed separately using only positions up to
    a cumulative window length k=1..max_len, the way Rao et al. plot entropy
    as a function of sequence length to compare curve *shapes* across
    systems, not just single numbers."""
    out = {}
    for k in range(1, max_len + 1):
        truncated = [seq[:k] for seq in sequences if len(seq) >= 2]
        out[k] = conditional_entropy(truncated)
    return out


def fully_random_control(vocab_size: int, n_sequences: int, mean_len: float,
                          seed: int = 0) -> list[list[str]]:
    """Upper-entropy control: every sign drawn uniformly at random,
    independent of context."""
    rng = random.Random(seed)
    vocab = [f"R{i}" for i in range(vocab_size)]
    seqs = []
    for _ in range(n_sequences):
        length = max(1, round(rng.gauss(mean_len, 1.5)))
        seqs.append([rng.choice(vocab) for _ in range(length)])
    return seqs


def rigid_fixed_control(vocab_size: int, n_sequences: int, mean_len: float,
                         n_templates: int = 5, seed: int = 0) -> list[list[str]]:
    """Lower-entropy control: sequences drawn from a small fixed set of
    templates (mimics heraldic/emblematic systems with little combinatorial
    freedom)."""
    rng = random.Random(seed)
    vocab = [f"F{i}" for i in range(vocab_size)]
    templates = []
    for _ in range(n_templates):
        length = max(1, round(mean_len))
        templates.append([rng.choice(vocab) for _ in range(length)])
    return [list(rng.choice(templates)) for _ in range(n_sequences)]


def external_control_entropy(tokenized_sequences: list[list[str]], label: str) -> dict:
    """Compute the same conditional-entropy metric on a user-supplied
    reference corpus (e.g. a real natural-language or non-linguistic
    dataset you have loaded yourself), so it can be compared on equal
    footing with the Indus corpus and the built-in synthetic controls."""
    return {"label": label, "conditional_entropy": conditional_entropy(tokenized_sequences)}


def compare_against_controls(sequences: list[list[str]], seed: int = 0) -> dict:
    vocab_size = len(set(s for seq in sequences for s in seq))
    lens = [len(s) for s in sequences]
    mean_len = sum(lens) / len(lens) if lens else 4.0

    real_h = conditional_entropy(sequences)
    random_h = conditional_entropy(fully_random_control(vocab_size, len(sequences), mean_len, seed))
    rigid_h = conditional_entropy(rigid_fixed_control(vocab_size, len(sequences), mean_len, seed=seed))

    return {
        "target_conditional_entropy": real_h,
        "fully_random_control_entropy": random_h,
        "rigid_fixed_control_entropy": rigid_h,
        "interpretation": (
            "intermediate-between-controls (consistent with a rule-governed, "
            "flexible sign system)"
            if rigid_h < real_h < random_h else
            "does not fall strictly between the two synthetic controls -- "
            "inspect the numbers directly rather than trusting this label"
        ),
    }
