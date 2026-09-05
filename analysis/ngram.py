"""
analysis/ngram.py
===================
n-gram Markov chain modeling of sign sequences: transition matrices,
log-likelihood significance testing of bigrams/trigrams, cross-validated
perplexity, and stochastic restoration of masked signs.

Follows the framework of Yadav et al. (2010, PLOS ONE) and Rao et al.
(2009, PNAS): treat the corpus as strings over a sign alphabet, estimate
conditional probabilities P(s_i | s_j) directly from counts, and use
log-likelihood ratio tests to find pairs whose co-occurrence rate is
higher than chance would predict from unigram frequencies alone.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random


def bigram_counts(sequences: list[list[str]]) -> tuple[Counter, Counter]:
    """Returns (unigram_counts, bigram_counts) where bigram keys are (s_j, s_i)
    meaning s_i follows s_j."""
    uni = Counter()
    bi = Counter()
    for seq in sequences:
        uni.update(seq)
        for a, b in zip(seq, seq[1:]):
            bi[(a, b)] += 1
    return uni, bi


def transition_matrix(sequences: list[list[str]]) -> dict[str, dict[str, float]]:
    """P(s_i | s_j) as a nested dict: matrix[s_j][s_i] = probability."""
    uni, bi = bigram_counts(sequences)
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    totals = defaultdict(int)
    for (a, b), c in bi.items():
        totals[a] += c
    for (a, b), c in bi.items():
        matrix[a][b] = c / totals[a]
    return dict(matrix)


def log_likelihood_ratio(sequences: list[list[str]], min_count: int = 3) -> list[dict]:
    """Log-likelihood ratio (G2) test for each observed bigram against the
    null hypothesis that the two signs are independent, given their
    unigram frequencies. Returns bigrams sorted by significance, so that
    'highly frequent but not significant' vs 'moderately frequent but highly
    significant' pairs (as reported in the literature) can both surface."""
    uni, bi = bigram_counts(sequences)
    N = sum(uni.values())
    results = []
    for (a, b), o11 in bi.items():
        if o11 < min_count:
            continue
        o1_ = uni[a]
        o_1 = uni[b]
        o12 = o1_ - o11
        o21 = o_1 - o11
        o22 = N - o1_ - o_1 + o11
        # expected counts under independence
        e11 = o1_ * o_1 / N
        e12 = o1_ * (N - o_1) / N
        e21 = (N - o1_) * o_1 / N
        e22 = (N - o1_) * (N - o_1) / N

        def term(o, e):
            return o * math.log(o / e) if o > 0 and e > 0 else 0.0

        g2 = 2 * (term(o11, e11) + term(o12, e12) + term(o21, e21) + term(o22, e22))
        results.append({
            "sign_a": a, "sign_b": b, "count": o11,
            "p_b_given_a": o11 / o1_ if o1_ else 0.0,
            "g2": g2,
        })
    results.sort(key=lambda r: r["g2"], reverse=True)
    return results


@dataclass
class NgramModel:
    n: int
    counts: dict[tuple, Counter]
    context_totals: Counter

    def prob(self, context: tuple, sign: str, alpha: float = 0.5, vocab_size: int = 1) -> float:
        """Add-alpha (Lidstone) smoothed conditional probability."""
        c_ctx = self.counts.get(context, Counter())
        return (c_ctx.get(sign, 0) + alpha) / (self.context_totals.get(context, 0) + alpha * vocab_size)


def train_ngram(sequences: list[list[str]], n: int) -> NgramModel:
    counts: dict[tuple, Counter] = defaultdict(Counter)
    context_totals = Counter()
    for seq in sequences:
        padded = ["<S>"] * (n - 1) + seq
        for i in range(len(padded) - (n - 1)):
            context = tuple(padded[i:i + n - 1])
            sign = padded[i + n - 1]
            counts[context][sign] += 1
            context_totals[context] += 1
    return NgramModel(n=n, counts=dict(counts), context_totals=context_totals)


def perplexity(model: NgramModel, sequences: list[list[str]], vocab_size: int, alpha: float = 0.5) -> float:
    n = model.n
    log_prob_sum = 0.0
    count = 0
    for seq in sequences:
        padded = ["<S>"] * (n - 1) + seq
        for i in range(len(padded) - (n - 1)):
            context = tuple(padded[i:i + n - 1])
            sign = padded[i + n - 1]
            p = model.prob(context, sign, alpha=alpha, vocab_size=vocab_size)
            log_prob_sum += math.log2(p)
            count += 1
    if count == 0:
        return float("inf")
    return 2 ** (-log_prob_sum / count)


def cross_validated_perplexity(sequences: list[list[str]], n: int, k_folds: int = 5,
                                alpha: float = 0.5, seed: int = 0) -> dict:
    rng = random.Random(seed)
    seqs = sequences[:]
    rng.shuffle(seqs)
    fold_size = max(1, len(seqs) // k_folds)
    vocab_size = len(set(s for seq in seqs for s in seq)) + 1  # +1 for <S>
    ppls = []
    for k in range(k_folds):
        held_out = seqs[k * fold_size:(k + 1) * fold_size]
        train = seqs[:k * fold_size] + seqs[(k + 1) * fold_size:]
        if not held_out or not train:
            continue
        model = train_ngram(train, n)
        ppls.append(perplexity(model, held_out, vocab_size, alpha))
    return {"n": n, "fold_perplexities": ppls,
            "mean_perplexity": sum(ppls) / len(ppls) if ppls else float("nan")}


def restore_masked_sign(model: NgramModel, seq: list[str], mask_index: int,
                         vocab: list[str], alpha: float = 0.5) -> str:
    """Predict the most likely sign at mask_index given left context only
    (mirrors the stochastic restoration task used for damaged inscriptions)."""
    n = model.n
    padded = ["<S>"] * (n - 1) + seq[:mask_index]
    context = tuple(padded[-(n - 1):]) if n > 1 else tuple()
    best_sign, best_p = None, -1.0
    for candidate in vocab:
        p = model.prob(context, candidate, alpha=alpha, vocab_size=len(vocab))
        if p > best_p:
            best_sign, best_p = candidate, p
    return best_sign


def restoration_accuracy(sequences: list[list[str]], n: int = 2, trials: int = 500,
                          seed: int = 0) -> float:
    """Mask one random sign per sampled inscription (len >= 2), predict it
    from left context with a model trained on everything else, and report
    STRICT top-1 accuracy: does the single most probable sign match?

    NOTE: this is stricter than the ~75% figure reported in Yadav et al.
    (2010), which used a different criterion -- see
    restoration_accuracy_top90mass() below for a metric matching their
    actual published methodology. Comparing this function's output
    directly to their 75% headline number is comparing two different
    definitions of "correct."
    """
    rng = random.Random(seed)
    candidates = [s for s in sequences if len(s) >= 2]
    if not candidates:
        return float("nan")
    vocab = sorted(set(s for seq in sequences for s in seq))
    correct = 0
    total = 0
    for _ in range(trials):
        seq = rng.choice(candidates)
        idx = rng.randrange(len(seq))
        train_seqs = [s for s in sequences if s is not seq]
        model = train_ngram(train_seqs, n)
        pred = restore_masked_sign(model, seq, idx, vocab)
        if pred == seq[idx]:
            correct += 1
        total += 1
    return correct / total if total else float("nan")


def restoration_accuracy_top90mass(sequences: list[list[str]], n: int = 2, trials: int = 500,
                                    mass_threshold: float = 0.90, alpha: float = 0.5,
                                    seed: int = 0) -> dict:
    """Restoration accuracy using the actual methodology from Yadav et al.
    (2010, PLOS ONE): rank candidate signs by predicted probability given
    left context, accumulate probability mass in descending order, and
    count a restoration as correct if the TRUE sign falls within the set
    of candidates needed to reach `mass_threshold` (90% in their paper) of
    the total probability mass -- not just whether it's the single top
    prediction. This is a more lenient, and more directly comparable,
    metric than the strict top-1 restoration_accuracy() above.

    Also reports the mean candidate-set size, since this metric can look
    artificially high on a corpus with high vocabulary/low context
    predictiveness simply because it takes many signs to reach 90% mass --
    a large set size alongside a high accuracy means the metric is doing
    little real work, and should be read together, not the accuracy alone.
    """
    rng = random.Random(seed)
    candidates = [s for s in sequences if len(s) >= 2]
    if not candidates:
        return {"accuracy": float("nan"), "mean_candidate_set_size": float("nan")}
    vocab = sorted(set(s for seq in sequences for s in seq))
    correct = 0
    total = 0
    set_sizes = []
    for _ in range(trials):
        seq = rng.choice(candidates)
        idx = rng.randrange(len(seq))
        train_seqs = [s for s in sequences if s is not seq]
        model = train_ngram(train_seqs, n)

        pad = ["<S>"] * (n - 1) + seq[:idx]
        context = tuple(pad[-(n - 1):]) if n > 1 else tuple()
        scored = [(cand, model.prob(context, cand, alpha=alpha, vocab_size=len(vocab)))
                  for cand in vocab]
        scored.sort(key=lambda t: t[1], reverse=True)

        total_mass = sum(p for _, p in scored)
        cum = 0.0
        accepted = set()
        for cand, p in scored:
            if cum / total_mass >= mass_threshold and accepted:
                break
            accepted.add(cand)
            cum += p

        set_sizes.append(len(accepted))
        if seq[idx] in accepted:
            correct += 1
        total += 1

    return {
        "accuracy": correct / total if total else float("nan"),
        "mean_candidate_set_size": sum(set_sizes) / len(set_sizes) if set_sizes else float("nan"),
        "vocab_size": len(vocab),
        "mass_threshold": mass_threshold,
    }
