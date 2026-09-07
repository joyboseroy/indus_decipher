"""
experiments/dependency_order_curve.py
=========================================
Sharpens "the corpus has structure beyond bigram order" (established in
experiments/permutation_controls.py) into a specific question: at what
context length does predictive information saturate?

For orders n=1..6, computes cross-validated held-out entropy
H_n = log2(cross_validated_perplexity(n)) using the existing add-alpha
n-gram machinery in analysis/ngram.py, then the information gain from
adding one more sign of context:

    gain_n = H_{n-1} - H_n

A large, sustained gain out to high n means real long-range structure. A
gain that drops to near zero after n=2 or 3 means the corpus's
predictable structure is mostly local, consistent with what
permutation_controls.py already showed (real data is still distinguishable
from a bigram-order null, but that doesn't by itself say HOW much
higher-order structure exists, or at what order it stops mattering).

Run for three corpora side by side: the real data, a bigram_markov_null
built from it (should show gain collapse to near zero after n=2 by
construction, a useful sanity check that this curve-fitting procedure
actually detects what it claims to), and the adversarial null model
(should show near-zero gain at every order, since it has no dependency
at all beyond position and frequency).

HONESTY NOTE, checked and confirmed by actually running this before
trusting it: naive add-alpha n-gram smoothing suffers badly from context
sparsity at high order on a corpus this size (a few thousand short
inscriptions). If held-out entropy starts RISING at high n rather than
flattening, that is the classic n-gram sparsity failure mode (most
high-order contexts are unseen, and smoothing falls back to something
close to a uniform, uninformative distribution), not evidence of some
exotic anti-correlation in the script. This module reports where that
starts happening rather than silently extending the curve past the point
where it stops meaning what it looks like it means.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from data.permutation_nulls import bigram_markov_null
from data.adversarial_null_model import generate_matched_null_corpus
from analysis.ngram import cross_validated_perplexity

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXP_DIR = Path(__file__).parent

MAX_ORDER = 6


def entropy_curve(sequences: list[list[str]], max_order: int = MAX_ORDER, k_folds: int = 5) -> dict:
    curve = {}
    for n in range(1, max_order + 1):
        ppl = cross_validated_perplexity(sequences, n=n, k_folds=k_folds)["mean_perplexity"]
        h = math.log2(ppl) if ppl and ppl > 0 else float("nan")
        curve[n] = {"perplexity": ppl, "entropy_bits": h}
    for n in range(2, max_order + 1):
        curve[n]["information_gain_bits"] = curve[n - 1]["entropy_bits"] - curve[n]["entropy_bits"]
    curve[1]["information_gain_bits"] = None  # no n=0 baseline
    return curve


def print_curve(name: str, curve: dict):
    print(f"\n{name}:")
    print(f"  {'order':>6s}  {'perplexity':>12s}  {'entropy (bits)':>15s}  {'gain vs n-1':>12s}")
    for n, row in curve.items():
        gain_str = f"{row['information_gain_bits']:+.3f}" if row["information_gain_bits"] is not None else "n/a"
        print(f"  {n:>6d}  {row['perplexity']:12.2f}  {row['entropy_bits']:15.3f}  {gain_str:>12s}")


def main():
    print("Loading corpora and building comparison nulls...")
    real_corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    real_filtered = real_corpus.filter(exclude_damaged=True)
    real_sequences = real_filtered.sequences(normalized=True)

    bigram_null_corpus = bigram_markov_null(real_corpus, n_inscriptions=len(real_filtered), seed=42)
    bigram_null_sequences = bigram_null_corpus.sequences(normalized=True)

    adv_null_corpus = generate_matched_null_corpus(real_corpus, n_inscriptions=len(real_filtered), seed=42)
    adv_null_sequences = adv_null_corpus.sequences(normalized=True)

    print(f"Computing cross-validated entropy for orders 1..{MAX_ORDER} "
          f"(this trains {MAX_ORDER} models x 5 folds x 3 corpora)...")

    real_curve = entropy_curve(real_sequences)
    print_curve("Real corpus (indus_website)", real_curve)

    bigram_curve = entropy_curve(bigram_null_sequences)
    print_curve("Bigram-order null (should show gain collapse after n=2)", bigram_curve)

    adv_curve = entropy_curve(adv_null_sequences)
    print_curve("Adversarial (no-dependency) null (should show ~zero gain at every order)", adv_curve)

    # find where real data's gain drops below a small threshold, as a rough
    # "saturation order" -- reported as a descriptive observation, not a
    # formal statistical test
    threshold = 0.05
    saturation_order = None
    for n in range(2, MAX_ORDER + 1):
        if real_curve[n]["information_gain_bits"] < threshold:
            saturation_order = n
            break
    if saturation_order:
        print(f"\nReal corpus's information gain drops below {threshold} bits at order {saturation_order}.")
    else:
        print(f"\nReal corpus's information gain has NOT dropped below {threshold} bits "
              f"by order {MAX_ORDER} -- either real structure extends further, or this is "
              f"where add-alpha sparsity starts distorting the estimate (see module docstring).")

    rising = [n for n in range(2, MAX_ORDER + 1) if real_curve[n]["entropy_bits"] > real_curve[n - 1]["entropy_bits"]]
    if rising:
        print(f"\nCAUTION: held-out entropy INCREASED from order {rising[0]-1} to {rising[0]} "
              f"(and possibly later orders: {rising}). This is the expected signature of n-gram "
              f"sparsity at high order on a corpus this size, not evidence of unusual structure. "
              f"Read any 'gain' numbers at or after this order with that in mind.")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for curve, label, marker in [(real_curve, "real corpus", "o"),
                                   (bigram_curve, "bigram-order null", "s"),
                                   (adv_curve, "adversarial (no-dependency) null", "^")]:
        ns = list(curve.keys())
        axes[0].plot(ns, [curve[n]["entropy_bits"] for n in ns], marker=marker, label=label)
        gain_ns = [n for n in ns if curve[n]["information_gain_bits"] is not None]
        axes[1].plot(gain_ns, [curve[n]["information_gain_bits"] for n in gain_ns], marker=marker, label=label)

    axes[0].set_xlabel("n-gram order"); axes[0].set_ylabel("cross-validated entropy (bits)")
    axes[0].set_title("Held-out entropy vs. context length")
    axes[0].legend()
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("n-gram order"); axes[1].set_ylabel("information gain (bits)")
    axes[1].set_title("Information gain from one more sign of context")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dependency_order_curve.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "dependency_order_curve_results.json", "w") as f:
        json.dump({
            "real_corpus": real_curve, "bigram_null": bigram_curve, "adversarial_null": adv_curve,
            "saturation_order_at_0.05_bits": saturation_order,
            "orders_with_rising_entropy": rising,
        }, f, indent=2, default=str)

    print(f"\nResults written to {EXP_DIR / 'dependency_order_curve_results.json'}")
    print(f"Plot written to {OUT_DIR / 'dependency_order_curve.png'}")


if __name__ == "__main__":
    main()
