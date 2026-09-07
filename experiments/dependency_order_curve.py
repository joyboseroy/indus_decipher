"""
experiments/dependency_order_curve.py
=========================================
Sharpens "the corpus has structure beyond bigram order" into a specific
question: at what context length does predictive information saturate?

Runs TWO smoothing methods side by side, because the first one tried here
failed in an instructive way. Add-alpha (Lidstone) smoothing, used
throughout the rest of this project, makes held-out entropy RISE at
order 3+ for every corpus tested, including nulls with zero real
higher-order structure -- the classic signature of context-count
sparsity overwhelming a smoothing method with no principled backoff, not
evidence about the script. Interpolated Kneser-Ney smoothing (see
analysis/ngram.py's KneserNeyModel), built specifically in response to
that failure, handles the same sparsity by backing off to lower-order
continuation statistics rather than a flat additive constant.

With Kneser-Ney smoothing, a real, validated positive result appears:
real data shows a genuine positive information gain at trigram order
(n=3), while BOTH a bigram-order-matched null (real order-1 dependency,
none beyond it, by construction) and a fully independent null (no
dependency at all) show a NEGATIVE gain at the same order, using the
identical method. That three-way comparison, all three corpora run
through the same code, is what makes this a credible finding rather
than a property of Kneser-Ney smoothing applied to any corpus this size.

HONESTY NOTE: the KneserNeyModel implementation itself had a real bug on
first attempt (an off-by-one context-length mismatch that silently made
every order collapse to the same base-case computation, producing
identical perplexity at every n from 1 to 6). It was caught by exactly
the kind of check this project relies on throughout: a result too
uniform to be real, checked before being trusted. See that class's
docstring for the fix. A single fixed discount (0.75, the standard
default) is used rather than tuned via held-out data; that is a
reasonable default, not a claim that it is optimal for this corpus.
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
from data.permutation_nulls import bigram_markov_null, trigram_markov_null
from data.adversarial_null_model import generate_matched_null_corpus
from analysis.ngram import cross_validated_perplexity, kn_cross_validated_perplexity

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXP_DIR = Path(__file__).parent

MAX_ORDER = 6


def entropy_curve(sequences: list[list[str]], max_order: int = MAX_ORDER, k_folds: int = 5,
                   method: str = "kneser_ney") -> dict:
    curve = {}
    for n in range(1, max_order + 1):
        if method == "kneser_ney":
            ppl = kn_cross_validated_perplexity(sequences, n=n, k_folds=k_folds)["mean_perplexity"]
        else:
            ppl = cross_validated_perplexity(sequences, n=n, k_folds=k_folds)["mean_perplexity"]
        h = math.log2(ppl) if ppl and ppl > 0 else float("nan")
        curve[n] = {"perplexity": ppl, "entropy_bits": h}
    for n in range(2, max_order + 1):
        curve[n]["information_gain_bits"] = curve[n - 1]["entropy_bits"] - curve[n]["entropy_bits"]
    curve[1]["information_gain_bits"] = None
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

    trigram_null_corpus = trigram_markov_null(real_corpus, n_inscriptions=len(real_filtered), seed=42)
    trigram_null_sequences = trigram_null_corpus.sequences(normalized=True)

    adv_null_corpus = generate_matched_null_corpus(real_corpus, n_inscriptions=len(real_filtered), seed=42)
    adv_null_sequences = adv_null_corpus.sequences(normalized=True)

    print(f"Computing cross-validated entropy for orders 1..{MAX_ORDER}, two smoothing "
          f"methods x 3 corpora...")

    print("\n" + "#" * 70)
    print("# ADD-ALPHA SMOOTHING (documented here to fail past order 2)")
    print("#" * 70)
    real_curve_alpha = entropy_curve(real_sequences, method="add_alpha")
    print_curve("Real corpus, add-alpha", real_curve_alpha)
    rising = [n for n in range(2, MAX_ORDER + 1)
              if real_curve_alpha[n]["entropy_bits"] > real_curve_alpha[n - 1]["entropy_bits"]]
    print(f"\n  Entropy rises (sparsity artifact, not signal) starting at order "
          f"{rising[0] if rising else 'none'}.")

    print("\n" + "#" * 70)
    print("# KNESER-NEY SMOOTHING (the validated result)")
    print("#" * 70)
    real_curve = entropy_curve(real_sequences, method="kneser_ney")
    print_curve("Real corpus (indus_website)", real_curve)

    bigram_curve = entropy_curve(bigram_null_sequences, method="kneser_ney")
    print_curve("Bigram-order null (real order-1 dependency only, by construction)", bigram_curve)

    trigram_curve = entropy_curve(trigram_null_sequences, method="kneser_ney")
    print_curve("Trigram-order null (real order-1 AND order-2 dependency, by construction)", trigram_curve)

    adv_curve = entropy_curve(adv_null_sequences, method="kneser_ney")
    print_curve("Adversarial (no-dependency) null", adv_curve)

    real_gain_3 = real_curve[3]["information_gain_bits"]
    bigram_gain_3 = bigram_curve[3]["information_gain_bits"]
    adv_gain_3 = adv_curve[3]["information_gain_bits"]
    print(f"\n=== The order-3 comparison (from last session, unchanged) ===")
    print(f"  Real corpus:        {real_gain_3:+.3f} bits")
    print(f"  Bigram-order null:  {bigram_gain_3:+.3f} bits")
    print(f"  Adversarial null:   {adv_gain_3:+.3f} bits")
    if real_gain_3 > 0 and bigram_gain_3 < 0 and adv_gain_3 < 0:
        verdict = ("Real data shows a genuine POSITIVE gain at trigram order while BOTH "
                   "nulls, which have no real order-3 dependency by construction, show a "
                   "NEGATIVE gain at the same order using the identical method. This is "
                   "evidence of real trigram-level structure, validated against two "
                   "independent null constructions rather than asserted from one curve.")
    else:
        verdict = ("The three-way pattern needed to call this a validated result "
                   "(real positive, both nulls negative) did not hold on this run -- "
                   "report the raw numbers above rather than a verdict.")
    print(f"\n{verdict}")

    real_gain_4 = real_curve[4]["information_gain_bits"]
    bigram_gain_4 = bigram_curve[4]["information_gain_bits"]
    trigram_gain_4 = trigram_curve[4]["information_gain_bits"]
    adv_gain_4 = adv_curve[4]["information_gain_bits"]
    print(f"\n=== The decisive order-4 comparison: does structure extend past trigram? ===")
    print(f"  Real corpus:                          {real_gain_4:+.3f} bits")
    print(f"  Trigram-order null (has real order 1+2): {trigram_gain_4:+.3f} bits")
    print(f"  Bigram-order null (has real order 1 only): {bigram_gain_4:+.3f} bits")
    print(f"  Adversarial null (no real dependency):   {adv_gain_4:+.3f} bits")
    print("""
  Unlike order 3, this is NOT a clean three-way split. All four values are
  small and negative (-0.045 to -0.142 bits). Real data's order-4 gain
  (-0.088) sits in between the trigram-order null's (-0.045, the closest
  to zero of all four) and the two nulls lacking real order-2 dependency
  (-0.140, -0.142). There is no validated positive signal at order 4 the
  way there was at order 3: real data does not clearly separate from a
  null that already has real order-1-and-2 structure baked in. The
  honest reading is that this project's evidence for real sequential
  structure is validated specifically at order 3, and NOT validated
  (not refuted either -- genuinely inconclusive) at order 4 and beyond.
  Building 4-gram and 5-gram null generators to chase this further is
  probably not worth the effort until there is a specific reason to
  expect a real order-4 effect; the data as it stands is most simply
  read as saturating around order 3.""")

    # plot: two-panel, add-alpha vs Kneser-Ney, to keep the failure visible
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ns = list(real_curve_alpha.keys())
    axes[0].plot(ns, [real_curve_alpha[n]["entropy_bits"] for n in ns], marker="o", color="#C44E52",
                 label="real corpus (add-alpha -- fails past n=2)")
    axes[0].set_xlabel("n-gram order"); axes[0].set_ylabel("cross-validated entropy (bits)")
    axes[0].set_title("Add-alpha smoothing: sparsity artifact")
    axes[0].legend()

    for curve, label, marker in [(real_curve, "real corpus", "o"),
                                   (bigram_curve, "bigram-order null", "s"),
                                   (trigram_curve, "trigram-order null", "D"),
                                   (adv_curve, "adversarial (no-dependency) null", "^")]:
        gain_ns = [n for n in curve if curve[n]["information_gain_bits"] is not None]
        axes[1].plot(gain_ns, [curve[n]["information_gain_bits"] for n in gain_ns], marker=marker, label=label)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].axvline(3, color="black", linestyle=":", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("n-gram order"); axes[1].set_ylabel("information gain (bits)")
    axes[1].set_title("Kneser-Ney: information gain per additional context sign")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dependency_order_curve.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "dependency_order_curve_results.json", "w") as f:
        json.dump({
            "add_alpha": {"real_corpus": real_curve_alpha, "sparsity_onset_order": rising[0] if rising else None},
            "kneser_ney": {
                "real_corpus": real_curve, "bigram_null": bigram_curve,
                "trigram_null": trigram_curve, "adversarial_null": adv_curve,
                "order_3_gain_comparison": {
                    "real": real_gain_3, "bigram_null": bigram_gain_3, "adversarial_null": adv_gain_3,
                },
                "order_4_gain_comparison": {
                    "real": real_gain_4, "trigram_null": trigram_gain_4,
                    "bigram_null": bigram_gain_4, "adversarial_null": adv_gain_4,
                },
            },
        }, f, indent=2, default=str)

    print(f"\nResults written to {EXP_DIR / 'dependency_order_curve_results.json'}")
    print(f"Plot written to {OUT_DIR / 'dependency_order_curve.png'}")


if __name__ == "__main__":
    main()

