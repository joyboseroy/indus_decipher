"""
experiments/discount_sensitivity_test.py
============================================
Closes the one flagged loose end on the headline order-3 finding
(README's "Order 3 is validated..." section): the discount parameter
used throughout (0.75) is the standard literature default, not tuned
against this corpus. This sweeps the discount from 0.5 to 0.9 and
reruns the exact same three-way comparison (real corpus vs. bigram-order
null vs. adversarial null) at each value, to check whether "real
positive, both nulls negative" survives across the whole range or was
an artifact of one specific discount choice.
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
from analysis.ngram import kn_cross_validated_perplexity

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXP_DIR = Path(__file__).parent

DISCOUNTS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def order_3_gain(sequences, discount, k_folds=5):
    h2 = math.log2(kn_cross_validated_perplexity(sequences, n=2, k_folds=k_folds, discount=discount)["mean_perplexity"])
    h3 = math.log2(kn_cross_validated_perplexity(sequences, n=3, k_folds=k_folds, discount=discount)["mean_perplexity"])
    return h2 - h3


def main():
    print("Loading real corpus and building comparison nulls (fixed seed, same as prior order-3 tests)...")
    real_corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    real_filtered = real_corpus.filter(exclude_damaged=True)
    real_sequences = real_filtered.sequences(normalized=True)

    bigram_null = bigram_markov_null(real_corpus, n_inscriptions=len(real_filtered), seed=42)
    bigram_sequences = bigram_null.sequences(normalized=True)

    adv_null = generate_matched_null_corpus(real_corpus, n_inscriptions=len(real_filtered), seed=42)
    adv_sequences = adv_null.sequences(normalized=True)

    print(f"\nSweeping discount over {DISCOUNTS}...\n")
    print(f"{'discount':>9s} {'real gain':>11s} {'bigram-null gain':>18s} {'adv-null gain':>15s} {'pattern holds?':>15s}")
    print("-" * 75)

    results = {}
    all_hold = True
    for d in DISCOUNTS:
        real_gain = order_3_gain(real_sequences, d)
        bigram_gain = order_3_gain(bigram_sequences, d)
        adv_gain = order_3_gain(adv_sequences, d)
        holds = real_gain > 0 and bigram_gain < 0 and adv_gain < 0
        all_hold = all_hold and holds
        results[d] = {"real_gain": real_gain, "bigram_null_gain": bigram_gain, "adv_null_gain": adv_gain,
                       "pattern_holds": holds}
        print(f"{d:>9.2f} {real_gain:>+11.3f} {bigram_gain:>+18.3f} {adv_gain:>+15.3f} "
              f"{'YES' if holds else 'no':>15s}")

    print(f"\n{'='*70}")
    if all_hold:
        verdict = (f"The three-way pattern (real positive, both nulls negative) holds at "
                   f"EVERY discount value tested from {min(DISCOUNTS)} to {max(DISCOUNTS)}. "
                   f"The order-3 finding is not an artifact of the specific default discount "
                   f"(0.75) chosen; it is robust across the range a reasonable analyst might "
                   f"have picked without tuning.")
    else:
        failed = [d for d in DISCOUNTS if not results[d]["pattern_holds"]]
        verdict = (f"The pattern does NOT hold at every discount tested; it fails at "
                   f"{failed}. The order-3 finding should be reported as discount-dependent, "
                   f"not as a discount-independent result, and the specific range where it "
                   f"holds should be stated explicitly rather than picking the default value "
                   f"and moving on.")
    print(verdict)

    # plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(DISCOUNTS, [results[d]["real_gain"] for d in DISCOUNTS], marker="o", color="#C44E52", label="real corpus")
    plt.plot(DISCOUNTS, [results[d]["bigram_null_gain"] for d in DISCOUNTS], marker="s", color="#4C72B0", label="bigram-order null")
    plt.plot(DISCOUNTS, [results[d]["adv_null_gain"] for d in DISCOUNTS], marker="^", color="#55A868", label="adversarial null")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Kneser-Ney discount parameter")
    plt.ylabel("order-3 information gain (bits)")
    plt.title("Does the order-3 finding survive discount choice?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "discount_sensitivity.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "discount_sensitivity_results.json", "w") as f:
        json.dump({"results": results, "all_hold": all_hold, "verdict": verdict}, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'discount_sensitivity_results.json'}")
    print(f"Plot written to {OUT_DIR / 'discount_sensitivity.png'}")


if __name__ == "__main__":
    main()
