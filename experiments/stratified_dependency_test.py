"""
experiments/stratified_dependency_test.py
=============================================
Tests whether the validated order-3 finding (see README's "Order 3 is
validated..." section) survives conditioning on archaeological context,
per the H1/H2/H3 design from external review:

  H1 (linguistic/sequential): trigram signal is intrinsic to sign
     sequencing, independent of motif/site.
  H2 (archaeological/compositional): pooling different motif/site
     populations, each with their own sign preferences but NO real
     within-group sequential dependency, is enough to manufacture the
     appearance of trigram structure on its own.
  H3 (mixed): both contribute.

Compares real data's order-3 information gain (Kneser-Ney, as in
experiments/dependency_order_curve.py) against THREE stratified nulls
(data/stratified_null_model.py): motif-stratified, site-stratified, and
site+motif-stratified. Each stratified null still has ZERO real
sequential dependency (every sign is still drawn independently), but now
captures "different strata prefer different signs" -- exactly what H2
would need to succeed. If real data still beats all three, that is
stronger, more specific evidence for H1 than the single pooled
adversarial null (already run) could provide alone.
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
from data.stratified_null_model import (motif_stratified_null, site_stratified_null,
                                          site_motif_stratified_null)
from data.adversarial_null_model import generate_matched_null_corpus
from analysis.ngram import kn_cross_validated_perplexity

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXP_DIR = Path(__file__).parent


def entropy_at_orders(sequences: list[list[str]], orders: list[int], k_folds: int = 5) -> dict:
    curve = {}
    for n in orders:
        ppl = kn_cross_validated_perplexity(sequences, n=n, k_folds=k_folds)["mean_perplexity"]
        curve[n] = math.log2(ppl) if ppl and ppl > 0 else float("nan")
    return curve


def main():
    print("Loading real corpus and building stratified + pooled nulls...")
    real_corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    real_filtered = real_corpus.filter(exclude_damaged=True)
    real_sequences = real_filtered.sequences(normalized=True)
    n = len(real_filtered)

    nulls = {
        "pooled_adversarial_null (original, no stratification)":
            generate_matched_null_corpus(real_corpus, n_inscriptions=n, seed=42),
        "motif_stratified_null (H2: motif composition alone)":
            motif_stratified_null(real_corpus, n_inscriptions=n, seed=42),
        "site_stratified_null (H2: site composition alone)":
            site_stratified_null(real_corpus, n_inscriptions=n, seed=42),
        "site_motif_stratified_null (H2: site+motif composition alone)":
            site_motif_stratified_null(real_corpus, n_inscriptions=n, seed=42),
    }

    orders = [1, 2, 3]
    print(f"Computing Kneser-Ney cross-validated entropy at orders {orders} "
          f"for real data and {len(nulls)} nulls...\n")

    real_curve = entropy_at_orders(real_sequences, orders)
    real_gain_3 = real_curve[2] - real_curve[3]
    print(f"Real corpus:  H1={real_curve[1]:.3f}  H2={real_curve[2]:.3f}  "
          f"H3={real_curve[3]:.3f}  order-3 gain={real_gain_3:+.3f} bits")

    results = {"real": {"entropy": real_curve, "order_3_gain": real_gain_3}}
    print()
    for label, null_corpus in nulls.items():
        seqs = null_corpus.sequences(normalized=True)
        curve = entropy_at_orders(seqs, orders)
        gain_3 = curve[2] - curve[3]
        results[label] = {"entropy": curve, "order_3_gain": gain_3}
        beats = "real STILL beats this null" if real_gain_3 > gain_3 else "does NOT beat this null"
        print(f"{label}:\n  H1={curve[1]:.3f}  H2={curve[2]:.3f}  H3={curve[3]:.3f}  "
              f"order-3 gain={gain_3:+.3f} bits  ({beats})\n")

    all_beat = all(real_gain_3 > results[label]["order_3_gain"] for label in nulls)
    print("=" * 70)
    if all_beat:
        verdict = ("Real data's order-3 information gain exceeds EVERY stratified null, "
                   "including the one that captures site+motif composition jointly. "
                   "Since each stratified null had every opportunity H2 (archaeological "
                   "composition alone) would need to succeed, and none did, this is real "
                   "evidence favoring H1: the order-3 signal is intrinsic to sign "
                   "sequencing, not an artifact of pooling archaeologically distinct "
                   "sign-frequency populations together.")
    else:
        failed = [label for label in nulls if real_gain_3 <= results[label]["order_3_gain"]]
        verdict = (f"Real data does NOT clearly beat: {failed}. This does not refute H1 "
                   f"outright, but it means archaeological composition (H2/H3) cannot yet "
                   f"be ruled out as at least a partial explanation for the order-3 signal.")
    print(verdict)

    # plot
    labels = ["real"] + list(nulls.keys())
    gains = [results[l]["order_3_gain"] for l in labels]
    short_labels = ["real corpus"] + [l.split(" (")[0] for l in nulls.keys()]
    plt.figure(figsize=(9, 4.5))
    colors = ["#C44E52" if l == "real corpus" else "#4C72B0" for l in short_labels]
    plt.barh(short_labels, gains, color=colors)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("order-3 information gain (bits)")
    plt.title("Does real data still beat archaeologically-stratified nulls?\n"
              "(each null has zero real sequential dependency, but per-stratum sign preferences)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stratified_dependency_test.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "stratified_dependency_test_results.json", "w") as f:
        json.dump({"results": results, "all_beat": all_beat, "verdict": verdict},
                   f, indent=2, default=str)

    print(f"\nResults written to {EXP_DIR / 'stratified_dependency_test_results.json'}")
    print(f"Plot written to {OUT_DIR / 'stratified_dependency_test.png'}")


if __name__ == "__main__":
    main()
