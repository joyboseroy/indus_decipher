"""
experiments/permutation_controls.py
======================================
Runs four permutation-based controls (data/permutation_nulls.py) against
the real corpus, each destroying a different, precisely scoped piece of
structure, and reports where the classifiable signal actually lives:

  within_inscription_shuffle   destroys ALL within-inscription order
  global_shuffle                destroys order AND which signs co-occur
  position_preserving_shuffle   destroys only middle-sequence order
  bigram_markov_null            destroys only order-2+ (trigram+) structure

For each control: build paired instances (each real subsample and its
transformed counterpart, so instance-to-instance corpus-composition
variance is held constant), compare mean conditional_entropy and
perplexity_ratio_n2_n1, and run the same leave-one-out discrimination
test used elsewhere in this project. The pattern across all four tells
you which order of structure is doing the work: if within_inscription
and global shuffle are trivially distinguishable (expected, since ANY
order at all beats no order) but bigram_markov_null is NOT distinguishable
from real, that means the corpus's structure is well explained by
order-1 (bigram) statistics alone. If bigram_markov_null IS still
distinguishable, that is evidence of real structure beyond bigrams.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import Corpus, load_corpus_csv
from data.permutation_nulls import (within_inscription_shuffle, global_shuffle,
                                     position_preserving_shuffle, bigram_markov_null)
from analysis.falsification import extract_features, leave_one_out_accuracy, FEATURE_NAMES

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXP_DIR = Path(__file__).parent


def build_real_subsamples(real_corpus: Corpus, n_instances: int, n_per_instance: int, seed: int) -> list[Corpus]:
    import random
    rng = random.Random(seed)
    filtered = real_corpus.filter(exclude_damaged=True)
    all_ins = filtered.inscriptions
    n_eff = min(n_per_instance, len(all_ins))
    return [Corpus(rng.sample(all_ins, n_eff)) for _ in range(n_instances)]


CONTROLS = {
    "within_inscription_shuffle": lambda real_corpus, sample, seed: within_inscription_shuffle(sample, seed=seed),
    "global_shuffle": lambda real_corpus, sample, seed: global_shuffle(sample, seed=seed),
    "position_preserving_shuffle": lambda real_corpus, sample, seed: position_preserving_shuffle(sample, seed=seed),
    "bigram_markov_null": lambda real_corpus, sample, seed: bigram_markov_null(
        real_corpus, n_inscriptions=len(sample.inscriptions), seed=seed),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/indus_website_real_corpus.csv")
    parser.add_argument("--n_instances", type=int, default=8)
    parser.add_argument("--n_per_instance", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading real corpus from {args.csv}...")
    real_corpus = load_corpus_csv(args.csv)

    print(f"Building {args.n_instances} real subsamples ({args.n_per_instance} inscriptions each)...")
    real_samples = build_real_subsamples(real_corpus, args.n_instances, args.n_per_instance, seed=0)
    real_features = [extract_features(c) for c in real_samples]

    def mean_feature(features_list, name):
        vals = [getattr(fv, name) for fv in features_list]
        return sum(vals) / len(vals)

    results = {}
    print("\n" + "=" * 70)
    for control_name, control_fn in CONTROLS.items():
        print(f"\n=== Control: {control_name} ===")
        control_instances = [control_fn(real_corpus, real_samples[s], 2000 + s)
                              for s in range(args.n_instances)]
        control_features = [extract_features(c) for c in control_instances]

        print("  Feature comparison (real vs. control):")
        feature_summary = {}
        for name in FEATURE_NAMES:
            r = mean_feature(real_features, name)
            c = mean_feature(control_features, name)
            pct_diff = abs(r - c) / (abs(r) + 1e-9) * 100
            feature_summary[name] = {"real_mean": r, "control_mean": c, "pct_diff": pct_diff}
            print(f"    {name:32s}  real={r:8.4f}  control={c:8.4f}  diff={pct_diff:6.1f}%")

        loo = leave_one_out_accuracy({"real": real_samples, control_name: control_instances})
        print(f"  Leave-one-out discrimination accuracy: {loo['overall_accuracy']:.1%} "
              f"(chance = 50.0%)")

        results[control_name] = {
            "feature_summary": feature_summary,
            "discrimination_accuracy": loo["overall_accuracy"],
            "per_label_accuracy": loo["per_label_accuracy"],
        }

    print("\n" + "=" * 70)
    print("=== Summary: where does the classifiable signal live? ===")
    for control_name, r in results.items():
        acc = r["discrimination_accuracy"]
        verdict = "DISTINGUISHABLE" if acc > 0.85 else "NOT distinguishable" if acc < 0.65 else "borderline"
        print(f"  {control_name:32s} accuracy={acc:.1%}  ({verdict})")

    print("""
Reading this table: controls are listed from most structure-destroying
(within_inscription_shuffle) to least (bigram_markov_null, which keeps
real order-1 statistics intact). If accuracy stays high even for
bigram_markov_null, real order-2+ structure exists beyond what a bigram
model captures. If accuracy drops to near chance once order-1 statistics
are preserved, most of the corpus's predictable structure is local.""")

    # plot
    names = list(results.keys())
    accs = [results[n]["discrimination_accuracy"] for n in names]
    plt.figure(figsize=(8, 4.5))
    colors = ["#C44E52" if a > 0.85 else "#4C72B0" if a < 0.65 else "#DD8452" for a in accs]
    plt.barh(names, accs, color=colors)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    plt.xlabel("leave-one-out discrimination accuracy (real vs. control)")
    plt.title("Where does the classifiable signal live?\n(controls ordered: most to least structure-destroying)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "permutation_controls_summary.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "permutation_controls_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults written to {EXP_DIR / 'permutation_controls_results.json'}")
    print(f"Plot written to {OUT_DIR / 'permutation_controls_summary.png'}")


if __name__ == "__main__":
    main()
