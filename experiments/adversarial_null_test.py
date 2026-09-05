"""
experiments/adversarial_null_test.py
=======================================
The test proposed after resolving the two-real-corpora disagreement: can
the falsification harness's six-feature classifier tell the real corpus
apart from a non-linguistic system deliberately built to match its own
length distribution, vocabulary, sign frequencies, and positional
statistics (data/adversarial_null_model.py)?

This reuses analysis/falsification.py's existing leave-one-out machinery
unchanged, just with two labels ("real" and "null_matched") instead of
three, so the comparison is on identical footing to the three-civilization
self-test already in this repo.

Also prints the six features side by side, averaged across instances of
each label, since the module docstring for the null model predicts WHICH
features should match by construction (zipf_gamma, zipf_r2,
top_sign_final_share, mean_length) and which should differ if the real
corpus has genuine sequential structure beyond position and frequency
(conditional_entropy, perplexity_ratio_n2_n1). Seeing that prediction
confirmed or not is more informative than the accuracy number alone.
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
from data.adversarial_null_model import generate_matched_null_corpus
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


def build_null_instances(real_corpus: Corpus, n_instances: int, n_per_instance: int, seed: int) -> list[Corpus]:
    return [generate_matched_null_corpus(real_corpus, n_per_instance, seed=seed + s)
            for s in range(n_instances)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/indus_website_real_corpus.csv")
    parser.add_argument("--n_instances", type=int, default=8)
    parser.add_argument("--n_per_instance", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading real corpus from {args.csv}...")
    real_corpus = load_corpus_csv(args.csv)

    print(f"Building {args.n_instances} real subsamples and {args.n_instances} "
          f"statistics-matched null instances, {args.n_per_instance} inscriptions each...")
    real_instances = build_real_subsamples(real_corpus, args.n_instances, args.n_per_instance, seed=0)
    null_instances = build_null_instances(real_corpus, args.n_instances, args.n_per_instance, seed=1000)

    print("\n=== Average feature values: real vs. statistics-matched null ===")
    real_features = [extract_features(c) for c in real_instances]
    null_features = [extract_features(c) for c in null_instances]

    def mean_feature(features_list, name):
        vals = [getattr(fv, name) for fv in features_list]
        return sum(vals) / len(vals)

    predicted_to_match = {"zipf_gamma", "zipf_r2", "top_sign_final_share", "mean_length"}
    predicted_to_differ = {"conditional_entropy", "perplexity_ratio_n2_n1"}
    feature_summary = {}
    for name in FEATURE_NAMES:
        r = mean_feature(real_features, name)
        n = mean_feature(null_features, name)
        pct_diff = abs(r - n) / (abs(r) + 1e-9) * 100
        tag = ("matched by construction" if name in predicted_to_match
               else "predicted to differ" if name in predicted_to_differ else "")
        feature_summary[name] = {"real_mean": r, "null_mean": n, "pct_diff": pct_diff}
        print(f"  {name:32s}  real={r:8.4f}  null={n:8.4f}  diff={pct_diff:6.1f}%   {tag}")

    print("\n=== Leave-one-out discrimination test: real vs. statistics-matched null ===")
    loo = leave_one_out_accuracy({"real": real_instances, "null_matched": null_instances})
    print(f"Overall accuracy: {loo['overall_accuracy']:.1%} (chance for 2 classes = 50.0%)")
    print("Per-label accuracy:")
    for label, acc in loo["per_label_accuracy"].items():
        print(f"  {label}: {acc:.1%}")

    if loo["overall_accuracy"] > 0.85:
        verdict = ("The classifier CAN reliably tell the real corpus apart from a "
                   "statistics-matched non-linguistic system. Check the feature table "
                   "above: if conditional_entropy and perplexity_ratio_n2_n1 show the "
                   "largest gaps, that's consistent with the real corpus having genuine "
                   "sequential structure beyond position and frequency effects, which is "
                   "what this test was designed to detect.")
    elif loo["overall_accuracy"] < 0.65:
        verdict = ("The classifier CANNOT reliably tell the real corpus apart from a "
                   "statistics-matched non-linguistic system. This is an important "
                   "negative result: it means the six-feature classifier's earlier "
                   "'language-like' classification of the real corpus is weaker evidence "
                   "than it looks, since a system with no real grammar at all, built only "
                   "to match surface statistics, fools the same classifier.")
    else:
        verdict = ("Result is intermediate, neither a clear pass nor a clear failure of "
                   "this adversarial test. Treat as inconclusive; consider more instances "
                   "per label or a repeat at a different n_per_instance.")
    print(f"\n{verdict}")

    # bar chart of feature-wise percent difference
    names = FEATURE_NAMES
    diffs = [feature_summary[n]["pct_diff"] for n in names]
    colors = ["#4C72B0" if n in predicted_to_match else "#C44E52" if n in predicted_to_differ else "#888888"
              for n in names]
    plt.figure(figsize=(8, 4.5))
    plt.barh(names, diffs, color=colors)
    plt.xlabel("% difference (real vs. matched null, mean feature value)")
    plt.title("Which features actually differ between real data\nand its statistics-matched null model?")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "adversarial_null_feature_diffs.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "adversarial_null_results.json", "w") as f:
        json.dump({
            "n_instances": args.n_instances, "n_per_instance": args.n_per_instance,
            "feature_summary": feature_summary,
            "leave_one_out_accuracy": {
                "overall_accuracy": loo["overall_accuracy"],
                "per_label_accuracy": loo["per_label_accuracy"],
            },
            "verdict": verdict,
        }, f, indent=2, default=str)

    print(f"\nResults written to {EXP_DIR / 'adversarial_null_results.json'}")
    print(f"Plot written to {OUT_DIR / 'adversarial_null_feature_diffs.png'}")


if __name__ == "__main__":
    main()
