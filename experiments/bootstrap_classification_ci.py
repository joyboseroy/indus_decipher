"""
experiments/bootstrap_classification_ci.py
==============================================
Replaces every point-estimate classification in this project ("the large
corpus classifies as language-like") with a proper confidence interval on
that classification, computed by repeated random subsampling WITHOUT
replacement, and reports the proportion landing on each label with a
Wilson score interval.

IMPORTANT CORRECTION, kept here rather than silently fixed: the first
version of this script used the standard bootstrap (resampling WITH
replacement at the corpus's own size). That produced a severe and
misleading result: the large corpus's classification flipped from
"language-like" (its unperturbed point estimate, and its result under
every earlier test in this project) to "mixed" in 97% of bootstrap
draws. Direct investigation traced this to a real bug in combining
with-replacement bootstrap with this project's cross-validated
perplexity feature: resampling with replacement at full size produces
roughly 37-49% exact duplicate sequences, and when a duplicate sequence
lands in both a training fold and the held-out fold of the internal
k-fold split, a bigram model can effectively "memorize" it from training
and then "predict" it correctly in the held-out fold, an information
leak that a unigram model, unable to memorize whole sequences, does not
benefit from nearly as much. Measured directly: a single with-replacement
resample showed a 26% drop in bigram cross-validated perplexity but only
a 5% drop in unigram perplexity, exactly the asymmetric pattern leakage
would produce, and enough to flip perplexity_ratio_n2_n1 (one of the six
classifier features) far enough to change the nearest-centroid label.

The fix used here is subsampling WITHOUT replacement at a fixed fraction
of each corpus's size (80% by default). This still produces genuine
resample-to-resample variability suitable for a confidence interval,
without ever creating an exact duplicate inscription that could leak
across a cross-validation fold boundary. This is a well-established
alternative to the classic bootstrap for exactly this kind of pipeline;
it is used here after empirically DISCOVERING why the more standard
approach failed, not chosen from the outset.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import Corpus, load_corpus_csv
from analysis.falsification import extract_features, classify_by_nearest_centroid
from data.synthetic_civilizations import GENERATORS

EXP_DIR = Path(__file__).parent


def wilson_score_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (z=1.96). More
    reliable than mean +/- 1.96*SE when the proportion is close to 0 or 1,
    which is common here (several corpora classify consistently)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def build_fixed_reference_centroids(seed: int = 100, n_instances: int = 4, n_per_instance: int = 500):
    reference_features = {}
    for label, gen_fn in GENERATORS.items():
        reference_features[label] = [
            extract_features(gen_fn(n_inscriptions=n_per_instance, seed=seed + s))
            for s in range(n_instances)
        ]
    return reference_features


def subsample_classification(corpus: Corpus, reference_features: dict,
                               n_resamples: int, fraction: float, seed: int) -> dict:
    rng = random.Random(seed)
    filtered = corpus.filter(exclude_damaged=True)
    all_ins = filtered.inscriptions
    n_full = len(all_ins)
    n_sub = max(1, int(round(n_full * fraction)))

    label_counts = {label: 0 for label in reference_features}
    for _ in range(n_resamples):
        resample = Corpus(rng.sample(all_ins, n_sub))  # WITHOUT replacement, no duplicates possible
        fv = extract_features(resample)
        result = classify_by_nearest_centroid(fv, reference_features)
        label_counts[result["predicted_label"]] += 1

    report = {"n_full": n_full, "n_subsample": n_sub, "fraction": fraction,
              "n_resamples": n_resamples, "label_counts": label_counts, "proportions": {}}
    for label, count in label_counts.items():
        p = count / n_resamples
        lo, hi = wilson_score_interval(count, n_resamples)
        report["proportions"][label] = {"p": p, "ci_95_lo": lo, "ci_95_hi": hi}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_resamples", type=int, default=200)
    parser.add_argument("--fraction", type=float, default=0.8,
                         help="Fraction of each corpus subsampled WITHOUT replacement per trial.")
    args = parser.parse_args()

    print("Building fixed reference civilization centroids...")
    reference_features = build_fixed_reference_centroids()

    corpora_to_test = {
        "large_corpus (indus_website)": load_corpus_csv("data/indus_website_real_corpus.csv"),
        "CISI_corpus": load_corpus_csv("data/cisi_real_corpus.csv"),
    }

    large = load_corpus_csv("data/indus_website_real_corpus.csv")
    filtered_large = large.filter(exclude_damaged=True)
    matched = [ins for ins in filtered_large.inscriptions
               if ins.site == "Mohenjo-daro" and ins.motif.startswith("Bull1")]
    corpora_to_test["Mohenjo-daro_unicorn_matched_subset"] = Corpus(matched)

    print(f"\nRunning subsample-based classification CI "
          f"({args.n_resamples} resamples per corpus, {args.fraction:.0%} of each corpus, "
          f"WITHOUT replacement)...\n")
    print(f"{'Corpus':38s} {'N':>6s}  {'P(language-like)':>17s}  {'95% CI':>18s}")
    print("-" * 85)

    all_results = {}
    for name, corpus in corpora_to_test.items():
        report = subsample_classification(corpus, reference_features, args.n_resamples, args.fraction, seed=42)
        all_results[name] = report
        lang = report["proportions"]["civ_a_language_like"]
        print(f"{name:38s} {report['n_full']:6d}  {lang['p']:17.3f}  "
              f"[{lang['ci_95_lo']:.3f}, {lang['ci_95_hi']:.3f}]")

    print("\nFull breakdown per corpus (all three labels):")
    for name, report in all_results.items():
        print(f"\n{name} (N_full={report['n_full']}, N_subsample={report['n_subsample']}):")
        for label, prop in report["proportions"].items():
            print(f"  {label:28s} P={prop['p']:.3f}  95% CI=[{prop['ci_95_lo']:.3f}, {prop['ci_95_hi']:.3f}]  "
                  f"(counts: {report['label_counts'][label]}/{report['n_resamples']})")

    with open(EXP_DIR / "bootstrap_ci_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'bootstrap_ci_results.json'}")


if __name__ == "__main__":
    main()
