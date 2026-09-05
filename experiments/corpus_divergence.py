"""
experiments/corpus_divergence.py
===================================
Investigates the open question in README.md: why does the large real
corpus (data/indus_website_real_corpus.csv, 2,543 inscriptions) classify
as `civ_a_language_like` in the falsification harness, while the small
CISI corpus (data/cisi_real_corpus.csv, 179 inscriptions, all
unicorn-motif Mohenjo-daro seals) classifies as `civ_c_mixed`?

Two experiments, run against a single FIXED set of reference civilization
centroids (built once, seed-locked) so every comparison in this file uses
the same yardstick:

Experiment 1 -- sample-size curve. Randomly subsample the large corpus at
increasing N (matching CISI's size and several points above and below
it), classify each subsample, and report P(classified as language-like)
with a binomial standard error at each N. If CISI's "mixed" result is
just small-sample instability, small subsamples of the LARGE corpus
should show a similarly elevated rate of landing on "mixed" or
"administrative", converging toward the large corpus's own answer as N
grows.

Experiment 2 -- matched subsample. Filter the large corpus to
site == "Mohenjo-daro" AND motif starting with "Bull1" (this scheme's
field-symbol code for the "unicorn" bull, the single most common Indus
seal motif in the literature -- see the count check in the module-level
comment below), producing a subset compositionally similar to CISI, then
classify it and compare against random (unrestricted) subsamples of the
same size. If the matched subsample behaves differently from same-size
random subsamples, that's evidence the site/motif restriction itself
matters, not just the sample size.

Both experiments write their results to experiments/divergence_results.json
and a plot to outputs/divergence_sample_size_curve.png.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import Corpus, load_corpus_csv
from data.synthetic_civilizations import GENERATORS
from analysis.falsification import extract_features, classify_by_nearest_centroid

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
EXP_DIR = Path(__file__).parent
LARGE_CORPUS_PATH = Path(__file__).parent.parent / "data" / "indus_website_real_corpus.csv"
CISI_CORPUS_PATH = Path(__file__).parent.parent / "data" / "cisi_real_corpus.csv"


def build_fixed_reference_centroids(seed: int = 100, n_instances: int = 4, n_per_instance: int = 500):
    """Built once and reused for every classification in this file, so
    every result here is measured against the identical yardstick -- not
    rebuilt per sample, which would let noise in the reference set itself
    contaminate the comparison across sample sizes."""
    reference_features = {}
    for label, gen_fn in GENERATORS.items():
        reference_features[label] = [
            extract_features(gen_fn(n_inscriptions=n_per_instance, seed=seed + s))
            for s in range(n_instances)
        ]
    return reference_features


def experiment_1_sample_size_curve(large_corpus: Corpus, reference_features: dict,
                                     sample_sizes: list[int], repeats: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    all_inscriptions = large_corpus.filter(exclude_damaged=True).inscriptions
    results = {}
    for n in sample_sizes:
        n_eff = min(n, len(all_inscriptions))
        label_counts = {label: 0 for label in reference_features}
        trials_run = 0
        for _ in range(repeats):
            sample = rng.sample(all_inscriptions, n_eff)
            sub_corpus = Corpus(sample)
            fv = extract_features(sub_corpus)
            result = classify_by_nearest_centroid(fv, reference_features)
            label_counts[result["predicted_label"]] += 1
            trials_run += 1
        p_language = label_counts.get("civ_a_language_like", 0) / trials_run
        se = (p_language * (1 - p_language) / trials_run) ** 0.5
        results[n] = {
            "n_effective": n_eff, "trials": trials_run,
            "p_language_like": p_language, "se": se,
            "label_counts": label_counts,
        }
        print(f"  N={n_eff:5d}  P(language-like)={p_language:.2f} +/- {se:.2f}  "
              f"(counts: {label_counts})")
    return results


def experiment_2_matched_subsample(large_corpus: Corpus, cisi_corpus: Corpus,
                                     reference_features: dict, repeats: int, seed: int = 1) -> dict:
    rng = random.Random(seed)
    filtered = large_corpus.filter(exclude_damaged=True)

    matched = [ins for ins in filtered.inscriptions
               if ins.site == "Mohenjo-daro" and ins.motif.startswith("Bull1")]
    n_matched = len(matched)
    print(f"  Matched subset (Mohenjo-daro + unicorn/Bull1 motif): {n_matched} inscriptions")

    matched_fv = extract_features(Corpus(matched))
    matched_result = classify_by_nearest_centroid(matched_fv, reference_features)
    print(f"  Matched subset classifies as: {matched_result['predicted_label']} "
          f"(distances: {matched_result['distances']})")

    # random (unrestricted) subsamples of the SAME size, for comparison
    all_inscriptions = filtered.inscriptions
    label_counts = {label: 0 for label in reference_features}
    for _ in range(repeats):
        sample = rng.sample(all_inscriptions, min(n_matched, len(all_inscriptions)))
        fv = extract_features(Corpus(sample))
        result = classify_by_nearest_centroid(fv, reference_features)
        label_counts[result["predicted_label"]] += 1
    p_language_random = label_counts.get("civ_a_language_like", 0) / repeats
    print(f"  Random same-size ({n_matched}) subsamples: P(language-like)="
          f"{p_language_random:.2f} (counts: {label_counts})")

    cisi_fv = extract_features(cisi_corpus.filter(exclude_damaged=True))
    cisi_result = classify_by_nearest_centroid(cisi_fv, reference_features)
    print(f"  CISI corpus itself classifies as: {cisi_result['predicted_label']} "
          f"(distances: {cisi_result['distances']})")

    return {
        "n_matched": n_matched,
        "matched_subset_classification": matched_result,
        "random_same_size_label_counts": label_counts,
        "random_same_size_p_language_like": p_language_random,
        "cisi_classification": cisi_result,
    }


def main():
    print("Loading corpora...")
    large_corpus = load_corpus_csv(LARGE_CORPUS_PATH)
    cisi_corpus = load_corpus_csv(CISI_CORPUS_PATH)

    print("\nBuilding fixed reference civilization centroids (seed-locked, reused throughout)...")
    reference_features = build_fixed_reference_centroids()

    print("\n=== Experiment 1: sample-size curve ===")
    sample_sizes = [50, 75, 100, 104, 150, 179, 250, 500, 1000, 1500, 2000, 2543]
    exp1_results = experiment_1_sample_size_curve(
        large_corpus, reference_features, sample_sizes, repeats=30)

    print("\n=== Experiment 2: matched subsample (Mohenjo-daro + unicorn motif) ===")
    exp2_results = experiment_2_matched_subsample(
        large_corpus, cisi_corpus, reference_features, repeats=30)

    # plot experiment 1
    ns = sorted(exp1_results.keys())
    ps = [exp1_results[n]["p_language_like"] for n in ns]
    ses = [exp1_results[n]["se"] for n in ns]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(ns, ps, yerr=ses, marker="o", capsize=3)
    plt.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="chance (3 classes)")
    plt.axvline(179, color="red", linestyle=":", linewidth=1, label="CISI corpus size (N=179)")
    plt.xscale("log")
    plt.xlabel("subsample size N (log scale)")
    plt.ylabel("P(classified as civ_a_language_like)")
    plt.title("Falsification-harness classification vs. subsample size\n(large real corpus, random subsamples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "divergence_sample_size_curve.png", dpi=130)
    plt.close()

    with open(EXP_DIR / "divergence_results.json", "w") as f:
        json.dump({
            "experiment_1_sample_size_curve": exp1_results,
            "experiment_2_matched_subsample": exp2_results,
        }, f, indent=2, default=str)

    print(f"\nResults written to {EXP_DIR / 'divergence_results.json'}")
    print(f"Plot written to {OUT_DIR / 'divergence_sample_size_curve.png'}")


if __name__ == "__main__":
    main()
