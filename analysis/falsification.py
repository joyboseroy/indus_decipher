"""
analysis/falsification.py
===========================
"Unit testing for decipherment" (see data/synthetic_civilizations.py):
extract a small feature vector from a corpus using ONLY the statistical
tools already in this repo (Zipf fit, positional asymmetry, conditional
entropy, n-gram perplexity, minimal-pair paradigm classes) and test
whether those features alone can distinguish a language-like generative
system from a non-linguistic administrative code from a mixed system --
on corpora where we, the experimenters, planted the ground truth.

This is the falsification-engine half of the architecture: before trusting
any statistical pattern found in a *real*, label-unknown corpus as
evidence for "this is/isn't language," we should first check whether that
same pattern-finding pipeline can correctly separate known synthetic
generators from each other. If it can't reliably do that, a claim like
"the real corpus's entropy falls in the natural-language range" needs to
be read with real caution.

Usage: run this module directly to print a labeled-classification
accuracy report, then feed any other corpus (including data/loader's real
corpus, or the plain synthetic_corpus.py demo fixture) through
`extract_features` + `classify_by_nearest_centroid` to see which
generative family it statistically resembles most -- with the explicit
caveat that "resembles most" is not "is."
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from data.loader import Corpus
from analysis.positional import fit_zipf_mandelbrot, positional_asymmetry_report
from analysis.entropy import conditional_entropy
from analysis.ngram import cross_validated_perplexity
from analysis.minimal_pairs import analyze_minimal_pairs


FEATURE_NAMES = [
    "zipf_gamma", "zipf_r2", "conditional_entropy",
    "perplexity_ratio_n2_n1", "top_sign_final_share",
    "mean_length", "paradigm_classes_per_1000_inscriptions",
]


@dataclass
class FeatureVector:
    zipf_gamma: float
    zipf_r2: float
    conditional_entropy: float
    perplexity_ratio_n2_n1: float
    top_sign_final_share: float
    mean_length: float
    paradigm_classes_per_1000_inscriptions: float

    def as_list(self) -> list[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]


def extract_features(corpus: Corpus) -> FeatureVector:
    filtered = corpus.filter(exclude_damaged=True)
    sequences = filtered.sequences(normalized=True)

    zipf = fit_zipf_mandelbrot(sequences)
    cond_h = conditional_entropy(sequences)

    ppl1 = cross_validated_perplexity(sequences, n=1, k_folds=3)["mean_perplexity"]
    ppl2 = cross_validated_perplexity(sequences, n=2, k_folds=3)["mean_perplexity"]
    ppl_ratio = ppl2 / ppl1 if ppl1 else float("nan")

    pos = positional_asymmetry_report(sequences, top_n=1)
    top_final_share = pos["top_signs"][0]["final_share"] if pos["top_signs"] else 0.0

    lens = [len(s) for s in sequences]
    mean_len = sum(lens) / len(lens) if lens else 0.0

    mp = analyze_minimal_pairs(filtered, min_edge_weight=1)
    n_ins = len(filtered) or 1
    classes_per_1000 = mp.n_paradigm_classes / n_ins * 1000

    return FeatureVector(
        zipf_gamma=zipf.gamma, zipf_r2=zipf.r_squared(), conditional_entropy=cond_h,
        perplexity_ratio_n2_n1=ppl_ratio, top_sign_final_share=top_final_share,
        mean_length=mean_len, paradigm_classes_per_1000_inscriptions=classes_per_1000,
    )


def _zscore_normalize(vectors: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    n_features = len(vectors[0])
    means = [sum(v[i] for v in vectors) / len(vectors) for i in range(n_features)]
    stds = []
    for i in range(n_features):
        var = sum((v[i] - means[i]) ** 2 for v in vectors) / len(vectors)
        stds.append(math.sqrt(var) or 1.0)
    normalized = [[(v[i] - means[i]) / stds[i] for i in range(n_features)] for v in vectors]
    return normalized, means, stds


def classify_by_nearest_centroid(query: FeatureVector, labeled_examples: dict[str, list[FeatureVector]]) -> dict:
    """Z-score normalize all features jointly (query + every labeled example),
    compute each label's centroid, and return the label whose centroid is
    closest to the query in Euclidean distance -- plus full distances so
    the margin of the decision is visible, not just the winner."""
    labels = []
    raw_vectors = []
    for label, examples in labeled_examples.items():
        for fv in examples:
            labels.append(label)
            raw_vectors.append(fv.as_list())
    query_idx = len(raw_vectors)
    raw_vectors.append(query.as_list())

    normalized, _, _ = _zscore_normalize(raw_vectors)
    query_vec = normalized[query_idx]
    train_vecs = normalized[:query_idx]

    centroids: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for label, vec in zip(labels, train_vecs):
        centroids.setdefault(label, [0.0] * len(vec))
        counts[label] = counts.get(label, 0) + 1
        centroids[label] = [a + b for a, b in zip(centroids[label], vec)]
    for label in centroids:
        centroids[label] = [x / counts[label] for x in centroids[label]]

    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    distances = {label: dist(query_vec, c) for label, c in centroids.items()}
    best_label = min(distances, key=distances.get)
    return {"predicted_label": best_label, "distances": distances}


def leave_one_out_accuracy(corpora_by_label: dict[str, list[Corpus]]) -> dict:
    """For every corpus instance, classify it using centroids built from
    ALL OTHER instances (never itself), and report accuracy per label and
    overall. This is the actual falsification test: can the pipeline tell
    known-different generators apart using only its own statistics?"""
    all_features: dict[str, list[FeatureVector]] = {
        label: [extract_features(c) for c in corpora] for label, corpora in corpora_by_label.items()
    }

    results = []
    for true_label, corpora in corpora_by_label.items():
        for idx in range(len(corpora)):
            query_fv = all_features[true_label][idx]
            train_set = {
                label: [fv for j, fv in enumerate(fvs) if not (label == true_label and j == idx)]
                for label, fvs in all_features.items()
            }
            result = classify_by_nearest_centroid(query_fv, train_set)
            results.append({"true_label": true_label, **result})

    correct = sum(r["true_label"] == r["predicted_label"] for r in results)
    per_label_correct = {}
    per_label_total = {}
    for r in results:
        per_label_total[r["true_label"]] = per_label_total.get(r["true_label"], 0) + 1
        if r["true_label"] == r["predicted_label"]:
            per_label_correct[r["true_label"]] = per_label_correct.get(r["true_label"], 0) + 1

    return {
        "overall_accuracy": correct / len(results) if results else float("nan"),
        "per_label_accuracy": {
            label: per_label_correct.get(label, 0) / per_label_total[label]
            for label in per_label_total
        },
        "n_trials": len(results),
        "detailed_results": results,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.synthetic_civilizations import GENERATORS

    print("Generating multiple instances per civilization (different seeds) "
          "for leave-one-out testing...")
    corpora_by_label = {}
    for label, gen_fn in GENERATORS.items():
        instances = [gen_fn(n_inscriptions=500, seed=100 + s) for s in range(4)]
        corpora_by_label[label] = instances
        print(f"  {label}: {len(instances)} instances of ~500 inscriptions each")

    print("\nRunning leave-one-out classification...")
    report = leave_one_out_accuracy(corpora_by_label)
    print(f"\nOverall accuracy: {report['overall_accuracy']:.1%} "
          f"(chance level for 3 classes = 33.3%)")
    print("Per-label accuracy:")
    for label, acc in report["per_label_accuracy"].items():
        print(f"  {label}: {acc:.1%}")
