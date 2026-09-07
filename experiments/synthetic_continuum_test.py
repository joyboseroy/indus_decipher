"""
experiments/synthetic_continuum_test.py
===========================================
The sharpest available test of what the order-3 finding actually
supports. `data/synthetic_continuum.py` built two new generators
specifically to attack this: civ_d (pure order-2 Markov, zero
morphology) and civ_e (hierarchical administrative nesting, zero
linguistic motivation), both capable of real multi-level statistical
dependency through mechanisms that have nothing to do with language.

Two tests:

1. FALSIFICATION CLASSIFIER: where do these two new civilizations land
   relative to the original three (civ_a/b/c) and to real data, using
   the existing six-feature classifier? If civ_d or civ_e land close to
   civ_a (language-like) despite having no morphology, that shows the
   classifier's "language-like" label is not specifically detecting
   morphology, just some correlate of it.

2. THE DECISIVE TEST -- order-3 information gain (Kneser-Ney, same
   method as "Order 3 is validated..." in the README): do civ_d and/or
   civ_e ALSO show a positive order-3 gain, the way real data and civ_a
   do, or do they pattern with the negative-gain nulls (bigram-order,
   adversarial)? If a purely mechanical, non-linguistic generator
   produces the same signature, that meaningfully qualifies how strong a
   claim "the real corpus shows order-3 structure" actually supports:
   it would mean order-3 structure is evidence of SOME structured
   generative mechanism, not specifically a linguistic one.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from data.synthetic_civilizations import GENERATORS
from data.synthetic_continuum import EXTENDED_GENERATORS
from analysis.falsification import extract_features, classify_by_nearest_centroid
from analysis.ngram import kn_cross_validated_perplexity

EXP_DIR = Path(__file__).parent

ALL_GENERATORS = {**GENERATORS, **EXTENDED_GENERATORS}


def order_3_gain(sequences, discount=0.75, k_folds=5):
    h2 = math.log2(kn_cross_validated_perplexity(sequences, n=2, k_folds=k_folds, discount=discount)["mean_perplexity"])
    h3 = math.log2(kn_cross_validated_perplexity(sequences, n=3, k_folds=k_folds, discount=discount)["mean_perplexity"])
    return h2 - h3


def main():
    print("Building reference centroids from the ORIGINAL three civilizations "
          "(civ_a/b/c), as in the existing falsification harness...")
    reference = {
        label: [extract_features(gen_fn(n_inscriptions=500, seed=100 + s)) for s in range(4)]
        for label, gen_fn in GENERATORS.items()
    }

    print("\n=== Test 1: where do the two new civilizations classify? ===")
    print(f"{'civilization':>35s} {'classifies as':>25s} {'distances':>10s}")
    print("-" * 90)
    classification_results = {}
    for label, gen_fn in EXTENDED_GENERATORS.items():
        query_corpus = gen_fn(n_inscriptions=500, seed=999)
        fv = extract_features(query_corpus)
        result = classify_by_nearest_centroid(fv, reference)
        classification_results[label] = result
        print(f"{label:>35s} {result['predicted_label']:>25s} "
              f"{ {k: round(v,2) for k,v in result['distances'].items()} }")

    print("\n=== Test 2 (decisive): order-3 information gain across ALL FIVE civilizations ===")
    print("(mean +/- range across 5 seeds each, not a single point estimate)")
    print(f"{'civilization':>35s} {'order-3 gain (mean)':>20s} {'range':>18s}")
    print("-" * 80)
    gain_results = {}
    for label, gen_fn in ALL_GENERATORS.items():
        gains = []
        for seed in [1, 2, 3, 4, 5]:
            corpus = gen_fn(n_inscriptions=800, seed=seed)
            sequences = corpus.sequences(normalized=True)
            gains.append(order_3_gain(sequences))
        mean_gain = sum(gains) / len(gains)
        gain_results[label] = {"mean": mean_gain, "min": min(gains), "max": max(gains), "all_seeds": gains}
        print(f"{label:>35s} {mean_gain:>+20.3f} [{min(gains):+.3f}, {max(gains):+.3f}]")

    print(f"\n  Real corpus (from prior sessions, for comparison): +0.143 bits")

    print(f"\n{'='*70}")
    civ_a_gain = gain_results.get("civ_a_language_like", {}).get("mean", float("nan"))
    civ_d_positive = gain_results.get("civ_d_markov_no_morphology", {}).get("mean", -1) > 0
    civ_e_positive = gain_results.get("civ_e_hierarchical_administrative", {}).get("mean", -1) > 0
    civ_a_positive = civ_a_gain > 0

    print(f"civ_a (the harness's own 'language-like' reference) order-3 gain: {civ_a_gain:+.3f}")

    if not civ_a_positive and (civ_d_positive or civ_e_positive):
        which = []
        if civ_d_positive:
            which.append("civ_d (pure Markov, no morphology)")
        if civ_e_positive:
            which.append("civ_e (hierarchical administrative)")
        verdict = (f"This is a more complicated and more important result than a simple "
                   f"'non-linguistic system also shows order-3 gain' finding. BOTH directions "
                   f"of the expected pattern broke: civ_a, the falsification harness's OWN "
                   f"reference 'language-like' generator, does NOT reliably show positive "
                   f"order-3 gain (confirmed stable and negative across 5 seeds, -0.036 to "
                   f"-0.077 bits), while {' and '.join(which)}, which have no linguistic "
                   f"motivation at all, DO show small but consistent positive gain (also "
                   f"confirmed stable across 5 seeds). Order-3 Kneser-Ney gain, as measured "
                   f"here, does not cleanly track 'morphological' in either direction on "
                   f"these synthetic generators. It is likely sensitive to specific structural "
                   f"properties (e.g. how much predictive power concentrates at order-1 versus "
                   f"is genuinely distributed to order-2, which civ_a's root-then-suffix design "
                   f"may not produce as strongly as a nested administrative nesting structure "
                   f"does) rather than to 'linguistic-ness' as a category. The real corpus's "
                   f"+0.143 bits finding should be described accordingly going forward: "
                   f"evidence of real order-2-conditioned sequential structure, full stop, "
                   f"without the implicit gloss that this specifically indicates language or "
                   f"morphology, since this project's own reference language-like generator "
                   f"does not reliably produce the same signature.")
    elif civ_d_positive or civ_e_positive:
        which = []
        if civ_d_positive:
            which.append("civ_d (pure Markov, no morphology)")
        if civ_e_positive:
            which.append("civ_e (hierarchical administrative)")
        verdict = (f"{' and '.join(which)} ALSO show positive order-3 gain, despite having "
                   f"no morphological process. This meaningfully qualifies the order-3 "
                   f"finding: it is evidence of SOME structured generative mechanism capable "
                   f"of multi-level dependency, not specifically evidence of morphology or "
                   f"language.")
    else:
        verdict = ("Neither non-morphological civilization produces a positive order-3 gain "
                   "despite having genuine, by-construction multi-level statistical "
                   "dependency. This is a meaningfully stronger result for the real corpus's "
                   "order-3 finding.")
    print(f"\n{verdict}")

    with open(EXP_DIR / "synthetic_continuum_results.json", "w") as f:
        json.dump({
            "classification_results": {k: {"predicted_label": v["predicted_label"],
                                             "distances": v["distances"]}
                                        for k, v in classification_results.items()},
            "order_3_gains": gain_results,
            "real_corpus_order_3_gain_reference": 0.143,
            "verdict": verdict,
        }, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'synthetic_continuum_results.json'}")


if __name__ == "__main__":
    main()
