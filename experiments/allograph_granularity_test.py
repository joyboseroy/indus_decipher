"""
experiments/allograph_granularity_test.py
=============================================
Resolves the question left open last session: does the CISI corpus's
allograph-level classification flip (language-like at primary-sign
granularity, mixed at full-allograph granularity) reflect real
information about allograph distinctions, or just sparse-data noise from
naive full fragmentation (every distinct feature combination becomes its
own sign, no minimum-count floor)?

Adds a third, middle representation: `data/convert_cisi_to_csv.py`'s
"hierarchical" granularity, which only splits an allograph out from its
primary sign when that specific allograph is independently attested at
least `min_count` times CORPUS-WIDE (default 3), collapsing rarer
variants back to the primary sign rather than treating a single
annotation as evidence of a functionally distinct sign.

Compares classification AND the validated order-3 information-gain test
across all three representations (primary, hierarchical, full
allograph). If the hierarchical (conservative) representation agrees
with primary, the full-allograph flip was likely sparsity noise. If
hierarchical agrees with full allograph instead, the flip reflects a
real consequence of allograph treatment, not an artifact of
over-fragmentation, since the hierarchical threshold specifically rules
out rare-variant noise as the explanation.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from analysis.falsification import extract_features, classify_by_nearest_centroid
from analysis.ngram import kn_cross_validated_perplexity
from data.synthetic_civilizations import GENERATORS

EXP_DIR = Path(__file__).parent

REPRESENTATIONS = {
    "primary": "data/cisi_real_corpus.csv",
    "hierarchical (min_count=3)": "data/cisi_real_corpus_hierarchical.csv",
    "full_allograph": "data/cisi_real_corpus_allograph.csv",
}


def main():
    print("Building reference civilization centroids...")
    reference_features = {
        label: [extract_features(gen_fn(n_inscriptions=500, seed=100 + s)) for s in range(4)]
        for label, gen_fn in GENERATORS.items()
    }

    results = {}
    print(f"\n{'representation':30s} {'N signs':>8s} {'top final':>10s} {'classification':>20s} {'order-3 gain':>13s}")
    print("-" * 90)
    for label, path in REPRESENTATIONS.items():
        corpus = load_corpus_csv(path).filter(exclude_damaged=True)
        vocab_size = len(corpus.vocab())

        fv = extract_features(corpus)
        result = classify_by_nearest_centroid(fv, reference_features)

        seqs = corpus.sequences(normalized=True)
        import math
        h2 = math.log2(kn_cross_validated_perplexity(seqs, n=2, k_folds=5)["mean_perplexity"])
        h3 = math.log2(kn_cross_validated_perplexity(seqs, n=3, k_folds=5)["mean_perplexity"])
        gain_3 = h2 - h3

        results[label] = {
            "n_signs": vocab_size, "top_sign_final_share": fv.top_sign_final_share,
            "classification": result["predicted_label"], "distances": result["distances"],
            "order_3_gain": gain_3,
        }
        print(f"{label:30s} {vocab_size:8d} {fv.top_sign_final_share:10.4f} "
              f"{result['predicted_label']:>20s} {gain_3:+13.3f}")

    prim = results["primary"]
    hier = results["hierarchical (min_count=3)"]
    full = results["full_allograph"]
    print(f"\n{'='*70}")
    if hier["classification"] == full["classification"] and hier["classification"] != prim["classification"]:
        verdict = ("Hierarchical agrees with full allograph, not primary, DESPITE only "
                   "splitting independently well-attested allographs. This is evidence the "
                   "classification flip reflects a real consequence of treating visually "
                   "distinct allograph forms as distinct signs, not sparse-data noise from "
                   "naive full fragmentation -- the conservative threshold specifically "
                   "rules out the noise explanation and the flip persists anyway.")
    elif hier["classification"] == prim["classification"]:
        verdict = ("Hierarchical agrees with primary, not full allograph. This supports the "
                   "sparsity-artifact explanation: once rare, poorly-attested allograph "
                   "splits are removed, the classification reverts to what primary-level "
                   "granularity already showed.")
    else:
        verdict = "No clean two-out-of-three agreement; report the raw numbers above."
    print(verdict)

    with open(EXP_DIR / "allograph_granularity_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'allograph_granularity_test_results.json'}")


if __name__ == "__main__":
    main()
