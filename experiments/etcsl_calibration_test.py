"""
experiments/etcsl_calibration_test.py
=========================================
The real external-language calibration this project has been missing:
not a synthetic control, but genuine attested Sumerian text (ETCSL,
see data/convert_etcsl_to_csv.py for full provenance and methodological
notes), run through the identical toolkit used throughout this project.

Two comparisons:
  1. The order-3 Kneser-Ney information gain test (same method as
     README's "Order 3 is validated..." and "The synthetic continuum"),
     run on ETCSL at both full scale (33,338 lines) and at a random
     subsample matched to the Indus corpus's own size (2,543), since
     sample size has repeatedly mattered in this project's own findings
     and a fair comparison needs matched scale.
  2. The WUCS-style network statistics (reciprocity, connectivity,
     beginner/ender counts) from experiments/wucs_comparison_test.py,
     run on ETCSL the same way, adding a THIRD independent data point
     (Indus corpus, WUCS corpus, now genuine Sumerian) to that
     comparison.

The question this answers that no synthetic control can: where does the
real Indus corpus's order-3 signal sit relative to a real, unambiguously
linguistic system of comparable inscription length? Real Sumerian lines
average 4.46 tokens; Indus inscriptions average 4.44 signs -- an
unusually close length match for this kind of cross-system comparison.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv, Corpus
from analysis.ngram import kn_cross_validated_perplexity
from experiments.wucs_comparison_test import compute_network_stats, WUCS_REFERENCE

EXP_DIR = Path(__file__).parent


def order_3_gain(sequences, discount=0.75, k_folds=5):
    h2 = math.log2(kn_cross_validated_perplexity(sequences, n=2, k_folds=k_folds, discount=discount)["mean_perplexity"])
    h3 = math.log2(kn_cross_validated_perplexity(sequences, n=3, k_folds=k_folds, discount=discount)["mean_perplexity"])
    return h2 - h3


def main():
    print("Loading corpora...")
    indus = load_corpus_csv("data/indus_website_real_corpus.csv").filter(exclude_damaged=True)
    etcsl = load_corpus_csv("data/etcsl_real_corpus.csv").filter(exclude_damaged=True)
    indus_seqs = indus.sequences(normalized=True)
    etcsl_seqs_full = etcsl.sequences(normalized=True)

    print(f"  Indus: {len(indus)} inscriptions, {len(indus.vocab())} signs, "
          f"mean length {sum(len(s) for s in indus_seqs)/len(indus_seqs):.2f}")
    print(f"  ETCSL (full): {len(etcsl)} lines, {len(etcsl.vocab())} lemmas, "
          f"mean length {sum(len(s) for s in etcsl_seqs_full)/len(etcsl_seqs_full):.2f}")

    rng = random.Random(0)
    etcsl_matched = rng.sample(etcsl.inscriptions, len(indus))
    etcsl_seqs_matched = [ins.normalized_signs() for ins in etcsl_matched]
    print(f"  ETCSL (matched to Indus size): {len(etcsl_matched)} lines\n")

    print("=== Test 1: order-3 information gain ===")
    print(f"{'corpus':>30s} {'order-3 gain (bits)':>20s}")
    print("-" * 55)
    indus_gain = order_3_gain(indus_seqs)
    print(f"{'Indus (2,543 inscriptions)':>30s} {indus_gain:>+20.3f}")
    etcsl_full_gain = order_3_gain(etcsl_seqs_full)
    print(f"{'ETCSL full (33,338 lines)':>30s} {etcsl_full_gain:>+20.3f}")
    etcsl_matched_gain = order_3_gain(etcsl_seqs_matched)
    print(f"{'ETCSL matched (2,543 lines)':>30s} {etcsl_matched_gain:>+20.3f}")

    print("\n=== Test 2: WUCS-style network statistics, now with genuine Sumerian added ===")
    etcsl_net = compute_network_stats(etcsl_seqs_matched, seed=1)
    print(f"{'metric':>30s} {'WUCS (published)':>18s} {'Indus (ours)':>14s} {'ETCSL (Sumerian)':>18s}")
    print("-" * 85)
    indus_net = compute_network_stats(indus_seqs, seed=0)
    rows = ["reciprocity_empirical", "connectivity_empirical", "n_beginners_empirical", "n_enders_empirical"]
    for key in rows:
        w = WUCS_REFERENCE[key]
        i = indus_net[key]
        e = etcsl_net[key]
        w_str = f"{w:.4f}" if isinstance(w, float) else str(w)
        i_str = f"{i:.4f}" if isinstance(i, float) else str(i)
        e_str = f"{e:.4f}" if isinstance(e, float) else str(e)
        print(f"{key:>30s} {w_str:>18s} {i_str:>14s} {e_str:>18s}")

    print(f"\n{'='*70}")
    if etcsl_matched_gain > 0:
        verdict = (
            f"Real Sumerian, at matched sample size, ALSO shows a positive order-3 gain "
            f"({etcsl_matched_gain:+.3f} bits), the same direction as the Indus corpus "
            f"({indus_gain:+.3f} bits) and unlike this project's own civ_a synthetic "
            f"'language-like' generator (which does not). This is the first genuine "
            f"positive control this project has had for the order-3 result: an "
            f"unambiguously linguistic system of closely matched inscription length "
            f"shows the same qualitative signature Indus does."
        )
    else:
        verdict = (
            f"Real Sumerian, at matched sample size, does NOT show a positive order-3 "
            f"gain ({etcsl_matched_gain:+.3f} bits) despite being an unambiguously "
            f"linguistic system. This considerably complicates the order-3 finding: if "
            f"genuine attested language doesn't reliably produce this signature at this "
            f"scale either, order-3 gain is even less specifically diagnostic of "
            f"'linguistic' than the synthetic continuum test already suggested."
        )
    print(verdict)

    with open(EXP_DIR / "etcsl_calibration_results.json", "w") as f:
        json.dump({
            "indus": {"order_3_gain": indus_gain, "network_stats": indus_net},
            "etcsl_full": {"order_3_gain": etcsl_full_gain, "n_lines": len(etcsl_seqs_full)},
            "etcsl_matched": {"order_3_gain": etcsl_matched_gain, "network_stats": etcsl_net,
                               "n_lines": len(etcsl_seqs_matched)},
            "wucs_reference": WUCS_REFERENCE,
        }, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'etcsl_calibration_results.json'}")


if __name__ == "__main__":
    main()
