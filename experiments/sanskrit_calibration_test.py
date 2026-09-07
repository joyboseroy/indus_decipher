"""
experiments/sanskrit_calibration_test.py
============================================
A second real external-language calibration corpus, alongside ETCSL
(experiments/etcsl_calibration_test.py): the Rigveda portion of the
Digital Corpus of Sanskrit (Hellwig, see data/convert_dcs_sanskrit_to_csv.py
for full provenance). Same two tests, same method, for direct
comparability across three real corpora now (Indus, Sumerian, Sanskrit)
plus the previously published WUCS network statistics.

Rigveda sentences (mean 8.0 tokens) are longer than Indus inscriptions
(4.4 signs) or ETCSL lines (4.46 tokens) -- unlike the ETCSL comparison,
this is not a close length match, and is reported as such rather than
adjusted.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
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
    sanskrit = load_corpus_csv("data/dcs_sanskrit_real_corpus.csv").filter(exclude_damaged=True)
    indus_seqs = indus.sequences(normalized=True)
    sanskrit_seqs_full = sanskrit.sequences(normalized=True)

    print(f"  Indus: {len(indus)} inscriptions, {len(indus.vocab())} signs, "
          f"mean length {sum(len(s) for s in indus_seqs)/len(indus_seqs):.2f}")
    print(f"  Sanskrit/Rigveda (full): {len(sanskrit)} sentences, {len(sanskrit.vocab())} lemmas, "
          f"mean length {sum(len(s) for s in sanskrit_seqs_full)/len(sanskrit_seqs_full):.2f}")

    rng = random.Random(0)
    sanskrit_matched = rng.sample(sanskrit.inscriptions, len(indus))
    sanskrit_seqs_matched = [ins.normalized_signs() for ins in sanskrit_matched]
    print(f"  Sanskrit (matched to Indus size): {len(sanskrit_matched)} sentences\n")

    print("=== Test 1: order-3 information gain ===")
    print(f"{'corpus':>32s} {'order-3 gain (bits)':>20s}")
    print("-" * 55)
    indus_gain = order_3_gain(indus_seqs)
    print(f"{'Indus (2,543 inscriptions)':>32s} {indus_gain:>+20.3f}")
    sanskrit_full_gain = order_3_gain(sanskrit_seqs_full)
    print(f"{'Sanskrit full (21,231 sentences)':>32s} {sanskrit_full_gain:>+20.3f}")
    sanskrit_matched_gain = order_3_gain(sanskrit_seqs_matched)
    print(f"{'Sanskrit matched (2,543)':>32s} {sanskrit_matched_gain:>+20.3f}")

    print("\n=== Test 2: WUCS-style network statistics ===")
    sanskrit_net = compute_network_stats(sanskrit_seqs_matched, seed=1)
    indus_net = compute_network_stats(indus_seqs, seed=0)
    print(f"{'metric':>30s} {'WUCS':>10s} {'Indus':>10s} {'Sanskrit':>10s}")
    print("-" * 65)
    rows = ["reciprocity_empirical", "connectivity_empirical", "n_beginners_empirical", "n_enders_empirical"]
    for key in rows:
        w, i, s = WUCS_REFERENCE[key], indus_net[key], sanskrit_net[key]
        w_str = f"{w:.4f}" if isinstance(w, float) else str(w)
        i_str = f"{i:.4f}" if isinstance(i, float) else str(i)
        s_str = f"{s:.4f}" if isinstance(s, float) else str(s)
        print(f"{key:>30s} {w_str:>10s} {i_str:>10s} {s_str:>10s}")

    print(f"\n{'='*70}")
    if sanskrit_matched_gain > 0:
        verdict = (f"Sanskrit, at matched sample size, ALSO shows positive order-3 gain "
                   f"({sanskrit_matched_gain:+.3f} bits). Combined with the ETCSL result, "
                   f"BOTH real languages tested now agree with Indus's direction at matched "
                   f"scale, unlike ETCSL's own earlier matched-size result, which was "
                   f"negative. This means the sample-size sensitivity seen with ETCSL was "
                   f"specific to that corpus/sentence-length combination, not a universal "
                   f"property of the order-3 test at ~2,500 examples.")
    else:
        verdict = (f"Sanskrit, at matched sample size, does NOT show positive order-3 gain "
                   f"({sanskrit_matched_gain:+.3f} bits), the same direction as ETCSL's own "
                   f"matched-size result. Two independent real languages now both fail to "
                   f"show this signal at Indus-corpus scale, strengthening the case that "
                   f"this is a genuine, general sample-size limitation of the order-3 test "
                   f"around 2,500 short sequences, not specific to one corpus.")
    print(verdict)

    with open(EXP_DIR / "sanskrit_calibration_results.json", "w") as f:
        json.dump({
            "indus": {"order_3_gain": indus_gain, "network_stats": indus_net},
            "sanskrit_full": {"order_3_gain": sanskrit_full_gain, "n_sentences": len(sanskrit_seqs_full)},
            "sanskrit_matched": {"order_3_gain": sanskrit_matched_gain, "network_stats": sanskrit_net,
                                  "n_sentences": len(sanskrit_seqs_matched)},
        }, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'sanskrit_calibration_results.json'}")


if __name__ == "__main__":
    main()
