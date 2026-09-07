"""
experiments/tamil_calibration_test.py
=========================================
A third real external-language calibration corpus, alongside ETCSL and
Sanskrit: the Sangam/Old Tamil corpus (see data/convert_tamil_to_csv.py
for full provenance and the methodological limitation specific to this
one). Same two tests, same method, for direct comparability.

IMPORTANT DIFFERENCE FROM ETCSL AND SANSKRIT, expected going in: this
corpus's vocabulary (35,301 distinct orthographic words) exceeds its own
line count (33,711 lines). Unlike ETCSL (33,338 lines, 4,168 lemmas) and
Sanskrit (21,231 sentences, 8,764 lemmas), where the source data provided
real lemmatization, no lemmatized Old Tamil corpus was found. This one
uses raw whitespace-separated orthographic words instead, and Tamil's
agglutinative morphology means most such "words" are functionally unique
inflected forms, not repeated lexical units. The severe resulting
sparsity (more distinct tokens than lines) is expected to make order-3
estimation close to meaningless here, and this script says so explicitly
rather than reporting a number without that context.
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
    tamil = load_corpus_csv("data/tamil_real_corpus.csv").filter(exclude_damaged=True)
    indus_seqs = indus.sequences(normalized=True)
    tamil_seqs_full = tamil.sequences(normalized=True)

    tamil_vocab = len(tamil.vocab())
    print(f"  Indus: {len(indus)} inscriptions, {len(indus.vocab())} signs, "
          f"mean length {sum(len(s) for s in indus_seqs)/len(indus_seqs):.2f}")
    print(f"  Tamil (full): {len(tamil)} lines, {tamil_vocab} distinct words (orthographic, "
          f"NOT lemmatized), mean length {sum(len(s) for s in tamil_seqs_full)/len(tamil_seqs_full):.2f}")
    if tamil_vocab > len(tamil):
        print(f"  WARNING: vocabulary ({tamil_vocab}) exceeds line count ({len(tamil)}). "
              f"Most tokens are near-unique inflected forms; order-3 estimation below is "
              f"expected to be unreliable due to sparsity, not because of anything about "
              f"Tamil as a language. Read the result accordingly.")

    rng = random.Random(0)
    tamil_matched = rng.sample(tamil.inscriptions, len(indus))
    tamil_seqs_matched = [ins.normalized_signs() for ins in tamil_matched]
    print(f"  Tamil (matched to Indus size): {len(tamil_matched)} lines\n")

    print("=== Test 1: order-3 information gain ===")
    print(f"{'corpus':>32s} {'order-3 gain (bits)':>20s}")
    print("-" * 55)
    indus_gain = order_3_gain(indus_seqs)
    print(f"{'Indus (2,543 inscriptions)':>32s} {indus_gain:>+20.3f}")
    tamil_full_gain = order_3_gain(tamil_seqs_full)
    print(f"{'Tamil full (33,711 lines)':>32s} {tamil_full_gain:>+20.3f}")
    tamil_matched_gain = order_3_gain(tamil_seqs_matched)
    print(f"{'Tamil matched (2,543)':>32s} {tamil_matched_gain:>+20.3f}")

    print("\n=== Test 2: WUCS-style network statistics ===")
    tamil_net = compute_network_stats(tamil_seqs_matched, seed=1)
    indus_net = compute_network_stats(indus_seqs, seed=0)
    print(f"{'metric':>30s} {'WUCS':>10s} {'Indus':>10s} {'Tamil':>10s}")
    print("-" * 65)
    rows = ["reciprocity_empirical", "connectivity_empirical", "n_beginners_empirical", "n_enders_empirical"]
    for key in rows:
        w, i, t = WUCS_REFERENCE[key], indus_net[key], tamil_net[key]
        w_str = f"{w:.4f}" if isinstance(w, float) else str(w)
        i_str = f"{i:.4f}" if isinstance(i, float) else str(i)
        t_str = f"{t:.4f}" if isinstance(t, float) else str(t)
        print(f"{key:>30s} {w_str:>10s} {i_str:>10s} {t_str:>10s}")

    print(f"\n{'='*70}")
    verdict = (
        f"Tamil full-scale order-3 gain: {tamil_full_gain:+.3f}; matched-scale: "
        f"{tamil_matched_gain:+.3f}. Given vocabulary ({tamil_vocab}) exceeds line count "
        f"({len(tamil)}) even at full scale, BOTH numbers should be read with real caution "
        f"-- this is a fundamentally sparser regime than ETCSL or Sanskrit had even before "
        f"subsampling, a direct consequence of using unlemmatized orthographic words for an "
        f"agglutinative language, not a finding about Tamil or about the Indus comparison. "
        f"Unlike the ETCSL/Sanskrit results, this one is not offered as informative evidence "
        f"either way; it is reported for completeness and transparency, with the limitation "
        f"stated up front rather than discovered by a careful reader."
    )
    print(verdict)

    with open(EXP_DIR / "tamil_calibration_results.json", "w") as f:
        json.dump({
            "indus": {"order_3_gain": indus_gain, "network_stats": indus_net},
            "tamil_full": {"order_3_gain": tamil_full_gain, "n_lines": len(tamil_seqs_full),
                            "vocab_size": tamil_vocab},
            "tamil_matched": {"order_3_gain": tamil_matched_gain, "network_stats": tamil_net,
                               "n_lines": len(tamil_seqs_matched)},
            "verdict": verdict,
        }, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'tamil_calibration_results.json'}")


if __name__ == "__main__":
    main()
