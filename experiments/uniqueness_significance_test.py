"""
experiments/uniqueness_significance_test.py
================================================
Formalizes the ad-hoc check run against the Kriger/Hunt "98.3% sequence
uniqueness" claim into a proper, reusable experiment, and extends it
properly with a null DISTRIBUTION rather than a single comparison point.

The registration-code hypothesis's strongest empirical claim is that
whole-sequence uniqueness is high enough to indicate deliberately
distinct identifiers. That claim is only as strong as its null: at a
given vocabulary size and length distribution, near-total uniqueness can
be close to mechanically inevitable (birthday-paradox territory) even
with zero real sequential structure. This project's own adversarial
null model (matched length distribution, matched initial/final/overall
sign marginals, zero real dependency) is the right instrument to test
this properly: generate MANY null corpora, build a distribution of
uniqueness rates under "no real structure, just this vocabulary and
these marginals," and see where the real corpus's own uniqueness rate
falls in that distribution.

Two versions of uniqueness are tested:
  - WHOLE-CORPUS uniqueness (naive, matches how the claim is usually
    stated): fraction of all sequences that are unique, regardless of
    length. This is the weaker, more easily inflated version, since
    comparing sequences of different lengths is trivially "unique."
  - LENGTH-STRATIFIED uniqueness: uniqueness computed separately within
    each length bucket, then averaged, matching how a real registration
    system would actually need to be non-repeating (two IDs of
    different digit-counts are not usefully "different" the way two
    same-length IDs sharing no digits are).
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from data.adversarial_null_model import generate_matched_null_corpus

EXP_DIR = Path(__file__).parent
N_TRIALS = 200


def whole_corpus_uniqueness(sequences: list[list[str]]) -> float:
    if not sequences:
        return float("nan")
    total = len(sequences)
    unique = len(set(tuple(s) for s in sequences))
    return unique / total


def length_stratified_uniqueness(sequences: list[list[str]]) -> float:
    by_length: dict[int, list[tuple]] = {}
    for s in sequences:
        by_length.setdefault(len(s), []).append(tuple(s))
    weighted_sum = 0.0
    total_n = 0
    for length, seqs in by_length.items():
        n = len(seqs)
        unique = len(set(seqs))
        weighted_sum += unique
        total_n += n
    return weighted_sum / total_n if total_n else float("nan")


def run_for_corpus(corpus_path: str, label: str) -> dict:
    print(f"\n{'='*70}\n{label}: {corpus_path}\n{'='*70}")
    corpus = load_corpus_csv(corpus_path).filter(exclude_damaged=True)
    sequences = corpus.sequences(normalized=True)
    n = len(sequences)
    print(f"  N={n} sequences, {len(corpus.vocab())} distinct signs")

    real_whole = whole_corpus_uniqueness(sequences)
    real_strat = length_stratified_uniqueness(sequences)
    print(f"  Real whole-corpus uniqueness:      {real_whole:.4f}")
    print(f"  Real length-stratified uniqueness: {real_strat:.4f}")

    print(f"  Generating {N_TRIALS} null-model trials (no real dependency, "
          f"matched length/frequency marginals)...")
    null_whole, null_strat = [], []
    for trial in range(N_TRIALS):
        null_corpus = generate_matched_null_corpus(corpus, n_inscriptions=n, seed=trial)
        null_seqs = null_corpus.sequences(normalized=True)
        null_whole.append(whole_corpus_uniqueness(null_seqs))
        null_strat.append(length_stratified_uniqueness(null_seqs))

    def summarize(real_val, null_vals, name):
        mean_null = sum(null_vals) / len(null_vals)
        sd_null = (sum((x - mean_null) ** 2 for x in null_vals) / len(null_vals)) ** 0.5
        n_null_higher_or_equal = sum(1 for x in null_vals if x >= real_val)
        empirical_p = n_null_higher_or_equal / len(null_vals)
        print(f"\n  {name}:")
        print(f"    Real: {real_val:.4f}")
        print(f"    Null distribution: mean={mean_null:.4f}, sd={sd_null:.4f}, "
              f"range=[{min(null_vals):.4f}, {max(null_vals):.4f}]")
        print(f"    Empirical P(null uniqueness >= real): {empirical_p:.3f}")
        if empirical_p > 0.10:
            verdict = ("Real uniqueness is NOT distinguishable from what a zero-dependency "
                       "null with the same vocabulary/length/frequency produces. High "
                       "uniqueness here is consistent with being a mechanical consequence "
                       "of the vocabulary and length regime, not evidence of a deliberate "
                       "identifier system.")
        elif real_val > mean_null:
            verdict = ("Real uniqueness IS higher than the null distribution predicts. "
                       "This is a real signal beyond what vocabulary/length/frequency alone "
                       "would produce -- consistent with (though not proof of) a system "
                       "actively avoiding repetition, as a registration-code hypothesis "
                       "would predict.")
        else:
            verdict = ("Real uniqueness is LOWER than the null distribution predicts -- "
                       "real data repeats MORE than a zero-dependency null would, the "
                       "opposite of what a registration-code hypothesis predicts.")
        print(f"    {verdict}")
        return {"real": real_val, "null_mean": mean_null, "null_sd": sd_null,
                "null_min": min(null_vals), "null_max": max(null_vals),
                "empirical_p_null_gte_real": empirical_p, "verdict": verdict}

    results = {
        "n_sequences": n, "n_signs": len(corpus.vocab()),
        "whole_corpus": summarize(real_whole, null_whole, "Whole-corpus uniqueness"),
        "length_stratified": summarize(real_strat, null_strat, "Length-stratified uniqueness"),
    }
    return results


def main():
    all_results = {}
    all_results["indus_large"] = run_for_corpus("data/indus_website_real_corpus.csv", "Indus (large corpus)")
    all_results["cisi_primary"] = run_for_corpus("data/cisi_real_corpus.csv", "CISI primary (the Kriger/Hunt corpus)")

    print(f"\n{'='*70}")
    print("SUMMARY")
    for key, r in all_results.items():
        ws = r["whole_corpus"]
        ls = r["length_stratified"]
        print(f"\n{key}:")
        print(f"  whole-corpus:      real={ws['real']:.4f} vs null mean={ws['null_mean']:.4f} "
              f"(p={ws['empirical_p_null_gte_real']:.3f})")
        print(f"  length-stratified: real={ls['real']:.4f} vs null mean={ls['null_mean']:.4f} "
              f"(p={ls['empirical_p_null_gte_real']:.3f})")

    with open(EXP_DIR / "uniqueness_significance_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'uniqueness_significance_results.json'}")


if __name__ == "__main__":
    main()
