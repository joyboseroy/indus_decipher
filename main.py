"""
main.py
========
End-to-end demonstration pipeline. Run with:

    python3 main.py                          # uses the built-in synthetic corpus
    python3 main.py --csv path/to/real.csv   # uses real data via data/loader.py

This runs, in order:
  1. Load corpus, print summary statistics
  2. Unigram / Zipf-Mandelbrot fit + positional (initial/final) asymmetry
  3. Bigram/trigram log-likelihood significant pairs
  4. Cross-validated n-gram perplexity (unigram..quadrigram) + restoration accuracy
  5. Conditional entropy vs. synthetic random/rigid controls
  6. Train the small NumPy masked-sign transformer, compare accuracy to n-grams

Outputs a text report and a few plots to ./outputs/.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import load_corpus_csv, Corpus
from data.synthetic_corpus import generate_synthetic_corpus
from analysis.positional import fit_zipf_mandelbrot, positional_asymmetry_report, doubling_rate
from analysis.ngram import (log_likelihood_ratio, cross_validated_perplexity,
                             restoration_accuracy, restoration_accuracy_top90mass)
from analysis.entropy import compare_against_controls, entropy_by_sequence_length
from analysis.minimal_pairs import analyze_minimal_pairs
from analysis.falsification import extract_features, classify_by_nearest_centroid, leave_one_out_accuracy
from analysis.direction_test import test_reading_direction
from data.synthetic_civilizations import GENERATORS
from models.transformer_mlm import Vocab, MaskedSignTransformer, train_mlm, evaluate_mlm_accuracy

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                         help="Path to a real corpus CSV (see data/loader.py schema). "
                              "If omitted, uses the built-in synthetic demo corpus.")
    parser.add_argument("--n_inscriptions", type=int, default=2000)
    parser.add_argument("--extended", action="store_true",
                         help="Also run minimal-pair mining and the "
                              "synthetic-civilization falsification harness. "
                              "Slower (mines all same-length pairs + trains "
                              "3x4 reference corpora).")
    args = parser.parse_args()

    report = {}

    # 1. Load
    if args.csv:
        corpus: Corpus = load_corpus_csv(args.csv)
        print(f"Loaded real corpus from {args.csv}")
    else:
        corpus = generate_synthetic_corpus(n_inscriptions=args.n_inscriptions)
        print("No --csv given: using the built-in SYNTHETIC demo corpus "
              "(see data/synthetic_corpus.py). Results below are NOT about "
              "the real Indus script.")

    filtered = corpus.filter(exclude_damaged=True)
    sequences = filtered.sequences(normalized=True)
    report["corpus_summary"] = filtered.summary()
    print("\n=== Corpus summary ===")
    print(json.dumps(report["corpus_summary"], indent=2))

    # 2. Zipf + positional
    zipf = fit_zipf_mandelbrot(sequences)
    report["zipf_fit"] = {"C": zipf.C, "q": zipf.q, "gamma": zipf.gamma, "r2": zipf.r_squared()}
    print("\n=== Zipf-Mandelbrot fit ===")
    print(json.dumps(report["zipf_fit"], indent=2))

    plt.figure(figsize=(6, 4))
    plt.loglog(zipf.ranks, zipf.freqs, "o", ms=3, label="observed")
    plt.loglog(zipf.ranks, zipf.predicted, "-", label="Zipf-Mandelbrot fit")
    plt.xlabel("sign rank"); plt.ylabel("frequency"); plt.legend()
    plt.title("Unigram rank-frequency distribution")
    plt.tight_layout(); plt.savefig(OUT_DIR / "zipf_fit.png", dpi=130); plt.close()

    pos_report = positional_asymmetry_report(sequences, top_n=8)
    report["positional_asymmetry"] = pos_report
    print("\n=== Positional asymmetry (top signs) ===")
    for row in pos_report["top_signs"]:
        print(f"  {row['sign']}: overall={row['overall_share']:.3f} "
              f"initial={row['initial_share']:.3f} final={row['final_share']:.3f}")

    # Reading-direction diagnostic: does the published Rao/Yadav fingerprint
    # (final position more constrained than initial) hold as-stored, or only
    # when reversed? See analysis/direction_test.py docstring for caveats.
    dir_result = test_reading_direction(filtered)
    report["direction_diagnostic"] = {
        "as_stored": dir_result.as_stored,
        "reversed": dir_result.reversed_,
        "likely_direction": dir_result.likely_direction,
        "note": dir_result.note,
    }
    print("\n=== Reading-direction diagnostic ===")
    print(f"  As-stored:  H_initial={dir_result.as_stored['h_initial']:.3f} bits, "
          f"H_final={dir_result.as_stored['h_final']:.3f} bits, "
          f"gap(final more constrained if >0)={dir_result.as_stored['final_minus_initial_gap']:+.3f}, "
          f"cond.entropy={dir_result.as_stored['conditional_entropy']:.3f} bits")
    print(f"  Reversed:   H_initial={dir_result.reversed_['h_initial']:.3f} bits, "
          f"H_final={dir_result.reversed_['h_final']:.3f} bits, "
          f"gap={dir_result.reversed_['final_minus_initial_gap']:+.3f}, "
          f"cond.entropy={dir_result.reversed_['conditional_entropy']:.3f} bits")
    print(f"  Likely direction (heuristic): {dir_result.likely_direction}")
    print(f"  {dir_result.note}")

    # doubling behaviour of the most frequent sign
    most_frequent_sign = pos_report["top_signs"][0]["sign"]
    dbl = doubling_rate(sequences, most_frequent_sign)
    report["doubling_rate_most_frequent_sign"] = dbl
    print(f"\nDoubling behaviour of most frequent sign ({most_frequent_sign}): "
          f"{dbl['doubled_occurrences']}/{dbl['total_occurrences']} "
          f"({dbl['doubling_rate']:.1%})")

    # 3. Significant bigrams
    sig_pairs = log_likelihood_ratio(sequences, min_count=3)[:10]
    report["top_significant_bigrams"] = sig_pairs
    print("\n=== Top significant bigrams (log-likelihood ratio) ===")
    for p in sig_pairs:
        print(f"  {p['sign_a']} -> {p['sign_b']}: count={p['count']}, "
              f"P(b|a)={p['p_b_given_a']:.2f}, G2={p['g2']:.1f}")

    # 4. n-gram cross-validated perplexity + restoration accuracy
    print("\n=== Cross-validated perplexity by n-gram order ===")
    for n in (1, 2, 3, 4):
        cv = cross_validated_perplexity(sequences, n=n, k_folds=5)
        report.setdefault("perplexity_by_n", {})[n] = cv["mean_perplexity"]
        print(f"  n={n}: mean perplexity = {cv['mean_perplexity']:.2f}")

    acc = restoration_accuracy(sequences, n=2, trials=300)
    report["ngram_restoration_accuracy_strict_top1"] = acc
    print(f"\nBigram-model masked-sign restoration accuracy (strict top-1): {acc:.1%}")

    top90 = restoration_accuracy_top90mass(sequences, n=2, trials=300)
    report["ngram_restoration_accuracy_top90mass"] = top90
    print(f"Bigram-model masked-sign restoration accuracy (top-90%-cumulative-mass, "
          f"matching Yadav et al. 2010's actual methodology): {top90['accuracy']:.1%} "
          f"(their published figure: ~75%; mean candidate-set size here: "
          f"{top90['mean_candidate_set_size']:.1f} of {top90['vocab_size']} signs -- "
          f"a large set size alongside a high accuracy means this metric is doing "
          f"little discriminating work, so read both numbers together)")

    # 5. Conditional entropy vs controls
    ent = compare_against_controls(sequences)
    report["entropy_vs_controls"] = ent
    print("\n=== Conditional entropy vs. synthetic controls ===")
    print(json.dumps(ent, indent=2))

    ent_by_len = entropy_by_sequence_length(sequences, max_len=6)
    plt.figure(figsize=(6, 4))
    plt.plot(list(ent_by_len.keys()), list(ent_by_len.values()), "o-")
    plt.xlabel("sequence window length"); plt.ylabel("conditional entropy (bits)")
    plt.title("Conditional entropy vs. sequence length")
    plt.tight_layout(); plt.savefig(OUT_DIR / "entropy_by_length.png", dpi=130); plt.close()

    # 6. Transformer masked-sign model vs n-gram baseline
    print("\n=== Training small NumPy masked-sign transformer ===")
    vocab = Vocab(sequences)
    model = MaskedSignTransformer(vocab_size=vocab.size, d_model=48, max_len=32)
    losses = train_mlm(model, vocab, sequences, epochs=60, lr=0.03)
    mlm_acc = evaluate_mlm_accuracy(model, vocab, sequences, trials=300)
    report["transformer_training_losses"] = losses
    report["transformer_mlm_accuracy"] = mlm_acc
    print(f"  final training loss: {losses[-1]:.3f}")
    print(f"  transformer masked-sign accuracy: {mlm_acc:.1%} "
          f"(bigram baseline was {acc:.1%})")

    plt.figure(figsize=(6, 4))
    plt.plot(losses, "-o", ms=3)
    plt.xlabel("epoch"); plt.ylabel("mean cross-entropy loss")
    plt.title("Masked-sign transformer training loss")
    plt.tight_layout(); plt.savefig(OUT_DIR / "transformer_training_loss.png", dpi=130); plt.close()

    # 7. (optional) minimal-pair mining + falsification harness
    if args.extended:
        print("\n=== Minimal-pair ('seal-twin') mining ===")
        mp_report = analyze_minimal_pairs(filtered, min_edge_weight=1, top_n_classes=6)
        report["minimal_pairs"] = {
            "n_total_pairs": mp_report.n_total_pairs,
            "n_corroborated_context_pairs": mp_report.n_corroborated_context_pairs,
            "n_corroborated_motif_pairs": mp_report.n_corroborated_motif_pairs,
            "n_inscriptions_with_known_motif": mp_report.n_inscriptions_with_known_motif,
            "n_paradigm_classes": mp_report.n_paradigm_classes,
            "n_paradigm_classes_motif_tier": mp_report.n_paradigm_classes_motif_tier,
            "position_distribution": mp_report.position_distribution,
            "largest_classes": mp_report.largest_classes,
            "largest_classes_motif_tier": mp_report.largest_classes_motif_tier,
        }
        print(f"  {mp_report.n_total_pairs} minimal pairs found "
              f"({mp_report.n_corroborated_context_pairs} corroborated by matching site+object_type)")
        print(f"  {mp_report.n_inscriptions_with_known_motif} of {len(filtered)} inscriptions "
              f"have a known iconographic motif")
        print(f"  {mp_report.n_corroborated_motif_pairs} pairs additionally corroborated by "
              f"matching motif (site + object_type + iconography all agree) -- the strongest tier")
        print(f"  {mp_report.n_paradigm_classes} candidate paradigm classes (all pairs); "
              f"{mp_report.n_paradigm_classes_motif_tier} classes from motif-corroborated pairs only")
        if mp_report.largest_classes_motif_tier:
            print("  Largest motif-corroborated classes:")
            for cls in mp_report.largest_classes_motif_tier[:3]:
                print(f"    size={cls['size']} signs={cls['signs'][:8]}"
                      f"{'...' if len(cls['signs']) > 8 else ''}")
        else:
            print("  No motif-corroborated minimal pairs found -- either this corpus "
                  "lacks motif/iconography metadata, or none of the same-motif same-length "
                  "inscriptions happen to differ at exactly one position.")
        print("  Note: raw connected components at weight=1 tend to merge into a "
              "few large blobs; try higher min_edge_weight for finer structure "
              "(see analysis/minimal_pairs.py).")

        print("\n=== Falsification harness: synthetic civilizations ===")
        print("Building reference corpora for 3 known generative systems "
              "(language-like / administrative-code / mixed)...")
        corpora_by_label = {
            label: [gen_fn(n_inscriptions=500, seed=100 + s) for s in range(4)]
            for label, gen_fn in GENERATORS.items()
        }
        loo = leave_one_out_accuracy(corpora_by_label)
        report["falsification_harness_self_test"] = {
            "overall_accuracy": loo["overall_accuracy"],
            "per_label_accuracy": loo["per_label_accuracy"],
        }
        print(f"  Self-test (can the pipeline tell 3 KNOWN generators apart?): "
              f"{loo['overall_accuracy']:.1%} accuracy "
              f"(chance = 33.3%)")

        # classify the loaded/synthetic-demo corpus against the reference set
        reference_features = {
            label: [extract_features(c) for c in corpora] for label, corpora in corpora_by_label.items()
        }
        query_features = extract_features(filtered)
        classification = classify_by_nearest_centroid(query_features, reference_features)
        report["target_corpus_classification"] = classification
        print(f"\n  Loaded corpus statistically resembles: {classification['predicted_label']}")
        print(f"  (distances to each reference centroid -- smaller is closer):")
        for label, d in sorted(classification["distances"].items(), key=lambda kv: kv[1]):
            print(f"    {label}: {d:.2f}")
        print("  CAVEAT: 'resembles most' is a statistical nearest-neighbor "
              "result against 3 specific synthetic generators, not a proof "
              "about what produced the loaded corpus.")

    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {OUT_DIR / 'report.json'}")
    print(f"Plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
