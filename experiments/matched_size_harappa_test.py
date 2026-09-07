"""
experiments/matched_size_harappa_test.py
============================================
Directly resolves the question left open across multiple sessions: is
Harappa's weaker substitution-class replication (README's cross-site
stability table, e.g. community 4: 1.00 Jaccard at Mohenjo-daro vs. 0.00
at Harappa) a genuine site-specific difference, or simply because
Mohenjo-daro supplies more motif-labeled data (948) than Harappa (414)?

Repeatedly subsamples Mohenjo-daro DOWN to Harappa's motif-labeled
inscription count (414), reruns the exact same minimal-pair mining and
Jaccard comparison on each subsample (analysis/substitution_graph.py's
new test_class_stability_subsampled), and compares the RESULTING
distribution of Jaccard scores against Harappa's own single observed
score for each of the five largest communities. If Harappa's score falls
within the range that matched-size Mohenjo-daro subsamples produce, that
is direct evidence for the sample-size explanation. If Harappa's score
falls clearly below even the worst matched-size Mohenjo-daro subsamples,
that is evidence of a real site-specific difference, not sample size.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv, Corpus
from analysis.minimal_pairs import find_minimal_pairs
from analysis.substitution_graph import (build_substitution_graph_nx, annotate_distinct_contexts,
                                          modularity_communities, test_class_stability_by_site,
                                          test_class_stability_subsampled)

EXP_DIR = Path(__file__).parent

N_TRIALS = 100


def main():
    print("Loading corpus, discovering communities on the FULL pooled corpus (as in "
          "substitution_graph_analysis.py, for a like-for-like comparison)...")
    corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    filtered = corpus.filter(exclude_damaged=True)

    pairs = find_minimal_pairs(filtered)
    G = build_substitution_graph_nx(pairs, tier="motif")
    annotate_distinct_contexts(G, filtered, pairs)
    communities = modularity_communities(G)

    harappa_motif_count = sum(1 for ins in filtered.inscriptions
                                if ins.site == "Harappa" and ins.motif != "unknown")
    mohenjo_daro_motif_count = sum(1 for ins in filtered.inscriptions
                                     if ins.site == "Mohenjo-daro" and ins.motif != "unknown")
    print(f"  Harappa motif-labeled count: {harappa_motif_count}")
    print(f"  Mohenjo-daro motif-labeled count: {mohenjo_daro_motif_count}")
    print(f"  Subsampling Mohenjo-daro down to Harappa's count, {N_TRIALS} trials per community...\n")

    print(f"{'community':>10s} {'size':>5s} {'Harappa Jaccard':>16s} {'MD-subsampled (mean)':>21s} "
          f"{'MD-subsampled (range)':>24s} {'sample-size explains it?':>26s}")
    print("-" * 110)

    results = []
    for i, community in enumerate(communities[:5]):
        harappa_result = test_class_stability_by_site(filtered, community, "Harappa", tier="motif")
        if not harappa_result.testable:
            print(f"{i:>10d} {len(community):>5d}  not testable at Harappa (too few members in vocab)")
            continue

        md_jaccards = test_class_stability_subsampled(
            filtered, community, source_site="Mohenjo-daro",
            target_size=harappa_motif_count, n_trials=N_TRIALS, tier="motif", seed=i)

        if not md_jaccards:
            print(f"{i:>10d} {len(community):>5d}  matched-size Mohenjo-daro subsampling produced no testable trials")
            continue

        md_mean = sum(md_jaccards) / len(md_jaccards)
        md_min, md_max = min(md_jaccards), max(md_jaccards)
        explains = md_min <= harappa_result.jaccard <= md_max or harappa_result.jaccard >= md_min
        print(f"{i:>10d} {len(community):>5d} {harappa_result.jaccard:>16.3f} {md_mean:>21.3f} "
              f"[{md_min:.2f}, {md_max:.2f}]{'':>10s} {'YES' if explains else 'NO':>26s}")

        results.append({
            "community_index": i, "size": len(community),
            "harappa_jaccard": harappa_result.jaccard,
            "md_subsampled_mean": md_mean, "md_subsampled_min": md_min, "md_subsampled_max": md_max,
            "md_subsampled_n_trials": len(md_jaccards),
            "sample_size_explains": explains,
        })

    n_explained = sum(r["sample_size_explains"] for r in results)
    print(f"\n{'='*70}")
    print(f"{n_explained} of {len(results)} communities: Harappa's Jaccard score falls within "
          f"(or above) the range produced by matched-size Mohenjo-daro subsamples.")

    if n_explained >= len(results) * 0.7:
        verdict = ("Sample size explains most of the apparent Harappa/Mohenjo-daro gap. When "
                   "Mohenjo-daro is subsampled down to Harappa's own motif-labeled count, it "
                   "produces similarly weak Jaccard scores. This resolves the open question: "
                   "the earlier cross-site stability finding was primarily a data-density "
                   "artifact of minimal-pair mining, not evidence of a genuine site-specific "
                   "difference in how signs substitute.")
    elif n_explained <= len(results) * 0.3:
        verdict = ("Sample size does NOT explain most of the gap. Even subsampled down to "
                   "Harappa's own count, Mohenjo-daro still reproduces classes more reliably "
                   "than Harappa's actual data does. This is evidence for a real site-specific "
                   "difference, not just a sample-size artifact -- worth investigating further "
                   "rather than dismissing as a data-density limitation.")
    else:
        verdict = "A mixed result across the five communities; report per-community, not as one finding."
    print(verdict)

    with open(EXP_DIR / "matched_size_harappa_test_results.json", "w") as f:
        json.dump({"harappa_motif_count": harappa_motif_count,
                    "mohenjo_daro_motif_count": mohenjo_daro_motif_count,
                    "n_trials": N_TRIALS, "results": results,
                    "n_explained": n_explained, "verdict": verdict}, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'matched_size_harappa_test_results.json'}")


if __name__ == "__main__":
    main()
