"""
experiments/cross_site_held_out_validation.py
=================================================
The fix for the independence concern raised about
experiments/sign_embeddings_analysis.py: that experiment discovered
substitution communities and evaluated embedding similarity on the SAME
corpus, which isn't circular (minimal pairs and PPMI co-occurrence are
genuinely different signals) but leaves open a real confound -- larger
communities likely skew toward higher-frequency signs, and higher-
frequency signs get less noisy embeddings regardless of any real
functional relationship.

This also happens to be the same fix needed for the long-open Harappa
vs. Mohenjo-daro cross-site stability question (see
experiments/substitution_graph_analysis.py), so this script does both at
once, per the site split proposed in external review:

  1. Discover substitution-graph communities using ONLY Mohenjo-daro
     inscriptions (the corpus this project's substitution work has so
     far been built and reported on).
  2. Train PPMI+SVD embeddings using ONLY Harappa inscriptions -- a
     corpus with no overlap with the discovery step at all, not even
     the same site.
  3. Test whether Mohenjo-daro-discovered communities still show
     elevated internal similarity when measured using Harappa-only
     embeddings, restricted to community members that actually appear
     in Harappa's own (much smaller, ~335-sign) vocabulary.

This is a considerably harder bar than the original same-corpus test.
Genuine loss of signal is expected simply from Harappa's smaller
vocabulary and different corpus; the question is whether ANY real
corroboration survives a fully disjoint discovery/evaluation split, not
whether the original 10/10 result exactly reproduces.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv, Corpus
from analysis.sign_embeddings import train_sign_embeddings
from analysis.minimal_pairs import find_minimal_pairs
from analysis.substitution_graph import build_substitution_graph_nx, annotate_distinct_contexts, modularity_communities
from experiments.sign_embeddings_analysis import mean_pairwise_similarity, random_baseline_similarity

EXP_DIR = Path(__file__).parent


def main():
    print("Loading corpus and splitting by site (no overlap between discovery and evaluation)...")
    large = load_corpus_csv("data/indus_website_real_corpus.csv")
    filtered = large.filter(exclude_damaged=True)

    md_corpus = Corpus([ins for ins in filtered.inscriptions if ins.site == "Mohenjo-daro"])
    hr_corpus = Corpus([ins for ins in filtered.inscriptions if ins.site == "Harappa"])
    print(f"  Discovery corpus (Mohenjo-daro): {len(md_corpus)} inscriptions")
    print(f"  Evaluation corpus (Harappa, fully disjoint): {len(hr_corpus)} inscriptions")

    print("\nDiscovering substitution communities on Mohenjo-daro ONLY...")
    md_pairs = find_minimal_pairs(md_corpus)
    G = build_substitution_graph_nx(md_pairs, tier="motif")
    annotate_distinct_contexts(G, md_corpus, md_pairs)
    communities = modularity_communities(G)
    print(f"  {len(communities)} communities found (compare to 19 found on the pooled full corpus)")

    print("\nTraining PPMI+SVD embeddings on Harappa ONLY (disjoint from discovery)...")
    hr_sequences = hr_corpus.sequences(normalized=True)
    hr_embeddings = train_sign_embeddings(hr_sequences, window=2, n_dims=20)
    hr_vocab = set(hr_embeddings.signs)
    print(f"  {len(hr_vocab)} signs in Harappa's own vocabulary")

    print(f"\n{'community':>10s} {'MD size':>8s} {'in HR vocab':>12s} {'internal sim':>13s} "
          f"{'random baseline':>16s} {'corroborated?':>14s}")
    print("-" * 80)
    results = []
    n_testable = 0
    n_corroborated = 0
    for i, community in enumerate(communities):
        members_in_hr = sorted(s for s in community if s in hr_vocab)
        if len(members_in_hr) < 2:
            print(f"{i:>10d} {len(community):>8d} {len(members_in_hr):>12d}"
                  f"{'  not testable (fewer than 2 members appear in Harappa vocab)':>50s}")
            results.append({"community_index": i, "md_size": len(community),
                             "n_in_harappa_vocab": len(members_in_hr), "testable": False})
            continue
        n_testable += 1
        internal_sim = mean_pairwise_similarity(hr_embeddings, members_in_hr)
        baseline = random_baseline_similarity(hr_embeddings, hr_embeddings.signs, len(members_in_hr))
        corroborated = internal_sim > baseline
        if corroborated:
            n_corroborated += 1
        mark = "YES" if corroborated else "no"
        print(f"{i:>10d} {len(community):>8d} {len(members_in_hr):>12d} {internal_sim:>13.4f} "
              f"{baseline:>16.4f} {mark:>14s}")
        results.append({
            "community_index": i, "md_size": len(community), "n_in_harappa_vocab": len(members_in_hr),
            "testable": True, "internal_similarity": internal_sim,
            "random_baseline": baseline, "corroborated": corroborated,
        })

    print(f"\n{'='*70}")
    print(f"{n_corroborated} of {n_testable} testable communities corroborated under the "
          f"fully disjoint (Mohenjo-daro discovery, Harappa evaluation) split.")
    print(f"(For comparison: the original same-corpus test corroborated 10 of 10.)")

    if n_testable == 0:
        verdict = "No communities had enough members in Harappa's vocabulary to test at all."
    elif n_corroborated / n_testable >= 0.6:
        verdict = ("A majority of testable communities survive the fully disjoint split. This "
                   "is meaningfully stronger evidence than the same-corpus test alone -- these "
                   "communities generalize to an independent site's own distributional "
                   "statistics, not just the corpus they were discovered on.")
    elif n_corroborated / n_testable <= 0.3:
        verdict = ("Most communities do NOT survive the fully disjoint split. This does not "
                   "mean the original same-corpus corroboration was wrong, but it means that "
                   "result should be read as corpus-internal convergence, not as evidence "
                   "these communities generalize across sites. The Harappa/Mohenjo-daro "
                   "cross-site stability question (substitution_graph_analysis.py) and this "
                   "result are likely pointing at the same underlying limitation.")
    else:
        verdict = "A mixed result. Report per-community rather than a single aggregate claim."
    print(f"\n{verdict}")

    with open(EXP_DIR / "cross_site_held_out_validation_results.json", "w") as f:
        json.dump({"n_communities": len(communities), "n_testable": n_testable,
                    "n_corroborated": n_corroborated, "results": results, "verdict": verdict},
                   f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'cross_site_held_out_validation_results.json'}")


if __name__ == "__main__":
    main()
