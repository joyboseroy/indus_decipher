"""
experiments/sign_embeddings_analysis.py
===========================================
Stage 2's last item: contextual sign embeddings, and the reason to build
them at all -- cross-validating against analysis/substitution_graph.py's
minimal-pair-derived communities (already found: 18 connected components,
19 modularity communities from motif-corroborated minimal pairs).

Minimal pairs are a strict, local test: do two signs substitute in an
otherwise-identical context? Embeddings are a loose, global test: do two
signs tend to co-occur with similar OTHER signs across the whole corpus,
regardless of exact context matches? If a substitution-graph community's
members ALSO sit close together in embedding space, that is convergent
evidence for a real functional class, found two structurally different
ways. If they don't, that tells us the two methods are picking up on
different kinds of structure (or one of them is picking up noise),
which is also worth knowing plainly rather than only reporting whichever
method gave the more exciting-looking result.

Metric: for each substitution-graph community, the mean pairwise cosine
similarity among its own members, compared against the mean pairwise
cosine similarity of the same number of RANDOMLY chosen sign pairs (as a
baseline for "how similar are two arbitrary signs, typically"). A
community whose internal similarity clearly exceeds the random baseline
is corroborated by the embeddings; one that doesn't is not.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from analysis.sign_embeddings import train_sign_embeddings
from analysis.minimal_pairs import find_minimal_pairs
from analysis.substitution_graph import (build_substitution_graph_nx, annotate_distinct_contexts,
                                          modularity_communities)

EXP_DIR = Path(__file__).parent


def mean_pairwise_similarity(embeddings, signs: list[str]) -> float:
    if len(signs) < 2:
        return float("nan")
    sims = []
    for i in range(len(signs)):
        for j in range(i + 1, len(signs)):
            s = embeddings.similarity(signs[i], signs[j])
            if not (s != s):  # skip NaN
                sims.append(s)
    return sum(sims) / len(sims) if sims else float("nan")


def random_baseline_similarity(embeddings, all_signs: list[str], community_size: int,
                                 n_trials: int = 200, seed: int = 0) -> float:
    rng = random.Random(seed)
    means = []
    for _ in range(n_trials):
        sample = rng.sample(all_signs, min(community_size, len(all_signs)))
        m = mean_pairwise_similarity(embeddings, sample)
        if m == m:  # not NaN
            means.append(m)
    return sum(means) / len(means) if means else float("nan")


def main():
    print("Loading corpus and training PPMI+SVD sign embeddings...")
    corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    filtered = corpus.filter(exclude_damaged=True)
    sequences = filtered.sequences(normalized=True)

    embeddings = train_sign_embeddings(sequences, window=2, n_dims=20)
    print(f"  Trained embeddings for {len(embeddings.signs)} signs, {embeddings.vectors.shape[1]} dimensions")

    print("\nExample nearest neighbors (sanity check the embeddings are non-degenerate):")
    for sign in embeddings.signs[:3]:
        neighbors = embeddings.nearest_neighbors(sign, top_n=3)
        print(f"  {sign} -> {[(n, round(s, 3)) for n, s in neighbors]}")

    print("\nBuilding motif-corroborated substitution graph (same as substitution_graph_analysis.py)...")
    pairs = find_minimal_pairs(filtered)
    G = build_substitution_graph_nx(pairs, tier="motif")
    annotate_distinct_contexts(G, filtered, pairs)
    communities = modularity_communities(G)
    print(f"  {len(communities)} modularity communities found")

    print(f"\n{'community':>10s} {'size':>5s} {'internal cos-sim':>18s} {'random baseline':>16s} {'corroborated?':>14s}")
    print("-" * 70)
    results = []
    for i, community in enumerate(communities[:10]):
        members = sorted(s for s in community if s in embeddings.sign_to_index)
        if len(members) < 2:
            continue
        internal_sim = mean_pairwise_similarity(embeddings, members)
        baseline = random_baseline_similarity(embeddings, embeddings.signs, len(members))
        corroborated = internal_sim > baseline
        results.append({
            "community_index": i, "size": len(members), "members": members,
            "internal_similarity": internal_sim, "random_baseline": baseline,
            "corroborated": corroborated,
        })
        mark = "YES" if corroborated else "no"
        print(f"{i:>10d} {len(members):>5d} {internal_sim:>18.4f} {baseline:>16.4f} {mark:>14s}")

    n_corroborated = sum(r["corroborated"] for r in results)
    print(f"\n{n_corroborated} of {len(results)} tested communities show higher internal "
          f"similarity than the random baseline.")
    if n_corroborated >= len(results) * 0.7:
        verdict = ("Most substitution-graph communities ARE corroborated by an independent, "
                   "structurally different method (distributional context similarity). This "
                   "is convergent evidence these are real functional classes, not artifacts "
                   "of the minimal-pair mining procedure alone.")
    elif n_corroborated <= len(results) * 0.3:
        verdict = ("Most substitution-graph communities are NOT corroborated by embedding "
                   "similarity. This does not necessarily mean the substitution-graph "
                   "communities are wrong -- minimal pairs and distributional similarity are "
                   "different questions -- but it means they should not be treated as two "
                   "confirmations of the same finding; report them as separate, only "
                   "partially overlapping lines of evidence.")
    else:
        verdict = ("A mixed result: some communities are corroborated, some are not. Worth "
                   "reporting per-community rather than as a single aggregate finding.")
    print(f"\n{verdict}")

    with open(EXP_DIR / "sign_embeddings_results.json", "w") as f:
        json.dump({"n_communities_tested": len(results), "n_corroborated": n_corroborated,
                    "results": results, "verdict": verdict}, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'sign_embeddings_results.json'}")


if __name__ == "__main__":
    main()
