"""
experiments/substitution_graph_analysis.py
=============================================
Runs the weighted substitution graph (analysis/substitution_graph.py) on
the large real corpus: builds the motif-corroborated graph, compares plain
connected components against greedy modularity communities, reports
degree centrality (which signs sit at the center of the most substitution
activity), and tests whether the largest classes survive being
recomputed independently at Mohenjo-daro and at Harappa, the corpus's two
largest sites.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from analysis.minimal_pairs import find_minimal_pairs
from analysis.substitution_graph import (build_substitution_graph_nx, annotate_distinct_contexts,
                                          connected_component_classes, modularity_communities,
                                          degree_centrality_top, test_class_stability_by_site)

EXP_DIR = Path(__file__).parent


def main():
    print("Loading large real corpus...")
    corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    filtered = corpus.filter(exclude_damaged=True)

    print("Mining minimal pairs and building the motif-corroborated substitution graph...")
    pairs = find_minimal_pairs(filtered)
    G = build_substitution_graph_nx(pairs, tier="motif")
    annotate_distinct_contexts(G, filtered, pairs)
    print(f"  Graph: {G.number_of_nodes()} signs, {G.number_of_edges()} substitution edges")

    print("\n=== Edges with the most independent corroborating contexts ===")
    edges_by_context = sorted(G.edges(data=True), key=lambda e: e[2].get("distinct_contexts", 0), reverse=True)
    for a, b, d in edges_by_context[:10]:
        print(f"  {a} <-> {b}: weight={d['weight']}, distinct_contexts={d.get('distinct_contexts', 0)}")

    print("\n=== Connected components (plain connectivity, min_edge_weight=1) ===")
    cc_classes = connected_component_classes(G, min_edge_weight=1)
    for c in cc_classes[:5]:
        print(f"  size={len(c)}: {sorted(c)[:10]}{'...' if len(c) > 10 else ''}")

    print("\n=== Greedy modularity communities (same graph, different clustering method) ===")
    mod_classes = modularity_communities(G)
    for c in mod_classes[:8]:
        print(f"  size={len(c)}: {sorted(c)[:10]}{'...' if len(c) > 10 else ''}")
    print(f"  ({len(cc_classes)} connected components vs. {len(mod_classes)} modularity "
          f"communities -- modularity splitting a component into more, smaller communities "
          f"means the naive connectivity view was merging genuinely distinct sub-structure)")

    print("\n=== Degree centrality: which signs sit at the center of substitution activity ===")
    for sign, cent in degree_centrality_top(G, top_n=10):
        print(f"  {sign}: {cent:.3f}")

    print("\n=== Cross-site stability: do the largest modularity communities survive "
          "being recomputed at individual sites? ===")
    stability_results = []
    for i, cls in enumerate(mod_classes[:5]):
        print(f"\n  Community {i} (size={len(cls)}): {sorted(cls)[:8]}"
              f"{'...' if len(cls) > 8 else ''}")
        for site in ["Mohenjo-daro", "Harappa"]:
            result = test_class_stability_by_site(filtered, cls, site, tier="motif")
            stability_results.append({
                "community_index": i, "community_size": len(cls), "site": site,
                "testable": result.testable, "jaccard": result.jaccard,
                "site_vocabulary_overlap_size": len(result.site_vocabulary_overlap),
                "best_matching_site_class_size": len(result.best_matching_site_class),
            })
            if result.testable:
                print(f"    at {site}: {len(result.site_vocabulary_overlap)} of this class's "
                      f"signs appear in {site}'s own vocabulary; best-matching {site}-only "
                      f"class has Jaccard overlap = {result.jaccard:.2f}")
            else:
                print(f"    at {site}: not testable (fewer than 2 of this class's signs "
                      f"appear in {site}'s own vocabulary, or no minimal pairs found there)")

    with open(EXP_DIR / "substitution_graph_results.json", "w") as f:
        json.dump({
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "n_connected_components": len(cc_classes),
            "n_modularity_communities": len(mod_classes),
            "stability_results": stability_results,
        }, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'substitution_graph_results.json'}")


if __name__ == "__main__":
    main()
