"""
analysis/substitution_graph.py
=================================
Upgrades the plain connected-components view in analysis/minimal_pairs.py
into a proper weighted, attributed graph (via networkx), and adds the
test that actually matters for trusting a candidate paradigm class:
does it survive being recomputed on an independent slice of the data?

Edge attributes recorded per substitution (sign_i, sign_j):
  weight            total number of raw pair observations
  distinct_contexts number of DISTINCT (site, motif) combinations
                    supporting this edge -- five identical pairs from the
                    same single site+motif combination are one piece of
                    corroborating context, not five; this is a simple,
                    conservative proxy for "independent evidence" per the
                    review that prompted this module
  positions         list of the differing position index per observation
  lengths           list of the inscription length per observation

Community detection offers two views on the same graph: plain connected
components (as before, everything reachable via ANY edge at a weight
threshold) and networkx's greedy modularity communities (which can split
a single connected component into denser sub-groups, useful when the
threshold-based view merges too aggressively, the exact failure mode
documented for the min_edge_weight=1 case in minimal_pairs.py).

Stability testing: for a class found on the full corpus, recompute the
same substitution-mining procedure restricted to inscriptions from ONE
site only, and report the Jaccard overlap between the full-corpus class
(restricted to signs that even appear in that site's own vocabulary,
since a class member absent from a site's vocabulary can't possibly be
tested there) and whichever site-restricted class overlaps it most. A
class with high overlap across multiple sites is a much stronger
candidate than one that only appears in the pooled, full-corpus view.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import networkx as nx

from data.loader import Corpus
from analysis.minimal_pairs import find_minimal_pairs, MinimalPair


def build_substitution_graph_nx(pairs: list[MinimalPair], tier: str = "motif") -> nx.Graph:
    G = nx.Graph()
    for p in pairs:
        if tier == "context" and not p.corroborated_context:
            continue
        if tier == "motif" and not p.corroborated_motif:
            continue
        if G.has_edge(p.sign_a, p.sign_b):
            data = G[p.sign_a][p.sign_b]
            data["weight"] += 1
            data["positions"].append(p.position)
        else:
            G.add_edge(p.sign_a, p.sign_b, weight=1, positions=[p.position])
    return G


def annotate_distinct_contexts(G: nx.Graph, corpus: Corpus, pairs: list[MinimalPair]) -> None:
    """Adds a `distinct_contexts` count per edge: the number of unique
    (site, motif) combinations among the inscriptions that produced each
    substitution edge. Mutates G in place."""
    by_id = {ins.inscription_id: ins for ins in corpus.inscriptions}
    contexts: dict[tuple[str, str], set] = {}
    for p in pairs:
        key = tuple(sorted((p.sign_a, p.sign_b)))
        a_ins = by_id.get(p.inscription_a)
        if a_ins is None:
            continue
        contexts.setdefault(key, set()).add((a_ins.site, a_ins.motif))
    for (a, b), ctxs in contexts.items():
        if G.has_edge(a, b):
            G[a][b]["distinct_contexts"] = len(ctxs)


def connected_component_classes(G: nx.Graph, min_edge_weight: int = 1) -> list[set[str]]:
    H = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= min_edge_weight)
    H.add_nodes_from(G.nodes())
    return sorted((c for c in nx.connected_components(H) if len(c) > 1), key=len, reverse=True)


def modularity_communities(G: nx.Graph) -> list[set[str]]:
    """Greedy modularity communities -- can split one loosely-connected
    component into denser sub-groups that plain connectivity can't see."""
    if G.number_of_edges() == 0:
        return []
    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    return sorted((set(c) for c in communities if len(c) > 1), key=len, reverse=True)


def degree_centrality_top(G: nx.Graph, top_n: int = 10) -> list[tuple[str, float]]:
    centrality = nx.degree_centrality(G)
    return sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


@dataclass
class StabilityResult:
    class_signs: set
    site: str
    site_vocabulary_overlap: set        # class signs that even appear at this site
    best_matching_site_class: set
    jaccard: float
    testable: bool


def test_class_stability_by_site(full_corpus: Corpus, full_class: set[str],
                                   site: str, tier: str = "motif",
                                   min_edge_weight: int = 1) -> StabilityResult:
    site_corpus = Corpus([ins for ins in full_corpus.filter(exclude_damaged=True).inscriptions
                           if ins.site == site])
    site_vocab = set(sign for ins in site_corpus.inscriptions for sign in ins.normalized_signs())
    class_at_site = full_class & site_vocab

    if len(class_at_site) < 2:
        return StabilityResult(class_signs=full_class, site=site,
                                site_vocabulary_overlap=class_at_site,
                                best_matching_site_class=set(), jaccard=float("nan"),
                                testable=False)

    site_pairs = find_minimal_pairs(site_corpus)
    site_graph = build_substitution_graph_nx(site_pairs, tier=tier)
    site_classes = connected_component_classes(site_graph, min_edge_weight=min_edge_weight)

    best_overlap, best_class = 0.0, set()
    for c in site_classes:
        inter = len(class_at_site & c)
        union = len(class_at_site | c) or 1
        jac = inter / union
        if jac > best_overlap:
            best_overlap, best_class = jac, c

    return StabilityResult(class_signs=full_class, site=site,
                            site_vocabulary_overlap=class_at_site,
                            best_matching_site_class=best_class, jaccard=best_overlap,
                            testable=True)
