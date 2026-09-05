"""
analysis/minimal_pairs.py
===========================
"Seal-twin" / minimal-pair mining.

Idea (linguistic minimal pairs, applied to an unknown script): if two
inscriptions are identical except at one position, whatever occupies that
position in each is, at minimum, *substitutable* in that syntactic slot --
without knowing what either sign means. Collecting many such substitutions
lets us build a graph of which signs interchange with which, and the
connected components of that graph are candidate paradigmatic classes
(morphological alternants, semantic categories, or grammatical markers --
we can't yet say which, but the clustering itself is evidence-neutral and
requires no assumption about the underlying language).

This is cheap to compute (needs only the sign-sequence corpus, no images,
no external linguistic data) and, as far as the literature reviewed for
this project shows, is under-exploited relative to raw n-gram/entropy work.

THREE strengths of a candidate pair are tracked, in increasing order of
confidence:
  - "sequence-only": same length, differ at exactly one position. Cheapest,
    weakest -- two unrelated inscriptions can coincidentally have the same
    length and differ at one slot.
  - "corroborated_context": sequence-only AND same site AND same
    object_type. Rules out "these are just different inscriptions that
    happen to share a length."
  - "corroborated_motif": corroborated_context AND same iconographic motif
    (Inscription.motif -- e.g. "Bull1:W", "Gaur", or a CISI-style
    "unicorn_IV"). This is the strongest tier: two seals sharing site,
    object type, AND field-symbol iconography, differing in only one sign,
    are a much better analogy to a true linguistic minimal pair than a
    same-length coincidence -- the iconographic match makes it plausible
    these are the same "kind" of seal (e.g. same workshop, same
    administrative category, same owner's guild), so a one-sign difference
    is more likely a genuine substitutable slot than noise. Only computed
    for inscriptions where BOTH have a known (non-"unknown") motif.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from data.loader import Corpus, Inscription


@dataclass
class MinimalPair:
    inscription_a: str
    inscription_b: str
    position: int
    sign_a: str
    sign_b: str
    corroborated_context: bool  # same site + object_type
    corroborated_motif: bool    # corroborated_context AND same known motif


def find_minimal_pairs(corpus: Corpus, max_group_size: int = 400) -> list[MinimalPair]:
    """O(n^2) within each length bucket -- fine for corpora up to a few
    thousand inscriptions. max_group_size guards against pathological
    buckets (e.g. thousands of length-1 inscriptions) blowing up runtime;
    such buckets are split into fixed-size chunks and compared within
    each chunk only, which undercounts rare cross-chunk pairs but keeps
    the miner tractable."""
    by_length: dict[int, list[Inscription]] = defaultdict(list)
    for ins in corpus.inscriptions:
        seq = ins.normalized_signs()
        if len(seq) < 2:
            continue  # a pair needs at least one position to hold constant
        by_length[len(seq)].append(ins)

    pairs: list[MinimalPair] = []
    for length, group in by_length.items():
        for start in range(0, len(group), max_group_size):
            chunk = group[start:start + max_group_size]
            for a, b in combinations(chunk, 2):
                sa, sb = a.normalized_signs(), b.normalized_signs()
                diffs = [i for i in range(length) if sa[i] != sb[i]]
                if len(diffs) != 1:
                    continue
                i = diffs[0]
                corroborated_context = (a.site == b.site and a.object_type == b.object_type
                                          and a.site != "unknown")
                corroborated_motif = (corroborated_context
                                       and a.motif == b.motif
                                       and a.motif != "unknown")
                pairs.append(MinimalPair(
                    inscription_a=a.inscription_id, inscription_b=b.inscription_id,
                    position=i, sign_a=sa[i], sign_b=sb[i],
                    corroborated_context=corroborated_context,
                    corroborated_motif=corroborated_motif,
                ))
    return pairs


def build_substitution_graph(pairs: list[MinimalPair], tier: str = "all") -> dict[str, Counter]:
    """Undirected graph: graph[sign] = Counter of {other_sign: count} for
    every observed substitution.
    tier: "all" (every sequence-only pair), "context" (corroborated_context
    only), or "motif" (corroborated_motif only -- strongest, sparsest)."""
    graph: dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        if tier == "context" and not p.corroborated_context:
            continue
        if tier == "motif" and not p.corroborated_motif:
            continue
        graph[p.sign_a][p.sign_b] += 1
        graph[p.sign_b][p.sign_a] += 1
    return dict(graph)


def connected_components(graph: dict[str, Counter], min_edge_weight: int = 1) -> list[set[str]]:
    """Candidate paradigm classes: signs reachable from each other via
    substitution edges of at least min_edge_weight. Raise min_edge_weight
    to keep only well-attested substitutions (fewer, more confident classes)."""
    visited: set[str] = set()
    components = []
    for start in graph:
        if start in visited:
            continue
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            for neighbor, weight in graph.get(node, {}).items():
                if weight >= min_edge_weight and neighbor not in component:
                    stack.append(neighbor)
        visited |= component
        if len(component) > 1:
            components.append(component)
    components.sort(key=len, reverse=True)
    return components


@dataclass
class MinimalPairReport:
    n_total_pairs: int
    n_corroborated_context_pairs: int
    n_corroborated_motif_pairs: int
    n_inscriptions_with_known_motif: int
    n_paradigm_classes: int
    n_paradigm_classes_motif_tier: int
    largest_classes: list[dict] = field(default_factory=list)
    largest_classes_motif_tier: list[dict] = field(default_factory=list)
    position_distribution: dict[int, int] = field(default_factory=dict)


def _summarize_components(components: list[set[str]], pairs: list[MinimalPair],
                            top_n: int) -> list[dict]:
    summaries = []
    for comp in components[:top_n]:
        examples = []
        for p in pairs:
            if p.sign_a in comp and p.sign_b in comp:
                examples.append((p.sign_a, p.sign_b))
            if len(examples) >= 5:
                break
        summaries.append({
            "size": len(comp),
            "signs": sorted(comp),
            "example_substitutions": examples,
        })
    return summaries


def analyze_minimal_pairs(corpus: Corpus, min_edge_weight: int = 1, top_n_classes: int = 8) -> MinimalPairReport:
    pairs = find_minimal_pairs(corpus)

    graph_all = build_substitution_graph(pairs, tier="all")
    components = connected_components(graph_all, min_edge_weight=min_edge_weight)

    graph_motif = build_substitution_graph(pairs, tier="motif")
    components_motif = connected_components(graph_motif, min_edge_weight=1)

    pos_dist = Counter(p.position for p in pairs)
    n_known_motif = sum(1 for ins in corpus.inscriptions if ins.motif != "unknown")

    return MinimalPairReport(
        n_total_pairs=len(pairs),
        n_corroborated_context_pairs=sum(p.corroborated_context for p in pairs),
        n_corroborated_motif_pairs=sum(p.corroborated_motif for p in pairs),
        n_inscriptions_with_known_motif=n_known_motif,
        n_paradigm_classes=len(components),
        n_paradigm_classes_motif_tier=len(components_motif),
        largest_classes=_summarize_components(components, pairs, top_n_classes),
        largest_classes_motif_tier=_summarize_components(components_motif, pairs, top_n_classes),
        position_distribution=dict(pos_dist),
    )
