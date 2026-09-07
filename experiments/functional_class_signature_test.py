"""
experiments/functional_class_signature_test.py
==================================================
Stage 3, item 9: now that the substitution-graph communities are
validated (embedding-corroborated, both within-corpus and cross-site
held-out; matched-size-tested against the Harappa sample-size confound),
the next question is not "are these real" but "what kind of real are
they." A morphological paradigm (e.g. a family of suffix variants) would
be expected to occupy a consistent SLOT -- the same relative position in
an inscription -- across its members, the way a suffix always comes at
the end regardless of which specific allomorph is used. A semantic or
lexical class (e.g. a family of related commodity signs) would not be
expected to share a slot at all; its members could appear anywhere a
noun-like sign can appear.

This tests two related signals for each of the top motif-corroborated
substitution communities:

  1. SUBSTITUTION POSITION CONCENTRATION: every minimal pair connecting
     two members of a community happened at a specific position index in
     some inscription (MinimalPair.position, already recorded by
     analysis/minimal_pairs.py). If a community's substitutions cluster
     at one or two specific position indices rather than scattering
     across many, that's consistent with a fixed grammatical slot.
  2. INDIVIDUAL POSITIONAL PROFILE SIMILARITY: for each community member,
     independently compute its own overall initial/medial/final usage
     tendency across the whole corpus (reusing analysis/positional.py).
     If members of a community share similar profiles (e.g. all tend to
     be final-heavy), that is consistent with a shared functional role;
     if profiles are scattered, that argues against it.

Neither signal alone proves morphology (a semantic class could
coincidentally share a slot; a grammatical class could show up in varied
surface positions depending on inscription length), but a community
showing BOTH strong slot concentration AND strong profile similarity is
a stronger morphology candidate than one showing neither.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_corpus_csv
from analysis.minimal_pairs import find_minimal_pairs
from analysis.substitution_graph import build_substitution_graph_nx, annotate_distinct_contexts, modularity_communities
from analysis.positional import positional_distribution

EXP_DIR = Path(__file__).parent


def position_concentration(pairs, community: set[str], by_id: dict, min_length: int = 3) -> dict:
    """For minimal pairs connecting two members of this community, how
    concentrated are the position indices they occurred at? Reports the
    share of pairs at the single most common position (higher = more
    slot-like), and the position histogram itself.

    IMPORTANT: restricted to pairs from inscriptions at least min_length
    signs long. Minimal pairs only ever compare same-length inscriptions,
    and length-2 inscriptions have only 2 possible differing positions
    (0 or 1) to begin with -- "concentration" there is close to
    mechanical, not evidence of anything. This was checked directly
    before trusting an earlier version of this function that didn't
    filter by length: two of the largest communities' apparent 92-97%
    position concentration turned out to be 93-96% driven by length-2
    pairs alone. Restricting to min_length=3+ removes that confound,
    at the cost of a much smaller n_pairs for communities whose evidence
    is mostly short inscriptions.
    """
    positions = []
    for p in pairs:
        if p.sign_a not in community or p.sign_b not in community:
            continue
        ins = by_id.get(p.inscription_a)
        if ins is None or len(ins.normalized_signs()) < min_length:
            continue
        positions.append(p.position)

    if not positions:
        return {"n_pairs": 0, "top_position_share": float("nan"), "histogram": {}}
    counts = Counter(positions)
    top_position, top_count = counts.most_common(1)[0]
    return {
        "n_pairs": len(positions),
        "top_position": top_position,
        "top_position_share": top_count / len(positions),
        "histogram": dict(counts),
    }


def individual_positional_profiles(sequences: list[list[str]], community: set[str]) -> dict:
    """For each community member, its own final-position share across the
    WHOLE corpus (not just within this community's substitution pairs).
    Reports the spread (std deviation) across members: low spread means
    members share a similar positional tendency."""
    final_counts = positional_distribution(sequences, "final")
    initial_counts = positional_distribution(sequences, "initial")
    overall_counts = Counter(s for seq in sequences for s in seq)

    profiles = {}
    for sign in community:
        overall = overall_counts.get(sign, 0)
        if overall == 0:
            continue
        final_share = final_counts.get(sign, 0) / overall if overall else 0.0
        initial_share = initial_counts.get(sign, 0) / overall if overall else 0.0
        profiles[sign] = {"final_share": final_share, "initial_share": initial_share}

    if len(profiles) < 2:
        return {"n_members_profiled": len(profiles), "final_share_std": float("nan"),
                "initial_share_std": float("nan"), "profiles": profiles}

    final_shares = [p["final_share"] for p in profiles.values()]
    initial_shares = [p["initial_share"] for p in profiles.values()]

    def std(xs):
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    return {
        "n_members_profiled": len(profiles),
        "final_share_mean": sum(final_shares) / len(final_shares),
        "final_share_std": std(final_shares),
        "initial_share_mean": sum(initial_shares) / len(initial_shares),
        "initial_share_std": std(initial_shares),
        "profiles": profiles,
    }


def main():
    print("Loading corpus and rebuilding motif-corroborated communities...")
    corpus = load_corpus_csv("data/indus_website_real_corpus.csv")
    filtered = corpus.filter(exclude_damaged=True)
    sequences = filtered.sequences(normalized=True)

    pairs = find_minimal_pairs(filtered)
    G = build_substitution_graph_nx(pairs, tier="motif")
    annotate_distinct_contexts(G, filtered, pairs)
    communities = modularity_communities(G)
    print(f"  {len(communities)} communities found\n")

    results = []
    print(f"{'community':>10s} {'size':>5s} {'top pos. share (all)':>21s} {'top pos. share (len>=3)':>24s} "
          f"{'n_pairs len>=3':>15s} {'final-share std':>17s} {'slot-like?':>11s}")
    print("-" * 105)
    by_id = {ins.inscription_id: ins for ins in filtered.inscriptions}
    for i, community in enumerate(communities[:10]):
        pos_result_all = position_concentration(pairs, community, by_id, min_length=1)
        pos_result_long = position_concentration(pairs, community, by_id, min_length=3)
        profile_result = individual_positional_profiles(sequences, community)

        # the length-controlled version is the one that actually means
        # something; the "all lengths" version is reported alongside for
        # transparency about how much the naive number was inflated
        share_long = pos_result_long["top_position_share"]
        slot_like = (
            share_long == share_long and share_long > 0.6 and pos_result_long["n_pairs"] >= 5 and
            profile_result["final_share_std"] == profile_result["final_share_std"] and
            profile_result["final_share_std"] < 0.15
        )

        share_all_str = f"{pos_result_all['top_position_share']:.3f}" if pos_result_all["top_position_share"] == pos_result_all["top_position_share"] else "n/a"
        share_long_str = f"{share_long:.3f}" if share_long == share_long else "n/a"
        print(f"{i:>10d} {len(community):>5d} {share_all_str:>21s} {share_long_str:>24s} "
              f"{pos_result_long['n_pairs']:>15d} "
              f"{profile_result.get('final_share_std', float('nan')):>17.3f} "
              f"{'YES' if slot_like else 'no':>11s}")

        results.append({
            "community_index": i, "size": len(community),
            "position_concentration_all_lengths": pos_result_all,
            "position_concentration_length_controlled": pos_result_long,
            "individual_profiles": profile_result,
            "slot_like_call": slot_like,
        })

    n_slot_like = sum(r["slot_like_call"] for r in results)
    print(f"\n{'='*70}")
    print(f"{n_slot_like} of {len(results)} tested communities show BOTH concentrated "
          f"substitution position (length-controlled, n>=5 pairs) AND low individual "
          f"profile spread.")
    print("""
Read this cautiously, and note the length-controlled column specifically:
an earlier version of this test did not control for inscription length,
and found 92-97% position concentration for the two largest, most
pair-rich communities. Checking why revealed that 93-96% of those
communities' pairs came from length-2 inscriptions alone, which have
only 2 possible differing positions to begin with -- concentration
there is close to mechanical, not evidence of grammar. The
length-controlled (>=3 signs) column removes that confound, at the cost
of far fewer usable pairs for communities whose evidence is mostly short
inscriptions. This is a coarse, threshold-based heuristic either way,
not a statistical test with a null distribution, and it does not
distinguish a grammatical slot from a coincidental positional habit
shared for other reasons. A "YES" here is a candidate worth a closer
look, not a demonstrated morphological paradigm.""")

    with open(EXP_DIR / "functional_class_signature_results.json", "w") as f:
        json.dump({"n_communities_tested": len(results), "n_slot_like": n_slot_like,
                    "results": results}, f, indent=2, default=str)
    print(f"\nResults written to {EXP_DIR / 'functional_class_signature_results.json'}")


if __name__ == "__main__":
    main()
