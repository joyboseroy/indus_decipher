"""
data/stratified_null_model.py
================================
Stage 2's first real question, per the design proposed in external
review: the validated order-3 finding (see README's "Order 3 is
validated..." section) could in principle be explained three ways.

  H1 (linguistic/sequential): the trigram signal exists within sign
     sequencing itself, independent of which motif or site an inscription
     belongs to.
  H2 (archaeological/compositional): different motifs or sites simply
     favor different signs, and POOLING these different sign-frequency
     populations together is itself enough to manufacture the appearance
     of trigram-level structure, with no real within-group sequential
     dependency at all.
  H3 (mixed): some of both.

`generate_stratified_null_corpus()` tests this directly. It is the same
no-dependency generation logic as data/adversarial_null_model.py --
every sign in a generated inscription is still drawn INDEPENDENTLY, so
there is still no real sequential dependency of any kind -- but the
marginal distributions used to draw from are now computed SEPARATELY
PER STRATUM (e.g. per motif, per site, or per site+motif combination)
rather than pooled across the whole corpus. If H2 is doing real work,
stratifying this way should let a still-zero-dependency null reproduce
more of the real corpus's apparent structure than the single pooled
adversarial null could, since it now captures "different groups prefer
different signs" even though it still captures no real sequencing.

If real data still shows a positive order-3 information gain relative to
ALL THREE stratified nulls (motif-stratified, site-stratified,
site+motif-stratified), that is stronger evidence for H1 than the
original single pooled-null comparison could provide, precisely because
each stratified null has been given every opportunity H2 would need to
succeed and still has no real sequential dependency to fall back on.

Strata with too few real inscriptions to estimate their own marginals
reliably (below `min_stratum_size`) fall back to the corpus-wide pooled
marginals for generation, the same kind of backoff used elsewhere in
this project (trigram_markov_null's backoff to bigram/unigram statistics
for sparse contexts) for the identical reason: a marginal estimated from
a handful of inscriptions is not trustworthy enough to generate from
directly.
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict
from typing import Callable

from data.loader import Corpus, Inscription


def broad_motif(motif: str) -> str:
    """Collapses fine-grained motif variants (Bull1:W, Bull1:J, ...) into
    their shared broad category (Bull1), matching the grouping already
    used elsewhere in this project (e.g. the Mohenjo-daro+unicorn matched
    subset in experiments/corpus_divergence.py) for a stratum large
    enough to estimate marginals from."""
    if motif == "unknown":
        return "unknown"
    return motif.split(":")[0]


def _stratum_marginals(sequences: list[list[str]]) -> dict:
    lengths = [len(s) for s in sequences if len(s) >= 1]
    overall_counts = Counter(sign for seq in sequences for sign in seq)
    initial_counts = Counter(seq[0] for seq in sequences if seq)
    multi = [s for s in sequences if len(s) >= 2]
    final_counts = Counter(seq[-1] for seq in multi) if multi else overall_counts
    return {"lengths": lengths, "overall": overall_counts,
            "initial": initial_counts, "final": final_counts}


def generate_stratified_null_corpus(real_corpus: Corpus, stratify_by: Callable[[Inscription], str],
                                      n_inscriptions: int, seed: int = 0,
                                      min_stratum_size: int = 20, id_prefix: str = "STRATNULL") -> Corpus:
    rng = random.Random(seed)
    filtered = real_corpus.filter(exclude_damaged=True)
    all_ins = [ins for ins in filtered.inscriptions if ins.normalized_signs()]

    by_stratum: dict[str, list[Inscription]] = defaultdict(list)
    for ins in all_ins:
        by_stratum[stratify_by(ins)].append(ins)

    pooled_sequences = [ins.normalized_signs() for ins in all_ins]
    pooled_marginals = _stratum_marginals(pooled_sequences)

    stratum_marginals: dict[str, dict] = {}
    for key, members in by_stratum.items():
        if len(members) >= min_stratum_size:
            stratum_marginals[key] = _stratum_marginals([m.normalized_signs() for m in members])
        else:
            stratum_marginals[key] = pooled_marginals  # backoff: too few to trust its own marginals

    strata_keys = list(by_stratum.keys())
    strata_weights = [len(by_stratum[k]) for k in strata_keys]

    def draw(signs_counter: Counter):
        if not signs_counter:
            signs_counter = pooled_marginals["overall"]
        signs, weights = zip(*signs_counter.items())
        return rng.choices(signs, weights=weights, k=1)[0]

    inscriptions = []
    for i in range(n_inscriptions):
        stratum_key = rng.choices(strata_keys, weights=strata_weights, k=1)[0]
        m = stratum_marginals[stratum_key]
        length = rng.choice(m["lengths"]) if m["lengths"] else 1

        if length <= 1:
            seq = [draw(m["initial"])]
        else:
            seq = [draw(m["initial"])]
            seq += [draw(m["overall"]) for _ in range(length - 2)]
            seq.append(draw(m["final"]))

        inscriptions.append(Inscription(
            inscription_id=f"{id_prefix}-{i:05d}", signs=seq,
            site="null_model", object_type="unknown",
            reading_direction="R-L", motif="unknown",
        ))

    return Corpus(inscriptions)


def motif_stratified_null(real_corpus: Corpus, n_inscriptions: int, seed: int = 0,
                            min_stratum_size: int = 20) -> Corpus:
    return generate_stratified_null_corpus(
        real_corpus, stratify_by=lambda ins: broad_motif(ins.motif),
        n_inscriptions=n_inscriptions, seed=seed, min_stratum_size=min_stratum_size,
        id_prefix="MOTIFNULL")


def site_stratified_null(real_corpus: Corpus, n_inscriptions: int, seed: int = 0,
                           min_stratum_size: int = 20) -> Corpus:
    return generate_stratified_null_corpus(
        real_corpus, stratify_by=lambda ins: ins.site,
        n_inscriptions=n_inscriptions, seed=seed, min_stratum_size=min_stratum_size,
        id_prefix="SITENULL")


def site_motif_stratified_null(real_corpus: Corpus, n_inscriptions: int, seed: int = 0,
                                 min_stratum_size: int = 20) -> Corpus:
    return generate_stratified_null_corpus(
        real_corpus, stratify_by=lambda ins: f"{ins.site}||{broad_motif(ins.motif)}",
        n_inscriptions=n_inscriptions, seed=seed, min_stratum_size=min_stratum_size,
        id_prefix="SITEMOTIFNULL")
