"""
data/synthetic_continuum.py
==============================
Extends data/synthetic_civilizations.py's three generators with two more,
chosen specifically to attack the biggest remaining weakness in this
project's falsification harness: the original three civilizations are
deliberately quite distinct, so a classifier separating them is a weak
test of whether the FEATURE SPACE can distinguish plausible alternative
mechanisms, as opposed to obviously different toy worlds.

These two generators are designed to be the sharpest available test of
one specific claim: does the real corpus's validated order-3 information
gain (README's "Order 3 is validated..." section) require something
morphology-like to produce, or can ANY sufficiently structured mechanism,
including one with zero linguistic motivation, produce it too?

  Civilization D -- "pure Markov, no morphology":
      a genuine order-2 (trigram) Markov chain with a HAND-DESIGNED
      transition structure (not fit from real data, unlike
      data/permutation_nulls.py's trigram_markov_null, which exists to
      test the real corpus specifically). Real sequential dependency
      exists by construction -- unlike civ_b's independent slots -- but
      there is no root+suffix compositional process, no agreement, no
      morphology of any kind. Just raw sign-to-sign order-2 transition
      probabilities. If this civilization ALSO shows a positive order-3
      information gain the way real data does, that is evidence order-3
      structure alone does not distinguish "morphological" from "merely
      Markov-structured," which would meaningfully qualify how strong a
      claim the order-3 finding actually supports.

  Civilization E -- "hierarchical administrative":
      a purely bureaucratic three-level nested structure: draw a
      CATEGORY, then a SUBCATEGORY conditioned on that category, then an
      ITEM sign conditioned on that subcategory. This creates genuine
      multi-level statistical dependency (an item's distribution depends
      on category two steps back) through a mechanism with no linguistic
      motivation at all -- the administrative equivalent of a filing
      system with nested folders. Tests whether multi-level dependency
      specifically requires anything language-like, or whether ordinary
      bureaucratic nesting produces the same statistical signature.

Neither of these is a claim about what the Indus script actually is;
like data/synthetic_civilizations.py's three, they exist so the analysis
battery's discriminating power can be measured against ground truth,
this time against mechanisms deliberately chosen to be harder to tell
apart from "linguistic" than the original three.
"""
from __future__ import annotations
import random
from data.loader import Corpus, Inscription


def _mk_signs(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(n)]


def generate_civilization_d_markov_no_morphology(n_inscriptions: int = 1500, seed: int = 4) -> Corpus:
    """Pure order-2 Markov chain, hand-designed transition structure, no
    morphological process of any kind. Vocabulary and transition
    structure sized comparably to civ_a/civ_b for a fair comparison."""
    rng = random.Random(seed)
    vocab = _mk_signs("MKD", 40)
    end_token = "<END>"

    # hand-design a genuine order-2 transition structure: for each
    # (prev2, prev1) pair we care about, a skewed distribution over next
    # sign, so real trigram-level dependency exists by construction. We
    # don't enumerate all 40x40 pairs (most would be arbitrary); instead
    # a smaller number of "special" bigram contexts get a strongly skewed
    # next-sign distribution, and all other contexts fall back to a
    # generic order-1 distribution, mimicking how real sparse structured
    # systems often work (most contexts are unremarkable; some are highly
    # regular).
    special_contexts = {}
    all_pairs = [(a, b) for a in vocab for b in vocab]
    rng.shuffle(all_pairs)
    for pair in all_pairs[:60]:  # 60 "special" trigram-level regularities
        preferred = rng.choice(vocab)
        special_contexts[pair] = preferred

    generic_weights = {s: rng.uniform(0.5, 1.5) for s in vocab}

    def next_sign(prev2, prev1, length_so_far, target_length):
        if length_so_far >= target_length:
            return end_token
        key = (prev2, prev1)
        if key in special_contexts and rng.random() < 0.85:
            return special_contexts[key]
        signs = list(generic_weights.keys())
        weights = list(generic_weights.values())
        return rng.choices(signs, weights=weights, k=1)[0]

    inscriptions = []
    for i in range(n_inscriptions):
        target_length = max(1, min(14, round(rng.gauss(4.4, 1.8))))
        seq = [rng.choice(vocab)]
        if target_length >= 2:
            seq.append(rng.choice(vocab))
        prev2, prev1 = (None, seq[0]) if len(seq) == 1 else (seq[0], seq[1])
        while len(seq) < target_length:
            nxt = next_sign(prev2, prev1, len(seq), target_length)
            if nxt == end_token:
                break
            seq.append(nxt)
            prev2, prev1 = prev1, nxt

        inscriptions.append(Inscription(
            inscription_id=f"CIVD-{i:05d}", signs=seq,
            site=rng.choice(["SiteD1", "SiteD2"]),
            object_type=rng.choice(["seal", "tablet"]),
            reading_direction="R-L",
        ))
    return Corpus(inscriptions)


def generate_civilization_e_hierarchical_administrative(n_inscriptions: int = 1500, seed: int = 5) -> Corpus:
    """Category -> subcategory -> item, a purely nested bureaucratic
    structure. An item's distribution depends on its subcategory, which
    depends on its category two steps back: genuine multi-level
    dependency through filing-system logic, no linguistic motivation."""
    rng = random.Random(seed)
    n_categories = 4
    n_subcats_per_category = 3
    n_items_per_subcat = 6

    categories = _mk_signs("CAT", n_categories)
    subcats = {cat: _mk_signs(f"SUB{cat}", n_subcats_per_category) for cat in categories}
    items = {}
    for cat in categories:
        for sub in subcats[cat]:
            items[sub] = _mk_signs(f"ITM{sub}", n_items_per_subcat)

    inscriptions = []
    for i in range(n_inscriptions):
        cat = rng.choice(categories)
        sub = rng.choice(subcats[cat])
        item = rng.choice(items[sub])

        # a record is [category, subcategory, item], optionally repeated
        # or extended with a second category-subcategory-item triple,
        # mimicking a multi-entry administrative record
        seq = [cat, sub, item]
        if rng.random() < 0.35:
            cat2 = rng.choice(categories)
            sub2 = rng.choice(subcats[cat2])
            item2 = rng.choice(items[sub2])
            seq += [cat2, sub2, item2]

        inscriptions.append(Inscription(
            inscription_id=f"CIVE-{i:05d}", signs=seq,
            site=rng.choice(["SiteE1", "SiteE2"]),
            object_type=rng.choice(["seal", "tablet", "copper_plate"]),
            reading_direction="R-L",
        ))
    return Corpus(inscriptions)


EXTENDED_GENERATORS = {
    "civ_d_markov_no_morphology": generate_civilization_d_markov_no_morphology,
    "civ_e_hierarchical_administrative": generate_civilization_e_hierarchical_administrative,
}
