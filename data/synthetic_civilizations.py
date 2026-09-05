"""
data/synthetic_civilizations.py
=================================
"Unit testing for decipherment": generate corpora from THREE known
generative systems, so the analysis pipeline's ability to tell them apart
using only surface statistics can be measured against ground truth.

  Civilization A -- "language-like" (Dravidian-inspired):
      agglutinative morphology: ROOT + zero-or-more SUFFIX signs, drawn
      from distinct root/suffix sign pools, with suffix choice weakly
      conditioned on root (mimics noun-class/gender agreement) and a
      preference for SOV-like final position for one suffix class
      (mimics a case/classifier marker). Real, if simplified, grammar.

  Civilization B -- "non-linguistic administrative code":
      fixed-slot record format: [CATEGORY][LOCATION][QUANTITY/AUTHORITY],
      each slot drawn independently from its own small fixed vocabulary.
      No agreement, no long-range dependency, low combinatorial freedom
      -- deliberately looks like a "heraldic/administrative" system per
      Farmer-Sproat-Witzel's non-linguistic hypothesis.

  Civilization C -- "mixed system":
      a coin flip per inscription between a template slot-filler (like B)
      and a short agglutinative string (like A), plus a separate pool of
      "numeral" signs that can be inserted almost anywhere. Meant to
      stress-test whether the pipeline can detect when a corpus is NOT
      cleanly one thing or the other -- arguably the closest analogy to
      what the real Indus corpus's ambiguous position in the literature
      suggests.

None of these are claims about the real Indus script. They exist so that
analysis/falsification.py has labeled data to validate against.
"""
from __future__ import annotations
import random
from data.loader import Corpus, Inscription


def _mk_signs(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(n)]


def generate_civilization_a(n_inscriptions: int = 1500, seed: int = 1) -> Corpus:
    """Agglutinative, language-like: ROOT (+ up to 2 suffixes), with a
    noun-class-like dependency (root pool index determines which suffix
    pool is preferred) and a final case-marker-like slot."""
    rng = random.Random(seed)
    roots = _mk_signs("RTA", 40)
    suffix_pools = [_mk_signs(f"SFA{c}", 6) for c in range(4)]  # 4 "noun classes"
    case_markers = _mk_signs("CMA", 3)
    root_class = {r: i % 4 for i, r in enumerate(roots)}

    inscriptions = []
    for i in range(n_inscriptions):
        root = rng.choice(roots)
        cls = root_class[root]
        seq = [root]
        n_suffixes = rng.choices([0, 1, 2], weights=[0.2, 0.55, 0.25])[0]
        for _ in range(n_suffixes):
            # 85% of the time draw from the "agreeing" class pool, else
            # from a random other pool (irregular / borrowed forms)
            pool = suffix_pools[cls] if rng.random() < 0.85 else rng.choice(suffix_pools)
            seq.append(rng.choice(pool))
        if rng.random() < 0.6:
            seq.append(rng.choice(case_markers))

        inscriptions.append(Inscription(
            inscription_id=f"CIVA-{i:05d}", signs=seq,
            site=rng.choice(["SiteA1", "SiteA2", "SiteA3"]),
            object_type=rng.choices(["seal", "tablet"], weights=[0.7, 0.3])[0],
            reading_direction="R-L",
        ))
    return Corpus(inscriptions)


def generate_civilization_b(n_inscriptions: int = 1500, seed: int = 2) -> Corpus:
    """Fixed-slot administrative code: [CATEGORY][LOCATION][AUTHORITY],
    each independently drawn, no cross-slot dependency, some slots
    optional. This is the "heraldic/emblem" non-linguistic null model."""
    rng = random.Random(seed)
    categories = _mk_signs("CATB", 8)
    locations = _mk_signs("LOCB", 12)
    authorities = _mk_signs("AUTB", 5)

    inscriptions = []
    for i in range(n_inscriptions):
        seq = [rng.choice(categories)]
        if rng.random() < 0.7:
            seq.append(rng.choice(locations))
        if rng.random() < 0.5:
            seq.append(rng.choice(authorities))
        inscriptions.append(Inscription(
            inscription_id=f"CIVB-{i:05d}", signs=seq,
            site=rng.choice(["SiteB1", "SiteB2"]),
            object_type=rng.choices(["seal", "tablet", "copper_plate"],
                                     weights=[0.5, 0.3, 0.2])[0],
            reading_direction="R-L",
        ))
    return Corpus(inscriptions)


def generate_civilization_c(n_inscriptions: int = 1500, seed: int = 3) -> Corpus:
    """Mixed: per-inscription coin flip between a B-style template and an
    A-style agglutinative string, plus numerals insertable almost anywhere."""
    rng = random.Random(seed)
    numerals = _mk_signs("NUMC", 9)
    civ_a = generate_civilization_a(n_inscriptions, seed=seed + 100)
    civ_b = generate_civilization_b(n_inscriptions, seed=seed + 200)

    inscriptions = []
    for i in range(n_inscriptions):
        base = rng.choice(civ_a.inscriptions if rng.random() < 0.5 else civ_b.inscriptions)
        seq = list(base.signs)
        if rng.random() < 0.3:
            pos = rng.randint(0, len(seq))
            seq.insert(pos, rng.choice(numerals))
        inscriptions.append(Inscription(
            inscription_id=f"CIVC-{i:05d}", signs=seq,
            site=rng.choice(["SiteC1", "SiteC2"]),
            object_type=rng.choice(["seal", "tablet", "pottery"]),
            reading_direction="R-L",
        ))
    return Corpus(inscriptions)


GENERATORS = {
    "civ_a_language_like": generate_civilization_a,
    "civ_b_administrative_code": generate_civilization_b,
    "civ_c_mixed": generate_civilization_c,
}
