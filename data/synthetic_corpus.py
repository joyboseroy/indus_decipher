"""
data/synthetic_corpus.py
=========================
Generates a SYNTHETIC test corpus that reproduces, by construction, several
statistical regularities reported in the literature:

  - Zipf-Mandelbrot unigram rank-frequency scaling (Yadav et al. 2010)
  - A small set of strongly preferred text-initial signs (e.g. #3, #10)
  - A single dominant text-final sign, the "jar" sign (#342), which rarely
    appears initially, and which sometimes doubles (sign repeated twice)
  - Non-trivial but non-random bigram structure (some transitions ~0.8)
  - Inscription lengths clustered around 4-5 signs, max ~14-26

THIS IS NOT REAL DATA. It exists purely so every module in this repo can
be run and unit-tested end to end before real digitized inscriptions are
plugged in via data/loader.py. Do not use conclusions drawn from this
corpus as evidence about the actual Indus script.
"""
from __future__ import annotations
import random
from .loader import Corpus, Inscription

N_SIGNS = 120          # smaller than the real ~417/584/676 sign catalogs, for speed
JAR_SIGN = "S342"      # stand-in for Mahadevan's #342


def _zipf_mandelbrot_probs(n_signs: int, q: float = 2.7, gamma: float = 1.15) -> list[float]:
    ranks = list(range(1, n_signs + 1))
    weights = [1.0 / ((r + q) ** gamma) for r in ranks]
    total = sum(weights)
    return [w / total for w in weights]


def generate_synthetic_corpus(
    n_inscriptions: int = 2000,
    n_signs: int = N_SIGNS,
    seed: int = 42,
) -> Corpus:
    rng = random.Random(seed)
    signs = [f"S{i:03d}" for i in range(1, n_signs + 1)]
    unigram_probs = _zipf_mandelbrot_probs(n_signs)

    # Rank 1 sign (most frequent) plays the role of the "jar" sign: dominant,
    # text-final, near-absent initially, occasionally doubled.
    jar = signs[0]
    # A handful of moderately-frequent signs get a strong preference for
    # the initial slot (mirrors #3 / #10 behaviour).
    initial_favored = signs[2:5]

    # Build a sparse, biased bigram preference table so transitions aren't
    # uniform (mirrors "moderately frequent pairs with P~0.8").
    strong_bigrams = {}
    for s in signs:
        if rng.random() < 0.15:
            strong_bigrams[s] = rng.choice(signs)

    inscriptions = []
    for i in range(n_inscriptions):
        length = max(1, min(26, round(rng.gauss(4.4, 2.0))))
        seq = []

        # initial sign
        if rng.random() < 0.55:
            seq.append(rng.choice(initial_favored))
        else:
            seq.append(rng.choices(signs, weights=unigram_probs, k=1)[0])

        # middle signs, with occasional strong bigram continuation
        while len(seq) < max(1, length - 1):
            prev = seq[-1]
            if prev in strong_bigrams and rng.random() < 0.8:
                nxt = strong_bigrams[prev]
            else:
                nxt = rng.choices(signs, weights=unigram_probs, k=1)[0]
            seq.append(nxt)

        # final sign: jar sign dominates the text-final slot
        if length >= 2:
            if rng.random() < 0.45:
                seq.append(jar)
                if rng.random() < 0.33:  # jar doubling, ~33% as in Rao et al.
                    seq.append(jar)
            else:
                seq.append(rng.choices(signs, weights=unigram_probs, k=1)[0])

        damaged = rng.random() < 0.08
        line_count = 1 if rng.random() < 0.9 else 2
        site = rng.choice(["Mohenjo-daro", "Harappa", "Lothal", "Dholavira", "Kalibangan"])
        object_type = rng.choices(
            ["seal", "tablet", "pottery", "copper_plate", "ivory"],
            weights=[0.55, 0.2, 0.15, 0.06, 0.04], k=1,
        )[0]

        inscriptions.append(Inscription(
            inscription_id=f"SYN-{i:05d}",
            signs=seq,
            site=site,
            object_type=object_type,
            line_count=line_count,
            damaged=damaged,
            reading_direction="R-L",
        ))

    return Corpus(inscriptions)


if __name__ == "__main__":
    from .loader import save_corpus_csv
    c = generate_synthetic_corpus()
    print(c.summary())
    save_corpus_csv(c, "example_corpus_synthetic.csv")
