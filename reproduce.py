"""
reproduce.py
===============
Runs the project's headline checks and reports PASS/FAIL/SKIPPED against
expected values, rather than just running scripts and leaving you to
read the numbers yourself. SKIPPED (not FAILED) is used for anything
that needs a real corpus not shipped in this repository (M77, ETCSL,
Sanskrit, Tamil -- see CORPUS_REGISTRY.md and CITATIONS.md for why);
those checks run automatically if you've regenerated the corresponding
CSV locally, and are skipped cleanly otherwise.

Expected runtime: under 2 minutes for the checks that run on a fresh
clone with no extra data; add roughly 1-2 more minutes if you've also
regenerated M77.

Usage:
    python3 reproduce.py
"""
from __future__ import annotations
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = []


def check(name: str, condition: bool, detail: str = "", skipped: bool = False):
    if skipped:
        status = "SKIPPED"
    else:
        status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"  {status:8s} {name}" + (f"  ({detail})" if detail else ""))


def within(value: float, expected: float, tol: float) -> bool:
    return abs(value - expected) <= tol


def order_3_gain(sequences, discount=0.75, k_folds=5):
    from analysis.ngram import kn_cross_validated_perplexity
    h2 = math.log2(kn_cross_validated_perplexity(sequences, n=2, k_folds=k_folds, discount=discount)["mean_perplexity"])
    h3 = math.log2(kn_cross_validated_perplexity(sequences, n=3, k_folds=k_folds, discount=discount)["mean_perplexity"])
    return h2 - h3


def section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    t0 = time.time()
    from data.loader import load_corpus_csv, Corpus

    section("CORPUS AUDIT")
    indus_path = Path("data/indus_website_real_corpus.csv")
    cisi_path = Path("data/cisi_real_corpus.csv")
    check("indus_website corpus file present", indus_path.exists())
    check("CISI primary corpus file present", cisi_path.exists())

    if indus_path.exists():
        indus = load_corpus_csv(str(indus_path)).filter(exclude_damaged=True)
        check("indus_website: N == 2,543", len(indus) == 2543, f"got {len(indus)}")
        check("indus_website: vocabulary == 592 signs", len(indus.vocab()) == 592, f"got {len(indus.vocab())}")
    else:
        indus = None

    m77_path = Path("data/m77_indusscript_real_corpus.csv")
    m77_present = m77_path.exists()
    if m77_present:
        m77 = load_corpus_csv(str(m77_path)).filter(exclude_damaged=True)
        check("M77: N == 3,573", len(m77) == 3573, f"got {len(m77)}")
        check("M77: vocabulary == 418 signs", len(m77.vocab()) == 418, f"got {len(m77.vocab())}")
    else:
        m77 = None
        check("M77 corpus", False, "not present locally -- see CORPUS_REGISTRY.md", skipped=True)

    section("DIRECTION TEST")
    from analysis.direction_test import test_reading_direction
    if indus is not None:
        result = test_reading_direction(load_corpus_csv(str(indus_path)))
        check("indus_website: as-stored direction is correct", result.likely_direction == "as-stored")
    if m77 is not None:
        result = test_reading_direction(load_corpus_csv(str(m77_path)))
        check("M77: as-stored direction is correct", result.likely_direction == "as-stored")

    section("ORDER-3 DEPENDENCY TEST (the headline result)")
    from data.permutation_nulls import bigram_markov_null
    from data.adversarial_null_model import generate_matched_null_corpus

    if indus is not None:
        seqs = indus.sequences(normalized=True)
        gain = order_3_gain(seqs)
        check("indus_website: real order-3 gain ~ +0.143", within(gain, 0.143, 0.03), f"got {gain:+.3f}")

        bigram_null = bigram_markov_null(load_corpus_csv(str(indus_path)), n_inscriptions=len(indus), seed=42)
        bigram_gain = order_3_gain(bigram_null.sequences(normalized=True))
        check("indus_website: bigram-order null is negative", bigram_gain < 0, f"got {bigram_gain:+.3f}")

        adv_null = generate_matched_null_corpus(load_corpus_csv(str(indus_path)), n_inscriptions=len(indus), seed=42)
        adv_gain = order_3_gain(adv_null.sequences(normalized=True))
        check("indus_website: adversarial null is negative", adv_gain < 0, f"got {adv_gain:+.3f}")

    if m77 is not None:
        seqs = m77.sequences(normalized=True)
        gain = order_3_gain(seqs)
        check("M77: real order-3 gain ~ +0.158", within(gain, 0.158, 0.03), f"got {gain:+.3f}")
    else:
        check("M77 order-3 test", False, "M77 not present locally", skipped=True)

    section("UNIQUENESS TEST")
    if indus is not None:
        seqs = indus.sequences(normalized=True)
        real_u = len(set(tuple(s) for s in seqs)) / len(seqs)
        null_us = []
        for trial in range(20):  # reduced from the full 200 for runtime; see experiments/uniqueness_significance_test.py for the full version
            null_corpus = generate_matched_null_corpus(load_corpus_csv(str(indus_path)), n_inscriptions=len(indus), seed=trial)
            null_seqs = null_corpus.sequences(normalized=True)
            null_us.append(len(set(tuple(s) for s in null_seqs)) / len(null_seqs))
        mean_null = sum(null_us) / len(null_us)
        check("indus_website: real uniqueness is below null mean", real_u < mean_null,
              f"real={real_u:.3f}, null mean={mean_null:.3f}")

    section("AUTOMATED TEST SUITE")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                                  capture_output=True, text=True, timeout=120)
        check("pytest suite (28 tests)", result.returncode == 0,
              "all passed" if result.returncode == 0 else result.stdout[-300:])
    except FileNotFoundError:
        check("pytest suite", False, "pytest not installed", skipped=True)

    section("EXTERNAL LANGUAGE CALIBRATION (skipped if not regenerated locally)")
    for label, path in [("ETCSL (Sumerian)", "data/etcsl_real_corpus.csv"),
                         ("DCS (Sanskrit)", "data/dcs_sanskrit_real_corpus.csv"),
                         ("Sangam (Tamil)", "data/tamil_real_corpus.csv")]:
        if Path(path).exists():
            corpus = load_corpus_csv(path).filter(exclude_damaged=True)
            check(f"{label}: file loads", len(corpus) > 0, f"N={len(corpus)}")
        else:
            check(f"{label}", False, "not present locally, see CORPUS_REGISTRY.md", skipped=True)

    section("SUMMARY")
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIPPED")
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped, in {time.time()-t0:.1f}s")
    if n_fail > 0:
        print("\n  REPRODUCTION STATUS: FAIL")
        print("  A FAIL here on a fresh clone with unmodified data is worth reporting as a")
        print("  potential bug -- see CLAIM_EVIDENCE_MATRIX.md for what each check is meant to show.")
        sys.exit(1)
    else:
        print("\n  REPRODUCTION STATUS: PASS" + (" (with skips -- see above)" if n_skip else ""))


if __name__ == "__main__":
    main()
