"""
analysis/positional.py
========================
Unigram frequency analysis, Zipf-Mandelbrot fit, and positional
(text-initial vs text-final) sign distributions.

Replicates the core descriptive results of Yadav et al. (2010, PLOS ONE)
and Rao et al. (2009, PNAS): sign frequencies follow a Zipf-Mandelbrot
law, and initial/final sign distributions are markedly unequal, which is
itself evidence of syntactic constraint (a purely random ordering would
show no such asymmetry).
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit


def unigram_counts(sequences: list[list[str]]) -> Counter:
    c = Counter()
    for seq in sequences:
        c.update(seq)
    return c


def zipf_mandelbrot(r, C, q, gamma):
    return C / np.power(r + q, gamma)


@dataclass
class ZipfFitResult:
    C: float
    q: float
    gamma: float
    ranks: np.ndarray
    freqs: np.ndarray
    predicted: np.ndarray

    def r_squared(self) -> float:
        ss_res = np.sum((self.freqs - self.predicted) ** 2)
        ss_tot = np.sum((self.freqs - np.mean(self.freqs)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_zipf_mandelbrot(sequences: list[list[str]]) -> ZipfFitResult:
    counts = unigram_counts(sequences)
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    # Reasonable starting guesses; bounds keep the fit well-posed on small corpora.
    p0 = [freqs[0], 2.0, 1.0]
    try:
        popt, _ = curve_fit(zipf_mandelbrot, ranks, freqs, p0=p0, maxfev=10000,
                             bounds=([0, 0, 0.01], [np.inf, 50, 5]))
    except RuntimeError:
        popt = p0
    predicted = zipf_mandelbrot(ranks, *popt)
    return ZipfFitResult(C=popt[0], q=popt[1], gamma=popt[2],
                          ranks=ranks, freqs=freqs, predicted=predicted)


def positional_distribution(sequences: list[list[str]], position: str = "initial") -> Counter:
    """position: 'initial' or 'final'. Sequences are assumed already in
    canonical reading order (see loader.Inscription.normalized_signs)."""
    c = Counter()
    for seq in sequences:
        if not seq:
            continue
        c[seq[0] if position == "initial" else seq[-1]] += 1
    return c


def positional_asymmetry_report(sequences: list[list[str]], top_n: int = 10) -> dict:
    """Compares each sign's share of the initial slot vs its share of the
    final slot vs its overall frequency, to surface signs like the 'jar'
    sign that are frequent overall but concentrated at one position."""
    overall = unigram_counts(sequences)
    initial = positional_distribution(sequences, "initial")
    final = positional_distribution(sequences, "final")
    n_overall = sum(overall.values())
    n_initial = sum(initial.values())
    n_final = sum(final.values())

    rows = []
    for sign, total in overall.most_common(top_n):
        rows.append({
            "sign": sign,
            "overall_freq": total,
            "overall_share": total / n_overall,
            "initial_count": initial.get(sign, 0),
            "initial_share": initial.get(sign, 0) / n_initial if n_initial else 0,
            "final_count": final.get(sign, 0),
            "final_share": final.get(sign, 0) / n_final if n_final else 0,
        })
    return {"top_signs": rows, "n_initial": n_initial, "n_final": n_final}


def doubling_rate(sequences: list[list[str]], sign: str) -> dict:
    """What fraction of a sign's occurrences are part of an immediate
    repeat (s_i == s_{i+1})? Mirrors the 'jar sign doubles in 33/58
    occurrences' finding, which constrains any future phonetic reading."""
    total = 0
    doubled = 0
    for seq in sequences:
        for i, s in enumerate(seq):
            if s != sign:
                continue
            total += 1
            if (i + 1 < len(seq) and seq[i + 1] == sign) or (i - 1 >= 0 and seq[i - 1] == sign):
                doubled += 1
    return {"sign": sign, "total_occurrences": total, "doubled_occurrences": doubled,
            "doubling_rate": doubled / total if total else 0.0}
