"""
analysis/sign_embeddings.py
==============================
Contextual sign embeddings via PPMI (positive pointwise mutual
information) + truncated SVD, the classic distributional-semantics
technique (essentially LSA applied to signs instead of words). Chosen
over a neural embedding (e.g. reading out models/transformer_mlm.py's
embedding table) deliberately: PPMI+SVD has no training dynamics to
second-guess, no risk of a toy-scale model's embeddings reflecting
optimization noise rather than real distributional structure, and is
fully deterministic given the corpus and a window size.

The point of building a second, independent method for finding sign
classes is cross-validation against analysis/substitution_graph.py's
minimal-pair-derived communities. Minimal pairs ask "do these two signs
substitute for each other in an otherwise-identical context?" (a strict,
local test). Distributional embeddings ask "do these two signs tend to
appear near similar OTHER signs, in general, across the whole corpus?"
(a looser, global test). These are different questions that could
disagree; if a substitution-graph community's members also cluster
together in embedding space, that is real convergent evidence for a
functional class, found two structurally different ways, not an
artifact of either method alone.

Method:
  1. Build a symmetric co-occurrence count matrix: for each sign, count
     how often every other sign appears within `window` positions of it,
     anywhere in the corpus (both directions).
  2. Convert counts to PPMI: pmi(i,j) = log2( P(i,j) / (P(i)*P(j)) ),
     clipped at 0 (the "positive" in PPMI -- negative PMI, meaning two
     signs co-occur LESS than chance, is set to 0 rather than kept,
     since negative PMI estimates are notoriously unreliable on sparse
     counts and the sign of a rare co-occurrence deficit is not
     trustworthy here).
  3. Truncated SVD on the PPMI matrix to a small number of dimensions
     (default 20 -- this corpus has ~600 signs, so 20 dimensions is
     already a substantial compression, not an arbitrary choice).
  4. Cosine similarity between the resulting dense vectors is the
     embedding-based notion of "these signs behave similarly."
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SignEmbeddings:
    signs: list[str]
    sign_to_index: dict[str, int]
    vectors: np.ndarray  # (n_signs, n_dims)

    def similarity(self, sign_a: str, sign_b: str) -> float:
        if sign_a not in self.sign_to_index or sign_b not in self.sign_to_index:
            return float("nan")
        va = self.vectors[self.sign_to_index[sign_a]]
        vb = self.vectors[self.sign_to_index[sign_b]]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def nearest_neighbors(self, sign: str, top_n: int = 5) -> list[tuple[str, float]]:
        if sign not in self.sign_to_index:
            return []
        sims = [(other, self.similarity(sign, other)) for other in self.signs if other != sign]
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[:top_n]


def build_cooccurrence_matrix(sequences: list[list[str]], window: int = 2) -> tuple[list[str], np.ndarray]:
    vocab = sorted(set(sign for seq in sequences for sign in seq))
    index = {s: i for i, s in enumerate(vocab)}
    n = len(vocab)
    counts = np.zeros((n, n), dtype=np.float64)

    for seq in sequences:
        L = len(seq)
        for i, s in enumerate(seq):
            si = index[s]
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if j == i:
                    continue
                sj = index[seq[j]]
                counts[si, sj] += 1.0

    return vocab, counts


def compute_ppmi(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return counts
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    p_ij = counts / total
    p_i = row_sums / total
    p_j = col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_ij / (p_i * p_j))
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(pmi, 0.0)  # PPMI: clip negative PMI to zero


def train_sign_embeddings(sequences: list[list[str]], window: int = 2, n_dims: int = 20,
                            seed: int = 0) -> SignEmbeddings:
    vocab, counts = build_cooccurrence_matrix(sequences, window=window)
    ppmi = compute_ppmi(counts)

    n_dims_eff = min(n_dims, min(ppmi.shape) - 1) if min(ppmi.shape) > 1 else 1
    rng = np.random.default_rng(seed)
    try:
        # truncated SVD via numpy's full SVD (this vocab size, ~600, makes
        # a full SVD entirely tractable; no need for a randomized/truncated
        # SVD algorithm at this scale)
        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
        vectors = U[:, :n_dims_eff] * S[:n_dims_eff]
    except np.linalg.LinAlgError:
        vectors = rng.normal(0, 0.01, (len(vocab), n_dims_eff))

    return SignEmbeddings(signs=vocab, sign_to_index={s: i for i, s in enumerate(vocab)}, vectors=vectors)
