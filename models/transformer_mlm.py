"""
models/transformer_mlm.py
===========================
A small masked-language-model transformer for sign sequences, implemented
in plain NumPy (forward + backward pass by hand) so this repo has zero
heavy ML dependencies and runs anywhere.

Task: given an inscription with one sign randomly masked, predict it from
bidirectional context (self-attention sees the whole sequence, both
directions -- unlike the n-gram models in analysis/ngram.py, which only
see left context). This mirrors the "adapt self-supervised pretrained
transformer architectures ... masking random signs ... bidirectional
self-attention" direction described in the literature review.

This is intentionally small (toy-scale: ~1-2 layers, d_model ~32-64) since
the real corpus has only a few thousand short inscriptions -- a large
transformer would badly overfit. The point is a working, inspectable
baseline you can grow once real data and more compute are available.

For production-scale experiments, swap this module for a PyTorch/JAX
implementation with the same interface (encode / train_step / predict) --
none of the other modules in this repo depend on the internals here.
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class Vocab:
    def __init__(self, sequences: list[list[str]]):
        signs = sorted(set(s for seq in sequences for s in seq))
        self.itos = ["<PAD>", "<MASK>"] + signs
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.size = len(self.itos)

    def encode(self, seq: list[str]) -> list[int]:
        return [self.stoi[s] for s in seq]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


class MaskedSignTransformer:
    """Single-head, single-layer transformer encoder with a masked-LM head.
    Small enough to train with plain SGD in pure NumPy in well under a
    minute on a corpus of a few thousand short sequences.
    """

    def __init__(self, vocab_size: int, d_model: int = 48, max_len: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        scale = 1.0 / np.sqrt(d_model)
        self.embed = rng.normal(0, scale, (vocab_size, d_model))
        self.pos_embed = rng.normal(0, scale, (max_len, d_model))
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))
        self.W1 = rng.normal(0, scale, (d_model, d_model * 2))
        self.b1 = np.zeros(d_model * 2)
        self.W2 = rng.normal(0, scale, (d_model * 2, d_model))
        self.b2 = np.zeros(d_model)
        self.Wout = rng.normal(0, scale, (d_model, vocab_size))
        self.bout = np.zeros(vocab_size)

    def _layernorm(self, x, eps=1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps)

    def forward(self, ids: np.ndarray):
        """ids: (seq_len,) int array. Returns logits (seq_len, vocab_size)
        and a cache dict of intermediates for the backward pass."""
        L = len(ids)
        x = self.embed[ids] + self.pos_embed[:L]           # (L, d)
        x_ln1 = self._layernorm(x)

        Q = x_ln1 @ self.Wq
        K = x_ln1 @ self.Wk
        V = x_ln1 @ self.Wv
        scores = Q @ K.T / np.sqrt(self.d_model)            # (L, L) full bidirectional attention
        attn = softmax(scores, axis=-1)
        attn_out = attn @ V
        attn_out = attn_out @ self.Wo
        x2 = x + attn_out                                    # residual
        x2_ln = self._layernorm(x2)

        h = np.maximum(0, x2_ln @ self.W1 + self.b1)         # ReLU FFN
        ffn_out = h @ self.W2 + self.b2
        x3 = x2 + ffn_out                                    # residual

        logits = x3 @ self.Wout + self.bout
        cache = dict(ids=ids, x=x, x_ln1=x_ln1, Q=Q, K=K, V=V, scores=scores,
                     attn=attn, attn_out=attn_out, x2=x2, x2_ln=x2_ln, h=h,
                     ffn_out=ffn_out, x3=x3)
        return logits, cache

    def loss_and_grads(self, ids: np.ndarray, mask_pos: int, target_id: int, lr: float):
        """Computes cross-entropy loss at mask_pos and takes one full-backprop
        SGD step through every layer (attention, FFN, embeddings). Layernorm
        is treated as a fixed rescaling for the backward pass (its own
        gradient is small relative to the rest at this scale and omitting it
        keeps the implementation readable); everything else is exact."""
        logits, cache = self.forward(ids)
        L = len(ids)
        probs = softmax(logits[mask_pos])
        loss = -np.log(probs[target_id] + 1e-9)

        # ---- output layer ----
        dlogits = probs.copy()
        dlogits[target_id] -= 1.0                      # (V,)
        x3_row = cache["x3"][mask_pos]                  # (d,)
        dWout = np.outer(x3_row, dlogits)
        dbout = dlogits
        dx3_row = dlogits @ self.Wout.T                 # (d,)

        # scatter gradient onto the full (L, d) x3 tensor: only mask_pos
        # received direct loss, everything else is zero at this step
        dx3 = np.zeros_like(cache["x3"])
        dx3[mask_pos] = dx3_row

        # ---- residual 2: x3 = x2 + ffn_out ----
        dffn_out = dx3.copy()
        dx2_a = dx3.copy()

        # ---- FFN: ffn_out = relu(x2_ln @ W1 + b1) @ W2 + b2 ----
        dh = dffn_out @ self.W2.T
        dh[cache["h"] <= 0] = 0.0                       # relu backward
        dW2 = cache["h"].T @ dffn_out
        db2 = dffn_out.sum(axis=0)
        dW1 = cache["x2_ln"].T @ dh
        db1 = dh.sum(axis=0)
        dx2_ln = dh @ self.W1.T

        # (layernorm gradient approximated as identity pass-through, see docstring)
        dx2_b = dx2_ln
        dx2 = dx2_a + dx2_b

        # ---- residual 1: x2 = x + attn_out ----
        dattn_out_full = dx2.copy()
        dx_a = dx2.copy()

        # ---- attn_out = (attn @ V) @ Wo ----
        dWo = (cache["attn"] @ cache["V"]).T @ dattn_out_full
        dattnV = dattn_out_full @ self.Wo.T             # (L, d)

        dattn = dattnV @ cache["V"].T                   # (L, L)
        dV = cache["attn"].T @ dattnV                   # (L, d)

        # softmax backward per row
        dscores = np.zeros_like(dattn)
        for i in range(L):
            a = cache["attn"][i]
            dscores[i] = a * (dattn[i] - np.dot(a, dattn[i]))

        dscores /= np.sqrt(self.d_model)
        dQ = dscores @ cache["K"]
        dK = dscores.T @ cache["Q"]

        dWq = cache["x_ln1"].T @ dQ
        dWk = cache["x_ln1"].T @ dK
        dWv = cache["x_ln1"].T @ dV
        dx_ln1 = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        # (layernorm gradient approximated as identity pass-through)
        dx_b = dx_ln1
        dx = dx_a + dx_b                                 # (L, d)

        # ---- gradient clipping (small model + small corpus can otherwise
        # take an occasional oversized step and diverge) ----
        def clip(g, max_norm=5.0):
            n = np.linalg.norm(g)
            return g * (max_norm / n) if n > max_norm else g

        # ---- embeddings ----
        for i, tok in enumerate(ids):
            self.embed[tok] -= lr * clip(dx[i])
        self.pos_embed[:L] -= lr * clip(dx)

        # ---- apply all weight/bias updates ----
        self.Wout -= lr * clip(dWout)
        self.bout -= lr * clip(dbout)
        self.W2 -= lr * clip(dW2)
        self.b2 -= lr * clip(db2)
        self.W1 -= lr * clip(dW1)
        self.b1 -= lr * clip(db1)
        self.Wo -= lr * clip(dWo)
        self.Wq -= lr * clip(dWq)
        self.Wk -= lr * clip(dWk)
        self.Wv -= lr * clip(dWv)

        return loss

    def predict(self, ids: np.ndarray, mask_pos: int) -> int:
        logits, _ = self.forward(ids)
        return int(np.argmax(logits[mask_pos]))


def train_mlm(model: MaskedSignTransformer, vocab: Vocab, sequences: list[list[str]],
              epochs: int = 20, lr: float = 0.05, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    losses = []
    encoded = [vocab.encode(s) for s in sequences if 1 <= len(s) <= model.max_len]
    for epoch in range(epochs):
        rng.shuffle(encoded)
        epoch_loss = 0.0
        n = 0
        for seq in encoded:
            if len(seq) < 2:
                continue
            ids = np.array(seq)
            mask_pos = rng.integers(0, len(ids))
            target_id = ids[mask_pos]
            masked_ids = ids.copy()
            masked_ids[mask_pos] = vocab.stoi["<MASK>"]
            loss = model.loss_and_grads(masked_ids, mask_pos, target_id, lr)
            epoch_loss += loss
            n += 1
        losses.append(epoch_loss / max(n, 1))
    return losses


def evaluate_mlm_accuracy(model: MaskedSignTransformer, vocab: Vocab,
                           sequences: list[list[str]], trials: int = 300, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    candidates = [s for s in sequences if 2 <= len(s) <= model.max_len]
    correct = 0
    for _ in range(trials):
        seq = candidates[rng.integers(0, len(candidates))]
        ids = np.array(vocab.encode(seq))
        mask_pos = int(rng.integers(0, len(ids)))
        target_id = ids[mask_pos]
        masked_ids = ids.copy()
        masked_ids[mask_pos] = vocab.stoi["<MASK>"]
        pred = model.predict(masked_ids, mask_pos)
        correct += int(pred == target_id)
    return correct / trials
