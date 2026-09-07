"""
tests/test_loader.py
=======================
Basic contract tests for data/loader.py's CSV schema, including every
field added over the course of this project (motif, mean_uncertainty)
that a future refactor could silently drop.
"""
import csv
from data.loader import Corpus, Inscription, save_corpus_csv, load_corpus_csv


def test_inscription_normalized_signs_reverses_only_for_l_r():
    rl = Inscription(inscription_id="x", signs=["A", "B", "C"], reading_direction="R-L")
    lr = Inscription(inscription_id="y", signs=["A", "B", "C"], reading_direction="L-R")
    assert rl.normalized_signs() == ["A", "B", "C"]
    assert lr.normalized_signs() == ["C", "B", "A"]


def test_corpus_filter_excludes_damaged(tiny_corpus):
    tiny_corpus.inscriptions[0].damaged = True
    filtered = tiny_corpus.filter(exclude_damaged=True)
    assert all(not ins.damaged for ins in filtered.inscriptions)
    assert len(filtered) == len(tiny_corpus) - 1


def test_csv_round_trip_preserves_all_schema_fields(tmp_path, tiny_corpus):
    """Regression guard: every field in the schema (including motif and
    mean_uncertainty, both added after the original schema was written)
    must survive a save -> load round trip. A future edit to
    save_corpus_csv or load_corpus_csv that forgets one of these fields
    would silently corrupt every corpus this project ships."""
    tiny_corpus.inscriptions[0].motif = "TestMotif"
    tiny_corpus.inscriptions[0].mean_uncertainty = 12.5

    path = tmp_path / "roundtrip.csv"
    save_corpus_csv(tiny_corpus, path)
    reloaded = load_corpus_csv(path)

    assert len(reloaded) == len(tiny_corpus)
    original = tiny_corpus.inscriptions[0]
    restored = next(i for i in reloaded.inscriptions if i.inscription_id == original.inscription_id)
    assert restored.signs == original.signs
    assert restored.site == original.site
    assert restored.object_type == original.object_type
    assert restored.motif == "TestMotif"
    assert abs(restored.mean_uncertainty - 12.5) < 1e-6
    assert restored.reading_direction == original.reading_direction


def test_csv_missing_optional_columns_default_sensibly(tmp_path):
    """A CSV lacking the newer optional columns (motif, mean_uncertainty)
    -- e.g. one hand-written before those fields existed -- must still
    load without crashing, defaulting to 'unknown' / 0.0."""
    path = tmp_path / "minimal.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["inscription_id", "sign_sequence"])
        writer.writerow(["M1", "A B C"])
    corpus = load_corpus_csv(path)
    assert len(corpus) == 1
    ins = corpus.inscriptions[0]
    assert ins.signs == ["A", "B", "C"]
    assert ins.motif == "unknown"
    assert ins.mean_uncertainty == 0.0


def test_vocab_and_summary(tiny_corpus):
    vocab = tiny_corpus.vocab()
    assert set(vocab) == {"A", "B", "C", "D", "X", "Y"}
    summary = tiny_corpus.summary()
    assert summary["n_inscriptions"] == 5
    assert summary["n_unique_signs"] == 6
