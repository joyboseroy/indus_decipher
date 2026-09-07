"""
tests/test_falsification.py
==============================
Tests for analysis/falsification.py. Includes a regression test for the
paradigm_classes_per_1000_inscriptions bug (a feature that silently
encoded sample size rather than the property it was named after) by
checking the CURRENT feature set doesn't reproduce that failure mode:
extracted features for the same underlying corpus at different sizes
should not show one feature swinging wildly (>10x) while all others stay
flat, which was the exact signature that exposed the original bug.
"""
from data.loader import Corpus
from data.synthetic_civilizations import GENERATORS
from analysis.falsification import extract_features, classify_by_nearest_centroid, FEATURE_NAMES


def test_feature_names_no_longer_includes_removed_paradigm_feature():
    """Direct regression guard: the buggy feature must not be reintroduced."""
    assert "paradigm_classes_per_1000_inscriptions" not in FEATURE_NAMES


def test_extract_features_runs_on_small_corpus(tiny_corpus):
    fv = extract_features(tiny_corpus)
    for name in FEATURE_NAMES:
        value = getattr(fv, name)
        assert value == value, f"{name} is NaN on a small but valid corpus"


def test_self_test_synthetic_civilizations_are_distinguishable():
    """Smoke test for the falsification harness's core claim: the three
    synthetic civilizations should be at least broadly separable from
    each other, not a basic sanity check that would fail if the
    generators or feature extraction were badly broken."""
    reference = {
        label: [extract_features(gen_fn(n_inscriptions=200, seed=100 + s)) for s in range(3)]
        for label, gen_fn in GENERATORS.items()
    }
    correct = 0
    total = 0
    for true_label, gen_fn in GENERATORS.items():
        query = extract_features(gen_fn(n_inscriptions=200, seed=999))
        train_set = {label: fvs for label, fvs in reference.items()}
        result = classify_by_nearest_centroid(query, train_set)
        total += 1
        correct += int(result["predicted_label"] == true_label)
    accuracy = correct / total
    assert accuracy >= 2 / 3, f"Self-test accuracy {accuracy:.2f} suggests the harness is broken"


def test_feature_extraction_size_stability():
    """Regression guard for the exact failure mode that exposed the
    original paradigm-class bug: extract features from the SAME
    underlying generator at two very different sizes, and check that no
    single feature swings by more than 5x while the corpus's actual
    generative process hasn't changed. (5x is a deliberately loose bound
    -- small samples ARE noisier -- but the original bug showed a ~40x
    swing, so this bound would have caught it.)"""
    gen_fn = GENERATORS["civ_a_language_like"]
    small = extract_features(gen_fn(n_inscriptions=50, seed=1))
    large = extract_features(gen_fn(n_inscriptions=2000, seed=1))
    for name in FEATURE_NAMES:
        small_val, large_val = getattr(small, name), getattr(large, name)
        if abs(small_val) < 1e-6 and abs(large_val) < 1e-6:
            continue  # both ~zero, ratio undefined but not a problem
        ratio = max(abs(small_val), abs(large_val), 1e-6) / max(min(abs(small_val), abs(large_val)), 1e-6)
        assert ratio < 5.0, (
            f"Feature '{name}' swings {ratio:.1f}x between N=50 and N=2000 on the SAME "
            f"generator. This is the exact signature that exposed the original "
            f"paradigm_classes_per_1000_inscriptions bug -- investigate before trusting "
            f"this feature."
        )
