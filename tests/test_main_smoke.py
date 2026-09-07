"""
tests/test_main_smoke.py
===========================
End-to-end smoke tests: does the full pipeline run without crashing?
These deliberately don't depend on the real data CSVs existing (a clone
of this repo might not have regenerated them yet), so they run
main.py's logic directly against the synthetic corpus, which is always
available.

Full runs against the real corpora (large + both CISI granularities) are
NOT duplicated here as automated tests, since they're slow (multiple
minutes with --extended) and are already covered by this project's
established practice of manually rerunning every script before each
delivery. This suite exists to catch a broken IMPORT or a crashed
function fast, not to replace that manual full-pipeline check.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_main_runs_on_synthetic_corpus():
    result = subprocess.run(
        [sys.executable, "main.py", "--n_inscriptions", "100"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"main.py failed:\n{result.stderr[-3000:]}"
    assert "Full report written to" in result.stdout


def test_falsification_module_runs_standalone():
    result = subprocess.run(
        [sys.executable, "-m", "analysis.falsification"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"analysis.falsification failed:\n{result.stderr[-3000:]}"
    assert "Overall accuracy" in result.stdout
