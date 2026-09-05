"""
data/loader.py
================
Defines the corpus schema and loads inscriptions from CSV/JSON into
a standard in-memory representation used by every downstream module.

SCHEMA
------
Each inscription is a row with:
    inscription_id   : str   unique ID (e.g. "M-1217", "H-935")
    sign_sequence    : str   space-separated sign codes, LEFT-TO-RIGHT
                              in file order (direction is normalized on load)
    site             : str   findspot, e.g. "Mohenjo-daro", "Harappa"
    object_type      : str   "seal", "tablet", "pottery", "copper_plate", "ivory", ...
    line_count       : int   number of physical lines the text spans
    damaged          : bool  True if the reading contains lost/illegible signs
    reading_direction: str   "R-L" (right-to-left, the Indus convention) or "L-R"
    motif            : str   iconographic field-symbol classification, e.g.
                              "Bull1:W", "Gaur", "Elep", "unicorn" -- OPTIONAL,
                              defaults to "unknown" when the source data
                              doesn't carry motif/iconography information.
                              Used by analysis/minimal_pairs.py for stronger
                              seal-twin corroboration (see that module).

WHERE TO GET REAL DATA
-----------------------
This repo ships with a small SYNTHETIC example (synthetic_corpus.py) that
mimics published statistical properties (Zipf-Mandelbrot unigram scaling,
positional asymmetry, sign #342 "jar" behavior) purely so the pipeline can
be tested end to end. It is NOT real epigraphic data and must not be used
to draw conclusions about the actual script.

To do real work, obtain one of:
  1. Mahadevan's M77 concordance / the EBUDS filtered subset (2906 texts,
     417 signs) - referenced in Yadav et al. 2010, PLOS ONE. Contact the
     Indus Research Centre, Roja Muthiah Research Library (also behind
     the indusscript.in web app), or the paper's supplementary materials.
  2. The Interactive Corpus of Indus Texts (ICIT), Wells & Fuls - request
     access via the contact info at user.tu-berlin.de/fuls/Homepage/indus/
  3. Wells & Fuls (2023) W09IMSc / WUCS datasets, described in Rao et al.
     2009 PNAS follow-up network-analysis papers.

Once you have real data, export it to CSV matching the schema above and
point load_corpus() at the file. Nothing else in this codebase needs to
change.
"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Inscription:
    inscription_id: str
    signs: List[str]
    site: str = "unknown"
    object_type: str = "unknown"
    line_count: int = 1
    damaged: bool = False
    reading_direction: str = "R-L"
    motif: str = "unknown"

    def normalized_signs(self) -> List[str]:
        """Return signs in canonical reading order (right-to-left convention,
        i.e. the order the sign was actually uttered/written), regardless of
        how the source file stored them."""
        if self.reading_direction == "L-R":
            return list(reversed(self.signs))
        return list(self.signs)


@dataclass
class Corpus:
    inscriptions: List[Inscription] = field(default_factory=list)

    def __len__(self):
        return len(self.inscriptions)

    def sequences(self, normalized: bool = True) -> List[List[str]]:
        if normalized:
            return [ins.normalized_signs() for ins in self.inscriptions]
        return [ins.signs for ins in self.inscriptions]

    def filter(self, *, exclude_damaged=True, exclude_multiline=False, min_len=1) -> "Corpus":
        """Common preprocessing used in the literature (EBUDS-style filtering)."""
        kept = []
        for ins in self.inscriptions:
            if exclude_damaged and ins.damaged:
                continue
            if exclude_multiline and ins.line_count > 1:
                continue
            if len(ins.signs) < min_len:
                continue
            kept.append(ins)
        return Corpus(kept)

    def vocab(self) -> List[str]:
        seen = set()
        for ins in self.inscriptions:
            seen.update(ins.signs)
        return sorted(seen)

    def summary(self) -> dict:
        lens = [len(ins.signs) for ins in self.inscriptions]
        return {
            "n_inscriptions": len(self.inscriptions),
            "n_unique_signs": len(self.vocab()),
            "total_sign_tokens": sum(lens),
            "mean_length": sum(lens) / len(lens) if lens else 0,
            "min_length": min(lens) if lens else 0,
            "max_length": max(lens) if lens else 0,
        }


def load_corpus_csv(path: str | Path) -> Corpus:
    """Load a corpus from CSV with columns matching the schema in this file's
    docstring. Missing optional columns fall back to sensible defaults."""
    inscriptions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signs = row["sign_sequence"].strip().split()
            inscriptions.append(Inscription(
                inscription_id=row.get("inscription_id", f"row{len(inscriptions)}"),
                signs=signs,
                site=row.get("site", "unknown") or "unknown",
                object_type=row.get("object_type", "unknown") or "unknown",
                line_count=int(row.get("line_count") or 1),
                damaged=str(row.get("damaged", "")).strip().lower() in ("1", "true", "yes"),
                reading_direction=row.get("reading_direction", "R-L") or "R-L",
                motif=row.get("motif", "unknown") or "unknown",
            ))
    return Corpus(inscriptions)


def load_corpus_json(path: str | Path) -> Corpus:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    inscriptions = []
    for row in raw:
        inscriptions.append(Inscription(
            inscription_id=row.get("inscription_id", f"row{len(inscriptions)}"),
            signs=list(row["sign_sequence"]) if isinstance(row["sign_sequence"], list)
                  else row["sign_sequence"].split(),
            site=row.get("site", "unknown"),
            object_type=row.get("object_type", "unknown"),
            line_count=int(row.get("line_count", 1)),
            damaged=bool(row.get("damaged", False)),
            reading_direction=row.get("reading_direction", "R-L"),
            motif=row.get("motif", "unknown"),
        ))
    return Corpus(inscriptions)


def save_corpus_csv(corpus: Corpus, path: str | Path) -> None:
    fieldnames = ["inscription_id", "sign_sequence", "site", "object_type",
                  "line_count", "damaged", "reading_direction", "motif"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ins in corpus.inscriptions:
            writer.writerow({
                "inscription_id": ins.inscription_id,
                "sign_sequence": " ".join(ins.signs),
                "site": ins.site,
                "object_type": ins.object_type,
                "line_count": ins.line_count,
                "damaged": ins.damaged,
                "reading_direction": ins.reading_direction,
                "motif": ins.motif,
            })
