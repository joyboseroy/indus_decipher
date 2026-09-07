"""
convert_etcsl_to_csv.py
==========================
Converts the real ETCSL (Electronic Text Corpus of Sumerian Literature,
Black et al. 1998-2006, Faculty of Oriental Studies, University of
Oxford) transliteration files into this toolkit's corpus CSV schema, for
use as a genuine external-language calibration corpus (see README's
"External validation" sections) -- a real attested language, not a
synthetic control, to compare the Indus corpus's statistics against.

SOURCE: https://etcsl.orinst.ox.ac.uk (394 compositions), obtained via
the Oxford Text Archive's successor repository. Licensed CC BY-NC-SA
3.0 (Black, Cunningham, Ebeling, Flu"ckiger-Hawker, Robson, Taylor,
Zo'lyomi).

METHODOLOGICAL CHOICES, made explicitly rather than left implicit:

  UNIT OF COMPARISON: each Sumerian LINE (<l> element) is treated as one
  "inscription," not each whole composition. Indus inscriptions are
  short (mean ~4.4 signs); ETCSL compositions run to hundreds of lines,
  but individual lines are clause-length and a much closer match to
  Indus inscription length. Comparing whole compositions to individual
  seal inscriptions would conflate very different units of analysis.

  SIGN IDENTITY: each <w> tag's `lemma` attribute (the standardized
  dictionary citation form) is used as the sign-equivalent token, NOT
  the `form` attribute (the exact inflected surface form). This is a
  real methodological choice, not a neutral default: lemma collapses
  inflectional variants together the way a fixed Indus sign catalog
  treats all instances of one sign as the same identity, whereas form
  would treat every inflected variant as a distinct token, inflating
  vocabulary size in a way that has no clean Indus analogue. Using
  `form` instead is a reasonable alternative for a robustness check
  (pass granularity="form" to convert()), analogous to this project's
  primary-vs-allograph granularity work on the CISI corpus.

  EXCLUDED TOKENS: lemma values "&X;", "X", and empty strings mark
  damaged, illegible, or unidentifiable text in the source and are
  dropped rather than treated as real sign tokens (comparable to this
  project's exclude_damaged filtering elsewhere, though here it is
  applied at the word level during conversion, not via the corpus
  schema's damaged flag).

  PARSING METHOD: regex-based extraction of <l>...</l> blocks and <w
  lemma="..."> attributes, not a full XML/DTD parse. The source files
  use a large custom SGML entity set (e.g. &j; &c; &d; for special
  Sumerian characters, &X; for illegible signs) that a standard XML
  parser cannot resolve without the original DTD/entity files; regex
  extraction of attribute values sidesteps this entirely rather than
  requiring the full (Windows-only, 2003-era) TEI toolchain shipped
  alongside the corpus.
"""
import csv
import glob
import re
import sys

LINE_PATTERN = re.compile(r'<l\s+n="[^"]*"[^>]*>(.*?)</l>', re.DOTALL)
W_PATTERN_LEMMA = re.compile(r'<w\s[^>]*\blemma="([^"]*)"[^>]*>')
W_PATTERN_FORM = re.compile(r'<w\s[^>]*\bform="([^"]*)"[^>]*>')
EXCLUDED_TOKENS = {"&X;", "X", "", "&amp;X;"}


def extract_composition_id(content: str) -> str:
    m = re.search(r'<TEI\.2\s+id="([^"]*)"', content)
    return m.group(1) if m else "unknown"


def convert(input_glob: str, output_csv: str, granularity: str = "lemma", min_tokens: int = 2):
    w_pattern = W_PATTERN_LEMMA if granularity == "lemma" else W_PATTERN_FORM
    rows = []
    n_files = 0
    n_lines_total = 0
    for path in sorted(glob.glob(input_glob)):
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        n_files += 1
        composition_id = extract_composition_id(content)

        line_blocks = LINE_PATTERN.findall(content)
        for i, block in enumerate(line_blocks):
            n_lines_total += 1
            tokens = [t for t in w_pattern.findall(block) if t not in EXCLUDED_TOKENS]
            if len(tokens) < min_tokens:
                continue
            rows.append({
                "inscription_id": f"{composition_id}.L{i+1}",
                "sign_sequence": " ".join(tokens),
                "site": composition_id,  # which composition this line belongs to
                "object_type": "literary_line",
                "line_count": 1,
                "damaged": False,  # damaged/illegible TOKENS already excluded above
                "reading_direction": "R-L",  # "R-L" in this schema means "keep as-stored";
                                              # Sumerian lines are already in natural
                                              # left-to-right reading order in the source
                "motif": "unknown",
                "mean_uncertainty": 0.0,
            })

    fieldnames = ["inscription_id", "sign_sequence", "site", "object_type",
                  "line_count", "damaged", "reading_direction", "motif", "mean_uncertainty"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {n_files} composition files, {n_lines_total} total lines, "
          f"wrote {len(rows)} usable lines (>= {min_tokens} tokens after excluding "
          f"damaged/illegible) to {output_csv} (granularity={granularity})")


if __name__ == "__main__":
    input_glob = sys.argv[1] if len(sys.argv) > 1 else "etcsl/transliterations/*.xml"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "etcsl_real_corpus.csv"
    granularity = sys.argv[3] if len(sys.argv) > 3 else "lemma"
    convert(input_glob, output_csv, granularity=granularity)
