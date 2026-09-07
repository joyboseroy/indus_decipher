"""
convert_tamil_to_csv.py
==========================
Converts the real Sangam/Old Tamil corpus (starhopp3r/sangam, scraped
from Vaidehi Herbert's translations at sangamtranslationsbyvaidehi.com,
c. 300 BCE - 300 CE, 2,377 poems) into this toolkit's corpus schema, as
the third real external-language calibration corpus alongside ETCSL and
Sanskrit (see README's "Real external-language calibration" sections).

SOURCE: https://huggingface.co/datasets/starhopp3r/sangam
CITATION: see CITATIONS.md.

METHODOLOGICAL CHOICES, and why this one is weaker than ETCSL/Sanskrit,
stated plainly rather than glossed over:

  UNIT OF COMPARISON: each verse LINE (the dataset's own \\n-separated
  convention within `tamil_text`) is one "inscription," matching the
  line/sentence-level unit already used for ETCSL and Sanskrit.

  SIGN IDENTITY: whitespace-separated ORTHOGRAPHIC WORDS from the raw
  Tamil text, NOT lemmas. This is the real limitation flagged when this
  language was first investigated: unlike ETCSL (TEI-tagged lemmas) and
  the Digital Corpus of Sanskrit (CoNLL-U lemmas), no lemmatized or
  morphologically segmented digital Sangam corpus was found. Tamil is
  agglutinative with productive sandhi; a single orthographic word here
  can bundle a root plus multiple case/tense/person morphemes (e.g. a
  noun plus a locative case ending written as one unbroken word), the
  way real Indus signs never do. This means the resulting vocabulary
  and n-gram statistics are NOT directly comparable to ETCSL/Sanskrit's
  lemma-level statistics on equal terms, and any finding here should be
  read as "orthographic-word-level Tamil," not simply "Tamil," alongside
  the other two.

  NUMBERED VERSE LINES: the source text embeds occasional standalone
  verse-line numbers (e.g. a bare "5" every five lines) as a citation
  convention, not part of the poem. These are stripped before tokenizing
  (a token that, after stripping surrounding punctuation, consists
  entirely of digits and is 1-3 characters long is treated as a line
  number and dropped, which will occasionally and incorrectly drop a
  genuine short numeral word if one existed in the source -- a real,
  accepted trade-off for a simple, auditable rule over a fragile
  heuristic).

  PUNCTUATION: commas, semicolons, and similar marks are Vaidehi
  Herbert's own modern editorial additions for readability, not part of
  the historical text; stripped from token edges before use.
"""
import csv
import re
import sys

LINE_NUMBER_RE = re.compile(r"^\d{1,3}$")
PUNCT_STRIP_RE = re.compile(r"^[,;:.!?()\"'\u2018\u2019\u201c\u201d]+|[,;:.!?()\"'\u2018\u2019\u201c\u201d]+$")


def tokenize_line(line: str) -> list[str]:
    raw_tokens = line.strip().split()
    tokens = []
    for t in raw_tokens:
        cleaned = PUNCT_STRIP_RE.sub("", t).strip()
        if not cleaned:
            continue
        if LINE_NUMBER_RE.match(cleaned):
            continue
        tokens.append(cleaned)
    return tokens


def convert(input_csv: str, output_csv: str, min_tokens: int = 2):
    import pandas as pd
    df = pd.read_csv(input_csv)

    rows = []
    n_poems = 0
    n_lines_total = 0
    for _, poem in df.iterrows():
        n_poems += 1
        text = poem.get("tamil_text")
        if not isinstance(text, str) or not text.strip():
            continue
        poem_id = poem["id"]
        thinai = poem.get("thinai") if isinstance(poem.get("thinai"), str) else "unknown"
        work = poem.get("work") if isinstance(poem.get("work"), str) else "unknown"

        for i, line in enumerate(text.split("\n")):
            n_lines_total += 1
            tokens = tokenize_line(line)
            if len(tokens) < min_tokens:
                continue
            rows.append({
                "inscription_id": f"{poem_id}.L{i+1}",
                "sign_sequence": " ".join(tokens),
                "site": work,
                "object_type": "literary_line",
                "line_count": 1,
                "damaged": False,
                "reading_direction": "R-L",  # "R-L" here means "keep as-stored";
                                              # Tamil is already in natural
                                              # left-to-right reading order
                "motif": thinai if thinai else "unknown",  # thinai = landscape/
                                                            # theme classification,
                                                            # the closest analogue
                                                            # to Indus motif
                "mean_uncertainty": 0.0,
            })

    fieldnames = ["inscription_id", "sign_sequence", "site", "object_type",
                  "line_count", "damaged", "reading_direction", "motif", "mean_uncertainty"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {n_poems} poems, {n_lines_total} total verse lines, "
          f"wrote {len(rows)} usable lines (>= {min_tokens} tokens) to {output_csv}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "sangam_full.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "tamil_real_corpus.csv"
    convert(input_csv, output_csv)
