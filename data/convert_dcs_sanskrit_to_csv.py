"""
convert_dcs_sanskrit_to_csv.py
=================================
Converts the Digital Corpus of Sanskrit's (Hellwig, ongoing since 1999)
Rigveda CoNLL-U files into this toolkit's corpus CSV schema, as a second
real external-language calibration corpus alongside ETCSL (see README's
"Real external-language calibration").

SOURCE: https://github.com/OliverHellwig/sanskrit (dcs/data/conllu/files/
Ṛgveda/, 1,028 files, one per hymn/chapter). Open-sourced by the author
in 2018; standard CoNLL-U format (Universal Dependencies-style), not a
custom SGML/TEI scheme, so parsing is a standard tab-separated-value
read rather than the regex-based entity workaround ETCSL needed.

METHODOLOGICAL CHOICES, matching the reasoning already documented for
ETCSL for consistency across both real-language calibrations:

  UNIT OF COMPARISON: each CoNLL-U sentence (blank-line-delimited block)
  is one "inscription." Rigveda mean sentence length here is 8.4 tokens,
  longer than Indus's 4.4 or ETCSL's 4.46; this is reported plainly
  rather than adjusted, since forcing an artificial length match would
  itself be a methodological choice requiring justification this project
  is not in a position to make (e.g. splitting sentences at clause
  boundaries would need real Sanskrit syntactic judgment this toolkit
  does not have).

  SIGN IDENTITY: column 3 (LEMMA), not column 2 (FORM), for the same
  reason as ETCSL: a lemma collapses inflectional variants the way a
  closed Indus sign catalog would, whereas surface forms would inflate
  vocabulary size with case/tense/person variation that has no clean
  Indus analogue.

  LICENSE: this data is open-sourced on GitHub with no explicit license
  file found at time of integration; treated here as intended for reuse
  given the author's own public announcements encouraging exactly this
  kind of downstream NLP use (see CITATIONS.md), but this should be
  confirmed against the repository's current license terms before any
  redistribution beyond this project's own local, non-commercial use.
"""
import csv
import glob
import sys


def parse_conllu_file(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    sentences = []
    current_tokens = []
    for line in content.split("\n"):
        line = line.rstrip("\r")
        if not line.strip():
            if current_tokens:
                sentences.append(current_tokens)
                current_tokens = []
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        token_id, form, lemma = fields[0], fields[1], fields[2]
        if "-" in token_id or "." in token_id:
            continue  # multi-word token or empty node lines, skip
        if lemma and lemma != "_":
            current_tokens.append(lemma)
    if current_tokens:
        sentences.append(current_tokens)
    return sentences


def convert(input_glob: str, output_csv: str, min_tokens: int = 2):
    rows = []
    n_files = 0
    n_sentences_total = 0
    for path in sorted(glob.glob(input_glob)):
        if path.endswith("_parsed"):
            continue
        n_files += 1
        composition_id = path.split("/")[-1].replace(".conllu", "")
        sentences = parse_conllu_file(path)
        for i, tokens in enumerate(sentences):
            n_sentences_total += 1
            if len(tokens) < min_tokens:
                continue
            rows.append({
                "inscription_id": f"{composition_id}.S{i+1}",
                "sign_sequence": " ".join(tokens),
                "site": composition_id,
                "object_type": "literary_sentence",
                "line_count": 1,
                "damaged": False,
                "reading_direction": "R-L",  # "R-L" here means "keep as-stored";
                                              # Sanskrit is already in natural
                                              # left-to-right reading order
                "motif": "unknown",
                "mean_uncertainty": 0.0,
            })

    fieldnames = ["inscription_id", "sign_sequence", "site", "object_type",
                  "line_count", "damaged", "reading_direction", "motif", "mean_uncertainty"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {n_files} CoNLL-U files, {n_sentences_total} total sentences, "
          f"wrote {len(rows)} usable sentences (>= {min_tokens} tokens) to {output_csv}")


if __name__ == "__main__":
    input_glob = sys.argv[1] if len(sys.argv) > 1 else "Ṛgveda/*.conllu"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "dcs_sanskrit_real_corpus.csv"
    convert(input_glob, output_csv)
