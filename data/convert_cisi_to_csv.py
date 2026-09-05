"""
convert_cisi_to_csv.py
========================
Converts the mayig/indus-valley-script-corpus JSON format (a real,
hand-transcribed digitization of Parpola's Corpus of Indus Seals and
Inscriptions, CISI) into the CSV schema expected by data/loader.py.

Source: https://github.com/mayig/indus-valley-script-corpus
        (MIT licensed, credits Parpola/Wells/Fuls -- see its README)

Each source JSON file is a list of "sides" of one artifact:
  {"id": "M-1A", "description": "unicorn I seal",
   "graphemes": [{"id": "P121", "features": [damage, line, uncertainty, ...]}]}

We treat each side as one inscription. features[0]=damage, features[1]=line,
features[2]=uncertainty are the documented "default features" that precede
any allograph-specific ones.
"""
import csv
import glob
import json
import sys

SITE_PREFIX_MAP = {"M": "Mohenjo-daro", "H": "Harappa", "L": "Lothal", "K": "Kalibangan"}


def object_type_from_description(desc: str) -> str:
    desc = (desc or "").lower()
    if "seal" in desc:
        return "seal"
    if "tablet" in desc:
        return "tablet"
    if "sealing" in desc:
        return "sealing"
    if "pot" in desc or "jar" in desc:
        return "pottery"
    return "unknown"


def motif_from_description(desc: str) -> str:
    """This corpus's descriptions follow a "<motif> <variant> seal" pattern
    (e.g. "unicorn IV seal") -- extract the motif + variant as a coarse
    field-symbol classification, analogous to the indus-website corpus's
    ICONOGRAPHY codes (e.g. "Bull1:W"), for seal-twin corroboration."""
    desc = (desc or "").strip()
    if not desc:
        return "unknown"
    parts = desc.split()
    if len(parts) >= 2 and parts[-1].lower() in ("seal", "tablet", "sealing"):
        parts = parts[:-1]
    return "_".join(parts) if parts else "unknown"


def convert(input_glob: str, output_csv: str):
    rows = []
    for path in sorted(glob.glob(input_glob)):
        with open(path, encoding="utf-8") as f:
            sides = json.load(f)
        for side in sides:
            ins_id = side["id"]
            graphemes = side.get("graphemes", [])
            signs = [g["id"] for g in graphemes]
            if not signs:
                continue
            damaged = any(
                len(g.get("features", [])) > 0 and g["features"][0] not in (0, None)
                for g in graphemes
            )
            lines = [g["features"][1] for g in graphemes if len(g.get("features", [])) > 1]
            line_count = max(lines) if lines else 1
            site = SITE_PREFIX_MAP.get(ins_id[0], "unknown")
            rows.append({
                "inscription_id": ins_id,
                "sign_sequence": " ".join(signs),
                "site": site,
                "object_type": object_type_from_description(side.get("description", "")),
                "line_count": line_count,
                "damaged": damaged,
                # CORRECTED based on analysis/direction_test.py evidence: the
                # published Rao/Yadav fingerprint (final position more
                # constrained than initial) only appeared when this corpus's
                # sign order was reversed relative to the original hardcoded
                # "R-L" (which meant "kept as-stored" -- see data/loader.py's
                # normalized_signs()). "L-R" makes the loader reverse it.
                # Still a heuristic correction, not verified ground truth for
                # how mayig's corpus actually orders graphemes -- rerun
                # analysis.direction_test on any new export to confirm.
                "reading_direction": "L-R",
                "motif": motif_from_description(side.get("description", "")),
            })

    fieldnames = ["inscription_id", "sign_sequence", "site", "object_type",
                  "line_count", "damaged", "reading_direction", "motif"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} inscriptions to {output_csv}")


if __name__ == "__main__":
    input_glob = sys.argv[1] if len(sys.argv) > 1 else "indus-valley-script-corpus-main/corpus/*/*.json"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "cisi_real_corpus.csv"
    convert(input_glob, output_csv)
