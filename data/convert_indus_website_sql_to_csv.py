"""
convert_indus_website_sql_to_csv.py
=====================================
Parses the MySQL population script from yajnadevam/indus-website
(https://github.com/yajnadevam/indus-website) into the CSV schema used by
this toolkit's data/loader.py.

This SQL dump's scale (2,543 seals, 11,280 glyph occurrences, 700 distinct
glyphs, 52 sites) matches the "ICIT/Yajnadevam digitization" cited in the
2026 arXiv paper "How Non-Linguistic Is the Indus Sign System?" (1,916
DEDUPLICATED inscriptions, 584 unique signs, 11,110 tokens, 52 sites) --
the raw counts here are pre-deduplication, which is consistent. This
appears to be the actual underlying digitization behind that published
analysis, not a synthetic or invented dataset.

CAVEAT: this is Yajnadevam's own digitization/database, built to support
his own (independently disputed -- see public critiques of his "Sanskrit
decipherment" claim) analysis. The SIGN SEQUENCES and metadata (site,
completeness, direction) are raw transcription data and are usable
regardless of what conclusions any particular researcher drew from them --
but treat GLYPHID-to-meaning mappings or any decipherment claims bundled
elsewhere in that project with real skepticism. This script extracts only
the structural data: which glyph IDs appear, in what order, on which seal,
at which site.
"""
import csv
import io
import re
import sys


def _parse_tuples(block: str, n_fields: int) -> list[list[str]]:
    """Extract each (...) tuple's fields, correctly handling quoted strings
    that may contain commas, using csv's parser on each tuple's inner text."""
    tuples = []
    for match in re.finditer(r'\(([^()]*)\)', block):
        inner = match.group(1)
        try:
            row = next(csv.reader(io.StringIO(inner), skipinitialspace=True))
        except StopIteration:
            continue
        row = [f.strip().strip('"') for f in row]
        if len(row) == n_fields:
            tuples.append(row)
    return tuples


def convert(sql_path: str, output_csv: str):
    with open(sql_path, encoding="utf-8") as f:
        content = f.read()

    def block_after(marker: str) -> str:
        return content.split(marker)[1].split("INSERT INTO")[0]

    seal_rows = _parse_tuples(
        block_after("INSERT INTO SEAL ("), n_fields=8)
    # SEAL: SEALID, SITEID, MATERIAL, CISI, MUSEUM, WIDTH, HEIGHT, THICKNESS
    site_by_seal = {r[0]: r[1] for r in seal_rows}

    insc_rows = _parse_tuples(
        block_after("INSERT INTO INSCRIPTION ("), n_fields=3)
    # INSCRIPTION: SEALID, ISCOMPLETE, DIRECTION
    insc_by_seal = {r[0]: (r[1], r[2]) for r in insc_rows}

    # ICONOGRAPHY: SEALID, DESCRIPTION -- real motif/field-symbol codes
    # (e.g. "Bull1:W", "Gaur", "Elep"), present for ~64% of seals. Used for
    # stronger seal-twin corroboration in analysis/minimal_pairs.py.
    iconography_rows = _parse_tuples(
        block_after("INSERT INTO ICONOGRAPHY("), n_fields=2)
    motif_by_seal = {r[0]: r[1] for r in iconography_rows}

    gs_rows = _parse_tuples(
        block_after("INSERT INTO GLYPHSEQUENCE ("), n_fields=3)
    # GLYPHSEQUENCE: SEALID, GLYPHID, IDX
    glyphs_by_seal: dict[str, list[tuple[int, str]]] = {}
    for seal_id, glyph_id, idx in gs_rows:
        glyphs_by_seal.setdefault(seal_id, []).append((int(idx), f"G{glyph_id}"))

    fieldnames = ["inscription_id", "sign_sequence", "site", "object_type",
                  "line_count", "damaged", "reading_direction", "motif"]
    rows_out = []
    for seal_id, glyph_list in glyphs_by_seal.items():
        glyph_list.sort(key=lambda t: t[0])
        signs = [g for _, g in glyph_list]
        if not signs:
            continue
        is_complete, direction = insc_by_seal.get(seal_id, ("Y", "R/L"))
        # CORRECTED based on analysis/direction_test.py evidence (both real
        # corpora tested showed the published Rao/Yadav fingerprint -- final
        # position more constrained than initial -- only when reversed
        # relative to the original mapping below). This mapping is
        # deliberately the flip of the naive DIRECTION-field reading:
        #   original (wrong):  "L/R" -> "L-R" (reversed by loader), "R/L" -> "R-L" (kept as-is)
        #   corrected:         "L/R" -> "R-L" (kept as-is),         "R/L" -> "L-R" (reversed by loader)
        # This is still a heuristic correction based on an aggregate
        # statistical fingerprint, not a verified ground truth for this
        # specific database's DIRECTION field semantics -- rerun
        # analysis.direction_test on any new export to confirm.
        reading_direction = "R-L" if direction == "L/R" else "L-R"
        rows_out.append({
            "inscription_id": f"S{seal_id}",
            "sign_sequence": " ".join(signs),
            "site": site_by_seal.get(seal_id, "unknown"),
            "object_type": "unknown",
            "line_count": 1,
            "damaged": (is_complete == "N"),
            "reading_direction": reading_direction,
            "motif": motif_by_seal.get(seal_id, "unknown"),
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Parsed {len(seal_rows)} SEAL rows, {len(insc_rows)} INSCRIPTION rows, "
          f"{len(iconography_rows)} ICONOGRAPHY rows, "
          f"{len(gs_rows)} GLYPHSEQUENCE rows -> wrote {len(rows_out)} inscriptions "
          f"to {output_csv}")


if __name__ == "__main__":
    sql_path = sys.argv[1] if len(sys.argv) > 1 else "indus-website-main/population-script.sql"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "indus_website_real_corpus.csv"
    convert(sql_path, output_csv)
