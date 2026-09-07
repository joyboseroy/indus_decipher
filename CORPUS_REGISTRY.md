# Corpus Registry

Every corpus this project uses, in one place, with numbers pulled
directly from the actual files (not copied from memory or from an
earlier write-up) at the time this registry was built. If a number here
ever disagrees with a number in `README.md`, treat that as a bug to
investigate, not a stylistic choice -- open an issue or check
`data/loader.py`'s `Corpus.vocab()` and `.filter()` methods, since a
mismatch usually means one of the two was computed against a stale file.

A CSV version of the same data is at `data/corpus_registry.csv`, for
anything that wants to load this programmatically instead of parsing
markdown.

## Real Indus-script corpora

| Corpus | N (full) | N (non-damaged) | Signs (full) | Signs (non-damaged) | Unit | Site metadata | Motif metadata | Direction | Access | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| M77 | 3,573 | 3,573 | 418 | 418 | line | none | none | as-stored (verified) | authenticated, not public bulk | verified |
| indus_website | 2,543 | 2,543 | 592 | 592 | inscription | 52 sites | 1,622/2,543 rows | as-stored, corrected from a real bug | public (GitHub) | verified |
| CISI, primary | 179 | 104 | 182 | 142 | inscription | 1 site (Mohenjo-daro only) | 179/179 rows | reversed from source, corrected | public (GitHub, MIT) | verified |
| CISI, hierarchical | 179 | 104 | 234 | 185 | inscription | 1 site | 179/179 rows | same as primary | public (GitHub, MIT) | verified |
| CISI, allograph | 179 | 104 | 337 | 230 | inscription | 1 site | 179/179 rows | same as primary | public (GitHub, MIT) | verified |
| WUCS | 1,821 | n/a | 593 | n/a | sequence | not accessed (published stats only) | not accessed | not tested | published statistics only, no raw data access | external (statistics only) |
| EBUDS (subset of M77 in the classic literature) | ~1,548 (published) | n/a | ~377 (published) | n/a | text | n/a | n/a | n/a | not separately reconstructed -- see note below | not attempted |

**Damage/duplicate/uncertainty treatment, since the table above doesn't
have room for it:**

- **M77:** 343 of 3,916 raw records are placeholder/zero-position
  records (`posnum=0`), excluded before the "N" above. No separate
  damage flag; 504 sign occurrences are "starred" (uncertain reading),
  captured as a `mean_uncertainty` percentage per line, not a boolean.
  Sign numbers were found to have inconsistent zero-padding in the
  source (`"1"` vs `"001"`) and are normalized through `int()` --
  documented in detail in `data/convert_m77_indusscript_to_csv.py`,
  since this specific normalization step is exactly the kind of thing
  that silently inflates a vocabulary count if skipped.
- **indus_website:** `damaged` is a real per-inscription boolean from
  the source; no allograph or uncertainty data available.
- **CISI (all three granularities):** `damaged` is derived from a
  per-grapheme damage code (true if ANY grapheme has a nonzero code);
  `mean_uncertainty` is a real 0-100 subjective score from the original
  annotator, distinct from damage. The three granularities differ only
  in whether/how allograph variants of a sign are split; see README's
  "Allograph granularity" section for why this is a real, resolved
  finding, not an arbitrary choice.
- **WUCS:** this project never received the raw WUCS sequences; only
  Sinha/Izhar/Pan/Wells's own published aggregate statistics
  (reciprocity, connectivity, beginner/ender counts) were used, for
  comparison against this project's own corpora computed the same way.
  Everything in the "N/Signs" columns for WUCS is what THEY published
  about their own data, not something independently verified here.
- **EBUDS:** the specific 1,548-text filtered subset used in Rao et al.
  2009 and Yadav et al. 2010 was never separately obtained or
  reconstructed. This project's M77 corpus (3,573 lines / 2,906 texts)
  is the fuller, unfiltered M77/IDF-80 corpus, not EBUDS specifically;
  the two are related but not identical, and this registry does not
  claim EBUDS-level replication, only M77-level.

## Real external-language calibration corpora (not Indus-script; used for comparison)

| Corpus | N (lines/sentences) | Vocabulary | Unit | Vocabulary type | Access | Status |
|---|---|---|---|---|---|---|
| ETCSL (Sumerian) | 33,338 | 4,168 | line | real dictionary lemmas | not public-bulk (login-free but no single-command mirror at time of use); see CITATIONS.md | verified |
| DCS (Sanskrit, Rigveda) | 21,231 | 8,764 | sentence | real dictionary lemmas (CoNLL-U) | public (GitHub) | verified |
| Sangam (Old Tamil) | 33,711 | 35,301 | line | raw orthographic words, NOT lemmatized | public (Hugging Face) | verified, but flagged unreliable (vocabulary exceeds line count -- see README's "Old Tamil" section) |

**Critical asymmetry, worth repeating here since it's easy to miss when
skimming a table:** ETCSL and DCS give real lemmas (a dictionary
citation form, collapsing inflectional variants the way a closed Indus
sign catalog would). Sangam Tamil does not -- no lemmatized digital
Sangam corpus was found, so raw whitespace-separated words are used
instead, and Tamil's agglutinative morphology means most such "words"
are functionally unique inflected forms. This is why Tamil's results
are reported as an honestly-flagged negative/inconclusive finding, not
folded into the calibration table on equal footing with the other two.
See "Do not compare these numbers" below, and README's "Old Tamil:
located, tried..." section for the full account.

## Synthetic corpora (ground-truth mechanisms, not real data)

| Generator | Mechanism | Used for |
|---|---|---|
| `civ_a_language_like` | root + conditioned suffix(es), simulated morphology | Falsification harness reference class |
| `civ_b_administrative_code` | fixed positional slots | Falsification harness reference class |
| `civ_c_mixed` | blend of the above two | Falsification harness reference class |
| `civ_d_markov_no_morphology` | hand-designed order-2 Markov, zero morphology | Tests whether order-3 gain requires morphology (it doesn't, on its own) |
| `civ_e_hierarchical_administrative` | category -> subcategory -> item nesting | Tests whether order-3 gain requires anything linguistic (it doesn't) |
| `bigram_markov_null` | real order-1 transitions from a target corpus, no more | Locates where in the dependency order a real corpus's signal lives |
| `trigram_markov_null` | real order-1+2 transitions (backoff second-order Markov), no more | Same, one order higher |
| `generate_matched_null_corpus` (adversarial null) | independent draws matching length/marginal frequency only | Tests whether ANY sequential dependency exists at all |
| `motif_stratified_null` / `site_stratified_null` | adversarial null computed per stratum instead of pooled | Tests whether archaeological composition alone explains a signal |

## How to regenerate any real corpus locally

```bash
python3 data/convert_indus_website_sql_to_csv.py <sql_path> <out.csv>
python3 data/convert_cisi_to_csv.py <json_glob> <out.csv> [primary|hierarchical|allograph]
python3 data/convert_m77_indusscript_to_csv.py <json_path> <out.csv>
python3 data/convert_etcsl_to_csv.py <xml_glob> <out.csv> [lemma|form]
python3 data/convert_dcs_sanskrit_to_csv.py <conllu_glob> <out.csv>
python3 data/convert_tamil_to_csv.py <csv_path> <out.csv>
```

None of the raw source files for M77, ETCSL, DCS, or Tamil are
redistributed in this repository; only `indus_website` and CISI ship
their derived CSVs directly (see `CITATIONS.md` for the license
reasoning behind that split).
