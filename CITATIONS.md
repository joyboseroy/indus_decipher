# Citations

Every data source, paper, and external tool this project relies on,
grouped by what it's used for. If you use this repo, cite the underlying
data sources below in addition to this repo itself: they did the actual
transcription and research work.

## Real data sources (included in this repo)

### yajnadevam/indus-website

Source of `data/indus_website_real_corpus.csv` (2,543 inscriptions, 592
signs, 1,622 with iconographic motif codes).

- Repository: https://github.com/yajnadevam/indus-website
- What we use: the `SEAL`, `INSCRIPTION`, `GLYPHSEQUENCE`, and
  `ICONOGRAPHY` tables from `population-script.sql`, parsed by
  `data/convert_indus_website_sql_to_csv.py`.
- License: check the repository directly before redistributing beyond
  this project's use; no explicit data license was found at the time of
  integration.
- Caveat: this repository is maintained by the same researcher (going by
  the GitHub handle "yajnadevam") behind a "cryptanalytic decipherment of
  the Indus script" claim (dated November 2024) proposing the script
  encodes post-Vedic Sanskrit. That specific decipherment claim has been
  publicly and substantively critiqued, including by the author's own
  later acknowledgment of errors in the original paper and procedures.
  This project does not use, endorse, or rely on that decipherment claim
  or any sign-to-meaning mapping from it. What this project uses is raw
  transcription data (which glyph appears where, on which seal, at which
  site), which is ordinary epigraphic fieldwork independent of the
  disputed interpretive claim. Users of this repo should form their own
  view of that dispute rather than take this project's use of the
  underlying data as an endorsement either way.
- Corroborating detail: this corpus's scale (2,543 raw inscriptions
  before deduplication, 700 total glyph codes, 52 archaeological sites)
  matches the "ICIT/Yajnadevam digitization" described in the arXiv paper
  cited below, and this toolkit's own entropy analysis on it (3.26 bits,
  after correcting a reading-direction error) lands close to that paper's
  reported 3.23 bits.

### mayig/indus-valley-script-corpus

Source of `data/cisi_real_corpus.csv` (179 inscriptions, 142 signs, all
with a motif derived from the seal description).

- Repository: https://github.com/mayig/indus-valley-script-corpus
- What we use: the `corpus/*/*.json` files, a hand-transcription of a
  subset of Parpola's Corpus of Indus Seals and Inscriptions (CISI),
  parsed by `data/convert_cisi_to_csv.py`.
- License: MIT (per the repository's LICENSE file).
- Acknowledgment (from the source repository's own README): the author
  thanks Dr. Asko Parpola for the underlying CISI corpus, and
  acknowledges Dr. Andreas Fuls and Bryan K. Wells for related
  contributions to Indus script theory and pedagogy.
- Note: this is explicitly a work-in-progress digitization (179
  inscriptions transcribed as of integration, covering roughly the M-1
  through M-199 range from Mohenjo-daro), not a complete corpus.

## Foundational academic literature (methodology this toolkit implements)

- Mahadevan, I. (1977). *The Indus Script: Texts, Concordance and
  Tables.* Archaeological Survey of India. The original M77 concordance
  and 417-sign catalog; not directly used as data here (not publicly
  downloadable at time of writing) but the methodological reference point
  for sign numbering and corpus filtering (EBUDS) throughout the field.
- Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., and
  Mahadevan, I. (2009). "Entropic Evidence for Linguistic Structure in
  the Indus Script." *Science*, 324(5931), 1165.
- Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., and
  Mahadevan, I. (2009). "A Markov Model of the Indus Script." *PNAS*,
  106(33), 13685-13690. Source of the n-gram Markov modeling approach
  implemented in `analysis/ngram.py`.
- Yadav, N., Joglekar, H., Rao, R. P. N., Vahia, M. N., Adhikari, R., and
  Mahadevan, I. (2010). "Statistical Analysis of the Indus Script Using
  n-Grams." *PLOS ONE*, 5(3), e9506.
  https://doi.org/10.1371/journal.pone.0009506
  Source of the Zipf-Mandelbrot fitting, positional asymmetry analysis,
  log-likelihood significant-bigram testing, cross-validated perplexity,
  and the restoration-accuracy methodology (the top-90%-cumulative-mass
  criterion implemented in `analysis/ngram.py`'s
  `restoration_accuracy_top90mass()`) that this toolkit reproduces. Open
  access under CC BY. No downloadable dataset was found attached to this
  paper; its corresponding authors, listed in the paper, are Nisha Yadav
  (`y_nisha@tifr.res.in`) and Ronojoy Adhikari (`rjoy@imsc.res.in`).
- Rao, R. P. N., Yadav, N., Vahia, M. N., Joglekar, H., Adhikari, R., and
  Mahadevan, I. (2010). "Entropy, the Indus Script, and Language: A Reply
  to R. Sproat." *Computational Linguistics*, 36(4), 795-805. Response to
  the Farmer-Sproat-Witzel non-linguistic-symbol hypothesis; the
  conceptual basis for this toolkit's `analysis/entropy.py` comparisons
  against random and rigid-fixed controls.
- Sinha, S., Pan, R. K., Yadav, N., Vahia, M., and Mahadevan, I. (2009).
  "Network Analysis of a Corpus of Undeciphered Indus Civilization
  Inscriptions Indicates Syntactic Organization." Describes the Wells
  W09IMSc / WUCS datasets as an alternative corpus to M77; not directly
  used here but a reference for corpus provenance terminology used in
  this project's documentation.
- Farmer, S., Sproat, R., and Witzel, M. (2004). "The Collapse of the
  Indus-Script Thesis: The Myth of a Literate Harappan Civilization."
  *Electronic Journal of Vedic Studies*, 11(2). The non-linguistic-symbol
  hypothesis that motivated the entropy-based falsification approach this
  toolkit implements.
- Anonymous or pseudonymous author (2026). "How Non-Linguistic Is the
  Indus Sign System? A Synthetic-Baseline Scorecard." arXiv:2604.17828.
  https://arxiv.org/abs/2604.17828
  Source of the "ICIT/Yajnadevam digitization" description (1,916
  deduplicated inscriptions, 584 unique signs, 11,110 tokens, 52 sites)
  that let us corroborate `data/indus_website_real_corpus.csv`'s
  provenance, and of the specific synthetic-baseline methodology
  (heraldic and administrative null-model generators) that inspired this
  project's `data/synthetic_civilizations.py` and
  `analysis/falsification.py`. This paper's own data-availability
  statement says its data is available from the corresponding author
  upon request; we did not obtain it directly and instead independently
  found and verified the underlying corpus via yajnadevam/indus-website.

## Other real-world sources referenced but not integrated

- Palaniappan, S., and Adhikari, R. (2017). "Deep Learning the Indus
  Script." arXiv:1702.00523. Source of the vision/OCR pipeline whose
  trained weights live at https://github.com/tpsatish95/indus-script-ocr.
  Not used in this toolkit, which works on transcribed sign sequences,
  not seal photographs.
- Wells, B. K. (2006). *Epigraphic Approaches to Indus Writing.* Harvard
  University. Source of the Interactive Corpus of Indus Texts (ICIT)
  described in this project's README as an unintegrated real-data lead.

## Context on disputed or independently unverified claims

These are discussed in this project's development history and README for
transparency, but nothing from them is used as data or methodology here:

- Yajnadevam (2024). "A Cryptanalytic Decipherment of the Indus Script."
  Claims the script encodes post-Vedic Sanskrit. Publicly and
  substantively critiqued; the author later acknowledged errors in the
  original paper's procedures. See public discussion threads for detail;
  not cited further here because this project takes no position on the
  claim beyond noting its disputed status.
- Pierson, T. K. (2026). "A Computational Decipherment Hypothesis for the
  Indus Script: 185 Proto-Dravidian Readings Validated Across Two
  Independent Corpora." Zenodo. DOI: 10.5281/zenodo.20414696. A
  self-published, not-yet-peer-reviewed preprint (repository:
  https://github.com/BitConcepts/glossa-lab) proposing a Proto-Dravidian
  decipherment. The repository is unusually transparent about its own
  audit process, including retracted claims, which is commendable, but
  the claim itself has not been independently peer-reviewed at time of
  writing. Not used as data or methodology here.

## Software

- NumPy, SciPy, pandas, Matplotlib: standard scientific Python stack,
  used throughout `analysis/` and `main.py`.
- No deep learning framework (PyTorch, TensorFlow) is used;
  `models/transformer_mlm.py` is a from-scratch NumPy implementation, by
  design, to keep this repo dependency-light and fully inspectable.
