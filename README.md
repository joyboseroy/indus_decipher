# Indus Script Computational Analysis Toolkit

A working pipeline for computational, statistics-first analysis of the
undeciphered Indus Valley script. It implements the established
methodology from Rao, Yadav, Vahia, Adhikari and Mahadevan's published
work (unigram/Zipf-Mandelbrot statistics, positional asymmetry, n-gram
Markov modeling, conditional entropy against non-linguistic controls),
plus a few extensions built for this project: a synthetic-civilization
falsification harness, seal-twin minimal-pair mining with iconographic
corroboration, a reading-direction diagnostic, and a small from-scratch
transformer for masked-sign prediction.

## What this project is, and is not

This is a toolkit for running known and new statistical tests on Indus
script corpora and reporting the numbers honestly, including when they
disagree with each other or with the literature. It is not a
decipherment, and it does not claim to identify the underlying language.
The field has a long history of confident claims (Sanskrit, various
Dravidian readings, Sumerian derivation, and others) that did not hold up
under independent scrutiny. This repo is built to avoid adding to that
pile: every number here comes with the method that produced it, and every
finding that could be read as "evidence for X" is stated with its actual
epistemic weight, not its most exciting-sounding interpretation.

If you're looking for a decipherment claim, this isn't one. If you're
looking for a codebase to run real, checkable statistical tests on real
digitized corpora and see where the evidence currently stands, that's
what this is.

## How this was built

This codebase was developed iteratively with help of Claude (Anthropic), under human
direction and review at every step, including the decision of which
statistical tests to run, which data sources to trust, and how to word
every caveat in this document. Several of the more interesting findings
below (the reading-direction correction, the two real corpora landing on
different sides of the falsification harness) came from actually running
the code on real data and following up on results that looked off,
rather than from the design phase. That back-and-forth is part of why
this project is reasonably confident in its own honesty notes: they were
earned by being wrong first and checking.

## Status

Active exploration, not a finished study. Two real corpora are integrated
(see "Real data" below) and the pipeline runs cleanly on both, but the
question of what their disagreement means is still open (see the
"Known open questions" section).

## Quick start

```bash
pip install numpy scipy pandas matplotlib
python3 main.py                      # runs on the synthetic demo corpus
python3 main.py --csv mydata.csv     # runs on your own corpus
python3 main.py --extended           # adds minimal-pair mining, the
                                      # falsification harness, and the
                                      # reading-direction diagnostic
```

Output: a console report, `outputs/report.json`, and three plots
(Zipf fit, entropy-vs-length, transformer training loss).

Real data ships with the repo, so this also works immediately:
```bash
python3 main.py --csv data/indus_website_real_corpus.csv --extended
python3 main.py --csv data/cisi_real_corpus.csv --extended
```

## Project layout

```
data/
  loader.py                    corpus schema (including an optional motif
                                field for iconography) and CSV/JSON loaders
  synthetic_corpus.py          synthetic test-fixture generator (NOT real data)
  synthetic_civilizations.py   three generators with KNOWN ground-truth
                                structure, used by the falsification harness
  convert_indus_website_sql_to_csv.py   real-data converter (see below)
  convert_cisi_to_csv.py                real-data converter (see below)
  indus_website_real_corpus.csv         real data (2,543 inscriptions)
  cisi_real_corpus.csv                  real data (179 inscriptions)
analysis/
  positional.py         unigram counts, Zipf-Mandelbrot fit, initial/final
                         positional asymmetry, sign-doubling rate
  ngram.py               bigram/trigram transition matrices, log-likelihood
                         significant pairs, cross-validated perplexity, two
                         restoration-accuracy metrics (see below)
  entropy.py             conditional entropy H(X|Y) against synthetic controls
  direction_test.py      reading-direction diagnostic (see below)
  minimal_pairs.py        seal-twin mining with three corroboration tiers,
                         including real iconographic motif matching
  falsification.py        feature-vector classification against three known
                         synthetic generative systems
models/
  transformer_mlm.py     small NumPy, zero-dependency masked-language
                         transformer for bidirectional sign prediction
main.py                  end-to-end pipeline / report generator
CITATIONS.md             every data source and paper this project relies on
```

Every analysis module takes a plain `list[list[str]]` of sign sequences,
so you can call them directly from a notebook without touching `main.py`.

## Real data (included)

Two real, non-synthetic corpora are included, both converted from public
GitHub repositories. Full citation and license detail is in
`CITATIONS.md`; the summary:

- **`data/indus_website_real_corpus.csv`** (2,543 inscriptions, 592
  signs, 1,622 with real iconographic motif codes) parsed from the
  MySQL dump in
  [yajnadevam/indus-website](https://github.com/yajnadevam/indus-website).
  Its scale (2,543 raw inscriptions, 700 total glyph codes, 52 sites)
  matches the "ICIT/Yajnadevam digitization" cited in a 2026 arXiv paper
  on non-linguistic sign-system baselines, and running this toolkit's
  entropy analysis on it, after correcting a reading-direction bug (see
  below), gives 3.26 bits, close to that paper's reported 3.23 bits. That
  agreement is a reasonable sanity check that this is genuinely the
  underlying data.
  **Important caveat:** this repository is maintained by the same
  researcher behind an independently disputed "cryptanalytic Sanskrit
  decipherment" claim (public critiques of that specific claim exist and
  are easy to find). The raw glyph sequences and site metadata used here
  are ordinary transcription data and stand on their own regardless of
  that dispute, but any sign-to-meaning mapping from that researcher's
  other work, not used anywhere in this toolkit, should be treated with
  real skepticism.
- **`data/cisi_real_corpus.csv`** (179 inscriptions, 142 signs, all with
  an iconographic motif derived from the seal description) parsed from
  [mayig/indus-valley-script-corpus](https://github.com/mayig/indus-valley-script-corpus),
  a smaller, transparently documented, in-progress hand-transcription of
  Parpola's CISI corpus. Every inscription in this set happens to be a
  unicorn-motif seal from Mohenjo-daro; see "Known open questions" for
  why that matters.

Regenerate either CSV from a local copy of its source repo with
`python3 data/convert_indus_website_sql_to_csv.py <sql_path> <out.csv>` or
`python3 data/convert_cisi_to_csv.py <json_glob> <out.csv>`.

## Other real-data leads (not yet integrated)

- **Mahadevan's M77 concordance / EBUDS** (2,906 texts, 417 signs), the
  corpus behind Yadav et al. 2010 and Rao et al. 2009. No public download
  found; the papers' corresponding authors are the most direct route
  (see `CITATIONS.md` for contact info found in the published papers).
- **ICIT** (Wells and Fuls, roughly 700 signs, 4,500+ objects),
  historically access-by-request via Andreas Fuls at TU Berlin.
- **tpsatish95/indus-script-ocr**, real CNN weights from Palaniappan and
  Adhikari's deep-learning seal-segmentation pipeline. This is the
  vision/OCR layer, not sign-sequence data, and would only matter if this
  project extends into image processing.

Once you have any other real export, converting it into the schema in
`data/loader.py` is the only integration work needed. Nothing else in
the codebase changes.

## The reading-direction diagnostic

`analysis/direction_test.py` checks a corpus both as-stored and reversed
against a specific published fingerprint: Rao and Yadav report that in
the correctly-oriented M77 corpus, the final sign position is more
strongly constrained (lower entropy) than the initial position. This
diagnostic computes that gap in both orientations and reports which one
matches the fingerprint more strongly.

This mattered here in practice. Both real corpora were originally
converted with a reading-direction assumption that turned out to be
backwards; the diagnostic caught it because neither corpus showed the
expected final-more-constrained pattern until reversed. After the fix,
both corpora show it clearly, and the large corpus's most frequent sign
(G740) went from wrongly initial-heavy to correctly final-heavy, the same
shape as M77's well-known "jar sign." The current converters in this repo
already have the fix applied; the diagnostic is included so any future
data source can be checked the same way rather than assumed correct.

This is a heuristic, not proof. A corpus could legitimately have
different conventions than M77, and the fingerprint itself, established
on one corpus in one sign scheme, is not guaranteed to generalize.
Treat a mismatch as a prompt to investigate the data's own direction
metadata or the conversion step, not as automatic confirmation of a bug.

## The falsification harness (`--extended`)

Before trusting any statistical pattern found in a real corpus as
evidence about language versus non-language, this repo first checks
whether its own pipeline can tell three known synthetic generators apart:

- `civ_a_language_like`: agglutinative morphology (root plus conditioned
  suffixes plus an optional case marker), a simplified stand-in for a
  real grammar
- `civ_b_administrative_code`: fixed independent slots
  (category/location/authority), the Farmer-Sproat-Witzel "heraldic
  emblem" non-linguistic null model
- `civ_c_mixed`: a coin-flip blend of the two plus free-floating numerals

`python3 -m analysis.falsification` runs a leave-one-out classification
test across several instances of each and reports accuracy, currently
100% on the seven features used (see the module docstring for the exact
list). `main.py --extended` additionally classifies whichever corpus you
loaded against these three reference families and reports the nearest
match with distances, not a verdict: "resembles the language-like
generator most closely" is a statement about a nearest-centroid distance
to three specific synthetic corpora, not a claim about the real script.

**Open finding:** after the reading-direction fix, the large real corpus
now classifies as `civ_a_language_like`, while the small CISI corpus
still classifies as `civ_c_mixed`, and only by a narrow margin. See
"Known open questions" below.

## Seal-twin minimal-pair mining, with iconographic corroboration

`analysis/minimal_pairs.py` looks for inscriptions identical except at
one sign position. Whatever occupies that position in each, without
knowing what either sign means, is at minimum substitutable in that slot.
Collecting these substitutions builds a graph whose connected components
are candidate paradigm classes.

Three corroboration tiers are tracked, in increasing confidence:
1. Sequence-only: same length, one differing position.
2. Context-corroborated: tier 1, plus matching site and object type.
3. Motif-corroborated: tier 2, plus matching iconographic field symbol
   (the real motif codes from the indus-website corpus's ICONOGRAPHY
   table, such as "Bull1:W" or "Gaur," or the CISI corpus's
   description-derived motif such as "unicorn_IV").

On the large real corpus, restricting to motif-corroborated pairs (144 of
them) does not just narrow the evidence, it resolves more structure: 18
distinct paradigm classes emerge, versus 7 from the unfiltered set. The
loose tier was letting spurious cross-links merge genuinely separate
classes into a few large blobs; motif corroboration removes exactly those
noise edges. The CISI corpus is too small (3 total minimal pairs) for
this tier to show anything.

## Known open questions

**Why do the two real corpora disagree on the falsification harness?**
After the direction fix, the 2,543-inscription corpus reads as
language-like; the 104-inscription CISI corpus (after filtering damaged
entries) reads as mixed, narrowly. Candidate explanations, not yet
distinguished: this could be ordinary sample-size noise given how much
smaller the CISI set is; it could reflect that CISI's slice is
exclusively unicorn-motif Mohenjo-daro seals, a genuinely narrower and
more homogeneous subset than the full corpus, rather than a random
sample; or it could reflect a real difference between how M77-derived
transcription conventions and this SQL dump's ICIT-derived conventions
segment or encode signs. This is under active investigation.

## Known limitations and other honesty notes

- **The transformer is small and NumPy-only on purpose.** It is a full,
  from-scratch backpropagation implementation (attention, feed-forward,
  embeddings), so it has zero heavy dependencies and is fully
  inspectable, but it is toy-scale (one layer, d_model 48). On the
  synthetic corpus it narrowly edges out the bigram baseline for
  masked-sign accuracy. A real corpus with richer long-range structure,
  or simply more data, is where a transformer's bidirectional context
  should show a bigger advantage over left-only n-grams. If you have more
  compute, swap in a proper PyTorch or JAX model; the `Vocab`,
  `train_mlm`, and `evaluate_mlm_accuracy` interface is designed as a
  drop-in replacement target.
- **The synthetic demo corpus's bigram structure is literally generated
  by a Markov process**, so a bigram model is close to the "correct"
  model for it by construction. Restoration accuracy and perplexity on
  the synthetic corpus are a sanity check that the code works, not a
  result comparable to the real script.
- **Layernorm's backward pass is approximated as identity** in the
  transformer, for readability. This does not break training, verified
  empirically, but a from-scratch reimplementation intended for real
  research use should implement the exact gradient.
- **The falsification harness's 100% self-test accuracy is a floor, not
  a ceiling of confidence.** The three synthetic civilizations were
  deliberately built to be structurally quite different from each other.
  Passing this test shows the pipeline can discriminate when the
  underlying systems really are different in these specific ways. It
  does not mean the same features would cleanly separate two more
  similar synthetic systems, and a real-data classification result is
  one data point requiring corroboration, not a verdict.
- **There are two restoration-accuracy metrics in `analysis/ngram.py`,
  and they are not interchangeable.** `restoration_accuracy()` is strict
  top-1 accuracy. `restoration_accuracy_top90mass()` matches Yadav et
  al.'s actual published methodology (a restoration counts as correct if
  the true sign falls within the smallest set of candidates needed to
  reach 90% of the model's predicted probability mass). On the real data
  here, the top-90-mass metric comes back around 90 to 98%, far above the
  published 75%, but with a mean candidate-set size of 60 to 85% of the
  entire vocabulary. That means the metric is barely discriminating
  anything on this corpus with this toolkit's simple smoothing; it is not
  evidence this toolkit outperforms the literature. The published 75%
  figure used Witten-Bell smoothing, which concentrates probability mass
  far more sharply than the add-alpha smoothing implemented here. Read
  both the accuracy and the candidate-set size together, never the
  accuracy alone.
- **Minimal-pair mining is O(n squared) within each length bucket.**
  Fine for a few thousand inscriptions, the realistic scale here; would
  need bucketing or hashing to scale further. It also compares raw sign
  identity only. It has no notion of "these two signs are probably
  damage versus clean variants of the same glyph," which would need to
  be fed in from allograph or damage annotations.
- **Conditional-entropy comparisons use synthetic random and rigid
  controls**, not the real Sumerian, Tamil, Sanskrit, or DNA control
  corpora used in the literature. Supply your own tokenized reference
  corpora via `analysis.entropy.external_control_entropy()` for a
  literature-comparable result.
- **The reading-direction diagnostic is a heuristic**, not a verified
  ground truth for either source project's actual conventions. See its
  section above.

## Extending this toolkit

Roughly in order of effort:
- Resolve the open question above: does the two-corpora disagreement
  survive controlling for sample size, or for CISI's motif/site
  homogeneity?
- Add a proper PyTorch transformer once more real data is in hand, sized
  to the corpus (still likely small; a few thousand four-to-five-sign
  sequences is not much training data).
- Build the CNN/YOLO seal-image segmentation pipeline (ASR-Net/MI-Net
  style) separately. That needs actual seal photographs, a different
  data-acquisition problem from the text-sequence work here.
- Sign-image visual similarity clustering against other Bronze Age
  scripts (Proto-Elamite, Sumerian) needs glyph image data per script,
  harder to source than text sequences. Treat as a stretch goal.

## Citations

See `CITATIONS.md` for every data source, paper, and tool this project
relies on, including licensing notes on the two real-data repositories.
