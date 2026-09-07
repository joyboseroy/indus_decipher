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

This codebase was developed iteratively with the help of Claude (Anthropic), under human
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
(see "Real data" below) and the pipeline runs cleanly on both. An earlier
disagreement between them on the falsification harness was investigated
and resolved; see "Resolved: why the two real corpora used to disagree."

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
  adversarial_null_model.py    statistics-matched non-linguistic generator
                                (see "The adversarial null-model test")
  stratified_null_model.py     motif/site-stratified non-linguistic
                                generators (see "Stage 2" section)
  permutation_nulls.py         four permutation-based controls (see
                                "Permutation controls")
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
  substitution_graph.py    weighted, attributed substitution graph and
                         cross-site stability testing (networkx-based)
  sign_embeddings.py       PPMI+SVD distributional sign embeddings (see
                         "Contextual sign embeddings")
  falsification.py        feature-vector classification against three known
                         synthetic generative systems
models/
  transformer_mlm.py     small NumPy, zero-dependency masked-language
                         transformer for bidirectional sign prediction
main.py                  end-to-end pipeline / report generator
experiments/
  corpus_divergence.py    investigates and resolves the two-real-corpora
                         disagreement described below (sample-size curve,
                         matched-subsample test)
  adversarial_null_test.py  the statistics-matched null-model discrimination
                         test (see "The adversarial null-model test")
  permutation_controls.py  four permutation-based controls locating where
                         the classifiable signal lives (see "Permutation
                         controls")
  bootstrap_classification_ci.py  proper confidence intervals on
                         classification, and the leakage bug found while
                         building it (see "Bootstrap confidence intervals")
  substitution_graph_analysis.py  runs substitution_graph.py on the large
                         corpus, including the cross-site stability test
  dependency_order_curve.py  order-1..6 information-gain curve, and the
                         n-gram sparsity wall it hit (see "The
                         dependency-order curve")
  stratified_dependency_test.py  Stage 2: does motif/site composition
                         explain the order-3 signal? (see "Stage 2")
  allograph_granularity_test.py  resolves the CISI allograph-granularity
                         question (see "Allograph granularity")
  sign_embeddings_analysis.py  cross-validates embeddings against the
                         substitution graph (see "Contextual sign
                         embeddings")
  cross_site_held_out_validation.py  discovery/evaluation split across
                         sites, the independence fix (see "Cross-site
                         held-out validation")
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
  unicorn-motif seal from Mohenjo-daro; see "Resolved: why the two real
  corpora used to disagree" for why that matters. This source data is
  genuinely richer than a flat sign sequence: each grapheme carries a
  damage code, a line number, and a 0-100 subjective uncertainty score
  (now captured in this project's schema as `mean_uncertainty`, distinct
  from `damaged`), plus per-sign allograph feature vectors. Two further
  exports represent each grapheme at finer granularity than the bare
  primary sign ID: `data/cisi_real_corpus_allograph.csv` (every distinct
  allograph is its own sign) and `data/cisi_real_corpus_hierarchical.csv`
  (an allograph is only split out when independently well-attested at
  least 3 times corpus-wide, otherwise it collapses to the primary sign).
  See "Allograph granularity" below: this was resolved, not left open,
  and it turned out to be a real effect, not sparse-data noise.

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
100% on the six features used (see the module docstring for the exact
list, and for a correction made to this feature set, described below).
`main.py --extended` additionally classifies whichever corpus you loaded
against these three reference families and reports the nearest match
with distances, not a verdict: "resembles the language-like generator
most closely" is a statement about a nearest-centroid distance to three
specific synthetic corpora, not a claim about the real script.

Both real corpora currently classify as `civ_a_language_like`. See
"Resolved: why the two real corpora used to disagree" below for how that
became a consistent answer, and read it before treating this
classification as strong evidence either way.

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

**Important methodological note, discovered while investigating the
corpus disagreement below:** a raw count of these paradigm classes is
NOT a stable per-corpus statistic. It depends heavily on corpus size,
because more inscriptions add more substitution edges, which mechanically
merges what would otherwise be several separate classes into fewer,
larger ones. A "paradigm classes per 1,000 inscriptions" density
statistic was originally included as a falsification-harness feature and
had to be removed once this size dependence was traced as the actual
cause of an apparent disagreement between the two real corpora (see
below). The minimal-pair mining itself is still useful, and the
motif-corroboration finding above is unaffected, but any summary
statistic built from raw paradigm-class counts should be checked across a
range of corpus sizes before being trusted, not used as-is.

## Resolved: why the two real corpora used to disagree

Earlier versions of this project reported that after the reading-direction
fix, the large corpus classified as `civ_a_language_like` while the small
CISI corpus classified as `civ_c_mixed`, and treated this as an open
question with several candidate explanations (sample size, CISI's
unicorn/Mohenjo-daro homogeneity, or transcription convention
differences).

`experiments/corpus_divergence.py` tested these directly:

- **Sample-size curve:** random subsamples of the large corpus at sizes
  from 50 to 2,543 were classified 30 times each. The result was highly
  unstable at small and medium sizes (P(language-like) bounced between
  0.03 and 0.57 for N below 1,000) and only became consistently
  language-like once N reached roughly 1,500, far above either real
  corpus's actual usable size other than the large corpus's own full
  count.
- **Matched subsample:** filtering the large corpus to
  site == "Mohenjo-daro" and motif starting with "Bull1" (this scheme's
  field-symbol code for the "unicorn," the single most common Indus seal
  motif in the literature) produced 638 inscriptions that classified as
  mixed, same as CISI. But RANDOM same-size (638) subsamples of the large
  corpus, with no site or motif restriction at all, also classified as
  mixed 30 out of 30 times. That ruled out CISI's unicorn/Mohenjo-daro
  composition as the driver: an unrestricted random sample of the same
  size behaved identically.

That pointed squarely at sample size, and inspecting the seven individual
features directly confirmed it: six of the seven were essentially flat
across sample sizes from 179 to 2,543, but
`paradigm_classes_per_1000_inscriptions` swung roughly 40-fold (61.5 at
N=179 down to 1.6 at N=2,543). Removing that single feature from the
classifier and rerunning every test above made the disagreement vanish
entirely: both real corpora, and every random subsample size from 50 to
2,543, now classify consistently as `civ_a_language_like`.

`analysis/falsification.py` has been corrected accordingly (six features,
not seven; see that module's docstring for the full account). The
takeaway kept for future work: a feature that reaches 100% self-test
accuracy on matched-size synthetic reference corpora can still silently
encode sample size rather than the property it's named after, and that
will not show up in a self-test that only ever compares corpora of the
same size to each other. Any future feature added to this classifier
should be checked across a range of real corpus sizes, the way this one
eventually was, before being trusted.

This does not mean "the Indus script is language-like" is now
established. It means one specific artifact that was producing an
apparent disagreement between two real corpora has been found, explained,
and fixed, and with it removed, the two corpora currently agree with each
other on this one nearest-centroid statistical test against three
specific synthetic generators. That is a narrower and more defensible
claim, and it's the one this project is making.

## The adversarial null-model test

The three synthetic civilizations in the falsification harness are each
built from their own invented vocabulary and rules, so a classifier
telling them apart from the real corpus isn't a very demanding test: they
differ in almost every surface respect. A harder, more honest question is
whether the real corpus can be told apart from a non-linguistic system
that shares its exact surface statistics.

`data/adversarial_null_model.py` builds exactly that: a generator that
reuses the real corpus's own empirical length distribution, initial-sign
distribution, final-sign distribution, and overall sign-frequency
distribution, then draws every sign in a synthetic inscription
INDEPENDENTLY from the relevant marginal. There is no bigram or
higher-order dependency at all between consecutive signs; only the
position class (initial, final, or middle) and the raw frequency table
carry over from the real data.

`experiments/adversarial_null_test.py` compares 8 real subsamples against
8 instances of this matched null model, all at 500 inscriptions, first by
listing the six classifier features side by side, then with the same
leave-one-out discrimination test used for the three-civilization
self-test. The four features that are matched by construction
(`zipf_gamma`, `zipf_r2`, `top_sign_final_share`, `mean_length`) came out
close between real and null, as expected (0.8 to 12 percent apart). The
two features that depend on genuine sequential order rather than position
identity (`conditional_entropy`, `perplexity_ratio_n2_n1`) came out
substantially different (31 and 68 percent apart), and the classifier
achieved 100% leave-one-out accuracy telling real data apart from the
matched null using all six features together.

This is a real, and reasonably informative, result: it means the real
corpus's classification as "language-like" is not simply an artifact of
its length distribution, vocabulary size, or which signs happen to sit at
the start and end of an inscription, since a system built to match all of
that exactly is still cleanly distinguishable. What actually carries the
distinguishing signal is sequential dependency between neighboring signs,
which is a substantially more specific and more interesting property than
"the numbers are in the right range." It is still not evidence that the
underlying system is a spoken human language specifically, only that it
has more sequential structure than shallow position-and-frequency effects
can explain, which is the same qualified claim the classical entropy
literature has always made and the one this project is comfortable
standing behind.

Run it yourself with `python3 experiments/adversarial_null_test.py`.

## Permutation controls: locating where the signal lives

The adversarial null model above tests one thing: real corpus versus a
system with no sequential dependency at all. `experiments/permutation_controls.py`
sharpens this into four controls, each destroying a different, precisely
scoped piece of structure while operating directly on the real corpus's
own tokens rather than resampled marginals:

1. `within_inscription_shuffle`: shuffle each inscription's own signs into
   a new order. Destroys all within-inscription order.
2. `global_shuffle`: pool every sign token corpus-wide and reshuffle,
   re-cut using the original length sequence. Destroys order AND which
   specific signs co-occurred in the same inscription.
3. `position_preserving_shuffle`: hold each inscription's own observed
   initial and final sign fixed, shuffle only the middle. Destroys only
   middle-sequence order.
4. `bigram_markov_null`: generate fresh sequences by sampling forward
   through the REAL empirical bigram transition table, including a
   trained end-of-sequence token so that stopping behavior itself comes
   from real data rather than an externally fixed length. Destroys only
   dependency beyond order-1 (trigram and higher).
5. `trigram_markov_null`: same idea one order higher, sampling from the
   real order-2 transition table with backoff to bigram then unigram
   statistics for the many order-2 contexts too sparse to estimate
   directly (added later, alongside "Order 3 is validated; order 4 and beyond is inconclusive"
   below). Destroys only dependency beyond order-2.

An earlier version of control 4 fixed each generated sequence's length to
a real observed value and let the chain run exactly that many steps. That
produced a large, misleading gap on the `top_sign_final_share` feature
(a bigram-generated sequence stopped by external fiat has no way to
reproduce a real closing-sign tendency that is actually encoded in real
P(END | current sign) probabilities). Adding a proper trained END token
fixed this: with real stopping behavior included, five of six features
came within 10% between real and this control.

Result: even against the fairest version of the bigram-order control,
the classifier still discriminates real data from it at 93.8% accuracy
(chance is 50%).

**Correction, added after building `experiments/dependency_order_curve.py`
(see below):** this section originally stated that 93.8% figure as
evidence the real corpus "contains sequential structure beyond what a
bigram model explains." That claim was too strong. The discriminating
signal in this test is concentrated in `perplexity_ratio_n2_n1` and
`conditional_entropy`, both of which compare only order-1 against
order-2 statistics; nothing in this test actually measures order-3 or
higher. A genuine order-3+ effect is one possible explanation for that
93.8%, but so is a subtler difference in how precisely order-1/2
statistics transfer between the real corpus and a freshly-generated,
same-size synthetic sample, which is a real effect but a much narrower
one than "beyond bigram structure." The defensible claim from this
section alone is: **the real corpus is distinguishable from a
bigram-order-matched null**, full stop, without a claim about which
order the distinguishing information lives at. See "The dependency-order
curve" below for the direct attempt to settle this, and why it could not.

**Follow-up confirming why this classifier is the wrong tool for the
order question at all:** once `trigram_markov_null` existed (see "Order
3 is validated; order 4 and beyond is inconclusive" below), it was added
to this same test.
It also scores 93.8%, identical to the bigram-order null, despite having
real order-1 AND order-2 dependency built in. If this classifier were
tracking the order of missing structure, a null with MORE real structure
built in should be harder to distinguish from real data, not equally
easy. It isn't harder here, which confirms this six-feature classifier
is picking up on something other than a clean measure of dependency
order, and is why "The dependency-order curve" below uses a different,
more direct method (cross-validated entropy at each order) rather than
extending this classifier test further.

Run it yourself with `python3 experiments/permutation_controls.py`.

## The dependency-order curve: a real trigram-level finding, after fixing the method that first hid it

The natural next question after the correction above is direct: does
predictive information keep increasing as more context (order 3, 4, 5...)
is added, or does it saturate at order 2? The first attempt at this,
using this project's existing add-alpha smoothing, produced a genuine
negative finding about the METHOD rather than the script: held-out
entropy for ALL THREE corpora tested (real data, the bigram-order null,
and the adversarial no-dependency null), including the null that by
construction has zero real dependency at any order, started RISING at
order 3 and kept rising through order 6. That is the textbook signature
of n-gram sparsity: add-alpha smoothing cannot handle the exploding
number of distinct order-3+ contexts on a corpus this size, most seen
zero or one times, and backs off toward something close to a uniform,
uninformative distribution instead of a real higher-order estimate.
Since even the zero-dependency null showed the identical rising curve,
nothing about order 3 and up was interpretable this way, for any corpus.

This motivated implementing interpolated Kneser-Ney smoothing in
`analysis/ngram.py` (`KneserNeyModel`), the standard fix for exactly this
sparsity failure mode in n-gram language modeling, and rerunning the
same three-corpus comparison. The first version of that implementation
had a real bug of its own, caught the same way every bug in this project
has been caught: it returned IDENTICAL perplexity at every order from 1
through 6, which is only possible if every order was silently computing
the same thing. The cause was an off-by-one context-length mismatch in
the recursive backoff, fixed and documented in the class's own
docstring.

With that fixed, a real, three-way-validated finding emerged:

| Corpus | Information gain at order 3 (trigram) |
|---|---|
| Real corpus | **+0.143 bits** |
| Bigram-order null (real order-1 dependency only) | -0.140 bits |
| Adversarial null (no dependency at all) | -0.179 bits |

Real data shows a genuine POSITIVE gain from adding a third sign of
context. Both nulls, which have no real trigram-level dependency by
construction, show a NEGATIVE gain at the identical order using the
identical method, the expected sparsity cost with no real signal to
offset it. This is not asserted from one curve: it is a three-way
comparison where two independently-constructed controls both behave as
predicted and only the real data breaks the pattern.

This restores, on firmer and more specific footing than before, a claim
close to what the original (pre-correction) permutation-controls section
asserted: **there is real structure in the corpus beyond what a bigram
model captures, and it is visible specifically at trigram order.** Beyond
order 3, the real corpus's own curve turns negative again (order 4
through 6 all show small negative gains), consistent with sparsity
reasserting itself even under Kneser-Ney smoothing at these higher
orders on a corpus this size, so this result should be read as "genuine
trigram-level structure, evidenced concretely," not "structure at every
order we could compute."

One important caveat: the discount parameter (0.75) is the standard
default from the n-gram literature, not tuned against this corpus via
held-out data. A discount sensitivity check (does the order-3 finding
survive discount values from, say, 0.5 to 0.9) has not been run yet and
would be a reasonable next step before treating +0.143 bits as a precise
number rather than a directionally clear one.

## Order 3 is validated; order 4 and beyond is inconclusive, not "saturated"

The natural follow-up to the order-3 finding is whether it keeps going:
does real data still show an edge at order 4 over a null that already
has real order-1 AND order-2 dependency baked in? `data/permutation_nulls.py`
gained a `trigram_markov_null` generator for exactly this (same design as
`bigram_markov_null`, one order higher, with backoff to bigram then
unigram statistics for the many order-2 contexts too sparse to estimate
directly, the generation-side version of the same sparsity problem
Kneser-Ney solves on the evaluation side). More precisely, per external
review, this is a **backoff second-order Markov null**, not a pure
trigram Markov chain: its predictions fall back through bigram and
unigram statistics whenever a specific two-sign context was never
observed, which is necessary given the corpus size but means it isn't
the textbook uniform-context trigram model the shorter name might
suggest. The code keeps the shorter `trigram_markov_null` name (it is
still generating from real order-2 conditional statistics wherever
they're available), but this is the precise description for anyone
citing this result.

The result at order 4 is not a clean three-way split the way order 3
was. All four corpora (real, trigram-order null, bigram-order null,
adversarial null) show small NEGATIVE gains, ranging from -0.045 to
-0.142 bits. Real data's own order-4 gain (-0.088) sits in between the
trigram-order null's (-0.045, closest to zero of the four) and the two
nulls lacking real order-2 dependency (-0.140, -0.142). There is no
validated positive signal here: real data does not clearly separate from
a null that already has real order-1-and-2 structure built in, the way
it clearly separated from bigram-only and no-dependency nulls at order 3.

The precise, defensible summary, worth stating exactly rather than in
shorthand: **evidence for higher-order sequential structure is
statistically validated at order 3, relative to independently generated
no-dependency and first-order Markov controls; evidence beyond order 3
is currently inconclusive, not absent.** An earlier draft of this
section used the phrase "saturating around order 3," which reads as a
positive claim that structure stops there. That overstates what a
negative result can show: the order-4 negative gain is equally
consistent with real order-4 structure that estimation limitations
(sparsity, even under Kneser-Ney, at this corpus size) simply cannot
detect yet, not with structure being genuinely absent. Building 4-gram
and 5-gram null generators to chase this further is probably not worth
the effort without a specific reason to expect a real order-4 effect
first, which is a claim about priority, not about what the data has
established. If a future finding (a structural or archaeological
hypothesis) predicts a specific order-4+ pattern, that would be a reason
to revisit this with a targeted test rather than another blind extension
of the
curve.

Run it yourself with `python3 experiments/dependency_order_curve.py`.

Run it yourself with `python3 experiments/dependency_order_curve.py`.

## Bootstrap confidence intervals, and a leakage bug found along the way

Every classification reported above until this point was a single point
estimate on one corpus. `experiments/bootstrap_classification_ci.py` was
built to replace that with a proper confidence interval, and its first,
textbook-standard version produced an alarming result worth documenting
rather than hiding: resampling the large corpus WITH replacement at its
own full size (the standard nonparametric bootstrap) flipped its
classification from language-like, its result in every other test in
this project, to mixed in 97% of resamples.

Direct investigation traced this to a real bug in combining
with-replacement bootstrap with this project's cross-validated perplexity
feature. Resampling with replacement at full size produces roughly 37 to
49% exact duplicate sequences. When a duplicate lands in both a training
fold and the held-out fold of the internal k-fold split, a bigram model
can effectively memorize it from training and then predict it correctly
in the held-out fold, a leak that a unigram model, unable to memorize
whole sequences, benefits from far less. Measured directly on one
resample: bigram cross-validated perplexity dropped 26% purely from this
leakage, while unigram perplexity dropped only 5%, comfortably enough of
a gap to flip `perplexity_ratio_n2_n1` and change the nearest-centroid
label.

The fix is subsampling WITHOUT replacement at a fixed fraction (80% by
default) of each corpus, which produces genuine resample-to-resample
variability without ever creating a duplicate that could leak across a
fold boundary. With that fix, the results are tight and consistent:

| Corpus | N | P(language-like) | 95% CI |
|---|---|---|---|
| Large corpus (indus_website) | 2,543 | 1.000 | [0.981, 1.000] |
| CISI corpus | 104 | 0.990 | [0.964, 0.997] |
| Mohenjo-daro + unicorn matched subset | 638 | 1.000 | [0.981, 1.000] |

This both confirms the earlier point-estimate finding and gives it a
real confidence level, and it is a second concrete instance (after the
`paradigm_classes_per_1000_inscriptions` bug) of this project's central
methodological lesson: a resampling or feature-engineering choice that
looks standard can silently interact with a downstream pipeline step
(here, k-fold cross-validation) in a way that produces a confident,
wrong-looking answer, and the only way to catch it is to notice when a
result contradicts everything else you already know and go find out why
rather than reporting it.

Run it yourself with `python3 experiments/bootstrap_classification_ci.py`.

## The substitution graph upgrade, and a cross-site stability finding

`analysis/substitution_graph.py` turns the motif-corroborated minimal
pairs into a proper weighted graph (via networkx) instead of a plain
connected-components view: edges carry a weight (raw pair count) and a
`distinct_contexts` count (the number of unique site+motif combinations
supporting that edge, a conservative proxy for independent corroborating
evidence, since five pairs from one site+motif combination are one piece
of evidence, not five). Greedy modularity community detection is offered
alongside plain connected components, since the latter is known from
`minimal_pairs.py`'s own documentation to over-merge at low thresholds;
on the large corpus, modularity splits what connectivity sees as 18
components into 19 communities, resolving at least one additional
genuine sub-structure that raw connectivity had merged.

`experiments/substitution_graph_analysis.py` then asks the harder
question: do these classes survive being recomputed independently on a
single site's own data? For the five largest communities, restricted to
signs that actually appear in each site's own vocabulary, and compared by
Jaccard overlap against whatever class the same procedure finds when run
on that site alone:

| Community | Size | Mohenjo-daro Jaccard | Harappa Jaccard |
|---|---|---|---|
| 0 | 15 | 0.64 | 0.17 |
| 1 | 12 | 0.92 | 0.30 |
| 2 | 11 | 0.80 | 0.44 |
| 3 | 8 | 0.71 | 0.38 |
| 4 | 6 | 1.00 | 0.00 |

Every community replicates far better at Mohenjo-daro than at Harappa,
one community (community 4) not replicating there at all. Mohenjo-daro
supplies the plurality of the corpus (1,202 of 2,543 inscriptions) and of
the motif-labeled subset these classes are built from, so at least part
of this gap is likely an ordinary sample-size effect rather than a
genuine site-specific grammatical difference, but that has not been
tested directly yet (see "Extending this toolkit"). Read this table as
what it is: most of the current motif-corroborated substitution classes
are substantially better supported by Mohenjo-daro material than shown to
be corpus-general, and that qualifier belongs on any claim made about
them until it is checked further.

Run it yourself with `python3 experiments/substitution_graph_analysis.py`.

## Allograph granularity: resolved, and it is a real effect, not sparsity noise

The mayig/CISI source data (see CITATIONS.md) carries more than plain
sign IDs: each grapheme has a documented feature vector, damage and line
number and a 0-100 subjective uncertainty score as defaults, then extra
allograph-specific features per sign (for example, sign P086's own
feature file defines branch_factor, branch_count, branch_direction, and
final_branch_shape). Earlier versions of `convert_cisi_to_csv.py`
collapsed every allograph of a sign down to its bare primary ID,
discarding this. Three things were added:

- `mean_uncertainty`, a real field now carried through the schema
  (`data/loader.py`), separate from `damaged`: a grapheme can be fully
  undamaged but still visually ambiguous to the annotator, and 78 of 179
  CISI inscriptions carry a nonzero uncertainty score that was previously
  silently dropped.
- `data/cisi_real_corpus_allograph.csv`, where each grapheme becomes its
  primary sign ID plus its own allograph-specific feature values as a
  suffix (e.g. `P086_3-1-0-0`), so two visually distinct allographs of
  the same primary sign become distinct sign identities.
- `data/cisi_real_corpus_hierarchical.csv` (`granularity="hierarchical"`
  in the converter), a middle representation: an allograph is only split
  out from its primary sign when that SPECIFIC allograph is independently
  attested at least 3 times corpus-wide; rarer variants collapse back to
  the primary sign rather than being treated as evidence of a
  functionally distinct sign from a single annotation.

Last session, the full-allograph classification flip (language-like to
mixed) was reported as inconclusive, on the reasoning that naive full
fragmentation (230 signs over 104 inscriptions, 61% singletons) was
plausibly just too sparse to trust. `experiments/allograph_granularity_test.py`
was built to settle this by adding the conservative hierarchical
representation as a tiebreaker:

| Representation | Signs | top_sign_final_share | Classification |
|---|---|---|---|
| Primary | 142 | 0.356 | `civ_a_language_like` |
| Hierarchical (min_count=3) | 185 | 0.000 | `civ_c_mixed` |
| Full allograph | 230 | 0.000 | `civ_c_mixed` |

The hierarchical representation agrees with full allograph, not primary,
despite only splitting allographs that are independently well-attested
at least 3 times across the whole corpus, specifically the check that
should have ruled out the sparsity explanation if that's all this was.
It didn't rule it out; the flip persisted anyway.

Tracing why mechanically confirms this is a real effect: the corpus's
single dominant final-position sign at primary granularity, P324 (37 of
104 inscriptions' final positions), turns out to have at least six
allograph variants that are each independently well-attested (P324_0-0-0-1-1,
P324_1-0-1-1-0, P324_1-0-0-1-1, P324_1-0-2-1-0, P324_1-0-0-1-0,
P324_0-0-1-1-0). Even the single most common of these reaches only 11
final-position occurrences under the hierarchical threshold, nowhere
close to the primary-level 37. This is not a handful of singleton
variants creating noise; it's several genuinely common, genuinely
distinct visual forms sharing one primary sign ID, and treating them
as one sign is exactly what was concentrating the apparent final-position
signal that primary-granularity classification depended on.

The corrected conclusion: **CISI's primary-level "language-like"
classification is not robust to a defensible, conservative allograph
treatment.** Whether the primary-level or allograph-aware view is the
"correct" one to trust for a linguistic claim is a genuine open question
this project does not resolve, but it is no longer honest to describe
the flip as likely sparsity noise. It survives the specific test built
to rule that out. One remaining honest caveat: this test only ran the
falsification-harness classification, not the order-3 Kneser-Ney gain
test, at all three granularities; that gain came out negative (-0.012 to
-0.037 bits) for ALL THREE representations on this 104-inscription
corpus, meaning the order-3 finding validated on the large 2,543-inscription
corpus does not (yet) replicate on CISI at any granularity, most likely
because CISI is simply too small for that specific test, not because of
anything granularity-specific.

## Stage 2: does archaeological composition explain the order-3 signal, or does it survive conditioning on it?

The validated order-3 finding has three candidate explanations, per a
design proposed in external review:

- **H1 (linguistic/sequential):** the signal is intrinsic to sign
  sequencing, independent of which motif or site an inscription belongs to.
- **H2 (archaeological/compositional):** different motifs or sites simply
  favor different signs, and pooling these different populations together
  is itself enough to manufacture the appearance of trigram structure,
  with no real within-group sequential dependency at all.
- **H3 (mixed):** both contribute.

`data/stratified_null_model.py` tests this directly. It uses the exact
same zero-dependency generation logic as the original adversarial null
(every sign in a generated inscription is still drawn independently, no
real sequencing at all) but now computes the marginal distributions
SEPARATELY PER STRATUM (motif, site, or site+motif jointly) rather than
pooled across the whole corpus. If H2 were doing real work, a null that
captures "different groups prefer different signs," even with zero real
sequential dependency, should be able to reproduce more of the real
corpus's apparent order-3 structure than the original single pooled null
could. Strata with fewer than 20 real inscriptions fall back to the
corpus-wide pooled marginals rather than trusting a small sample's own
statistics, the same kind of backoff used elsewhere in this project.

Result: real data's order-3 information gain (+0.143 bits) exceeds every
one of the four nulls tested, including the site+motif-stratified one:

| Null | Order-3 gain |
|---|---|
| Real corpus | **+0.143 bits** |
| Pooled adversarial null (no stratification) | -0.179 bits |
| Motif-stratified null | -0.185 bits |
| Site-stratified null | -0.177 bits |
| Site+motif-stratified null | -0.180 bits |

Notably, stratification barely moved the null's own behavior at all,
four numbers within 0.008 bits of each other regardless of how much
archaeological context each null was given to work with. If H2 had any
real purchase, giving the null more archaeological structure to exploit
should have pushed at least one of these closer to zero or positive; none
did. Since each stratified null had every opportunity H2 would need to
succeed and none came close, this is real, specific evidence favoring
H1: the order-3 signal is intrinsic to sign sequencing, not an artifact
of pooling archaeologically distinct sign-frequency populations together.

This does not rule out H3 (some archaeological contribution alongside
real sequential structure) -- it rules out H2 being SUFFICIENT on its
own, which is the relevant test here, not a claim that motif and site
are irrelevant to everything in this corpus.

Run it yourself with `python3 experiments/stratified_dependency_test.py`.

## Contextual sign embeddings: two independent methods agree

Stage 2's last item: does distributional similarity corroborate the
substitution-graph communities, or are minimal pairs finding something
those communities alone can't be checked against?

`analysis/sign_embeddings.py` builds PPMI (positive pointwise mutual
information) plus truncated SVD embeddings per sign, the classic
distributional-semantics technique (essentially LSA applied to signs).
Chosen over reading out `models/transformer_mlm.py`'s own embedding
table deliberately: PPMI+SVD has no training dynamics to second-guess
and is fully deterministic given the corpus and a context window, so
there's no risk of mistaking a toy-scale model's optimization noise for
real distributional structure.

The two methods ask genuinely different questions. Minimal pairs (the
substitution graph) ask: do these two signs substitute for each other in
an otherwise-identical local context? Embeddings ask: do these two signs
tend to co-occur with similar OTHER signs, in general, across the whole
corpus? `experiments/sign_embeddings_analysis.py` tests whether they
agree: for each of the ten largest motif-corroborated substitution-graph
communities, compare the mean pairwise cosine similarity among the
community's own members against a random-pair baseline of the same size.

Result: **all 10 tested communities show higher internal similarity than
the random baseline**, several by a wide margin (community 7: 0.80
internal similarity vs. 0.20 baseline). Two structurally independent
methods, one local and strict, one global and loose, agree on which
signs cluster together. That is convergent evidence these communities
capture reproducible distributional and substitutional structure, not
an artifact specific to how the minimal-pair miner happens to work.
(An earlier draft of this section said "genuine functional classes" --
downgraded per external review, since "functional class" quietly smuggles
in an interpretation the next paragraph explicitly disclaims. The
weaker, more precise phrasing is the one this project should stand
behind.)

What this does NOT establish: what these classes mean. A sign class
found this way could be a morphological paradigm, a semantic category,
an administrative code family, or something else entirely; distributional
and substitutional agreement says the classes are real and stable, not
what function they serve. That is future work, not something claimed here.

**A more important limitation than the wording, flagged by the same
review: this test evaluates embeddings on the SAME corpus used to
discover the communities.** That isn't circular in the strict sense
(minimal pairs and PPMI co-occurrence are genuinely different signals),
but it does leave open a real confound: larger substitution communities
likely also skew toward higher-frequency signs, and higher-frequency
signs get less noisy embeddings for reasons that have nothing to do with
any real functional relationship. See "Cross-site held-out validation"
below for the actual fix, not just an acknowledgment of the problem.

Run it yourself with `python3 experiments/sign_embeddings_analysis.py`.

## Cross-site held-out validation: the fix for the independence concern above

The corroboration above has a real limitation, raised in external
review: embeddings were evaluated on the SAME corpus used to discover
the communities. Not circular in the strict sense (minimal pairs and
PPMI co-occurrence are genuinely different signals), but a real
confound remains open: larger communities likely skew toward
higher-frequency signs, and higher-frequency signs get less noisy
embeddings for reasons unrelated to any real functional relationship.

`experiments/cross_site_held_out_validation.py` fixes this properly, and
happens to make progress on the long-open Harappa/Mohenjo-daro stability
question at the same time, since both need the identical fix: discover
substitution communities using ONLY Mohenjo-daro inscriptions, train
PPMI+SVD embeddings using ONLY Harappa inscriptions (a fully disjoint
corpus, different site, no overlap with discovery at all), then test
whether the Mohenjo-daro-discovered communities still show elevated
internal similarity measured against Harappa's own, much smaller
(335-sign) vocabulary.

Result: **10 of 11 testable communities survive the fully disjoint
split** (one community's members simply weren't present in Harappa's
vocabulary in testable numbers, and one, community 2, did not
corroborate). That is a considerably harder bar than the original
same-corpus test, and most of the original result survives it.

This also reframes, rather than just confirms, the earlier cross-site
finding in "Stage 2: does archaeological composition..." above. That
section found substitution-graph communities replicate more weakly at
Harappa using strict Jaccard overlap on minimal-pair mining (which needs
literal matching-context substitution pairs, and Harappa has less
motif-labeled data to find them in). This result suggests the
DISTRIBUTIONAL signature of those same communities is still present in
Harappa's data even where the stricter substitution test struggled --
consistent with the earlier weak replication being partly a data-density
limitation of the minimal-pair method specifically, not clean evidence
the underlying classes are Mohenjo-daro-specific. Both results should be
read together, not the weaker one in isolation.

Run it yourself with `python3 experiments/cross_site_held_out_validation.py`.

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

Roughly in order of effort, and reflecting several concrete suggestions
this project received during external review of the resolved corpus
disagreement above:
- ~~Report classification with a confidence interval, not a point
  estimate~~ **Done, see "Bootstrap confidence intervals" above** (and
  note the leakage bug documented there before trusting a naive version
  of this).
- **Test whether Harappa's weaker class-stability is a sample-size
  effect, specifically.** "Cross-site held-out validation" above showed
  the DISTRIBUTIONAL signature of Mohenjo-daro-discovered communities
  largely survives evaluation on Harappa's own data (10 of 11), which
  points toward the earlier weaker Jaccard-based replication being a
  data-density limitation of minimal-pair mining specifically, not the
  classes being absent from Harappa. That's suggestive, not proof; the
  matched-size subsampling test originally proposed here (subsample
  Mohenjo-daro DOWN to Harappa's motif-labeled count, rerun the SAME
  strict Jaccard test, see if the gap shrinks) still hasn't been run and
  would settle the sample-size question directly rather than inferring
  it from a different method's result.
- ~~Resolve the allograph-granularity ambiguity properly~~ **Done, see
  "Allograph granularity" above.** Follow-up not yet done: rerun the
  order-3 Kneser-Ney test on the large corpus's own site+motif-known
  subset at multiple granularities, now that CISI alone was shown too
  small for that specific test to validate at any granularity.
- **Make the synthetic controls harder.** The three civilizations are
  deliberately quite distinct from each other, which is why the
  self-test hits 100%. A useful next test is a continuum between
  `civ_a_language_like` and `civ_b_administrative_code` (mixing
  parameter alpha from 0 to 1) to find the point at which this
  classifier's features stop being able to tell them apart, and multiple
  independent generator variants per class, so the classifier is checked
  against variation within a category, not just between categories.
- ~~Build an adversarial, statistics-matched null model~~ **Done, see
  "The adversarial null-model test" above.**
- ~~Add permutation controls to locate where the classifiable signal
  lives~~ **Done, see "Permutation controls" above.**
- ~~Upgrade minimal pairs into a weighted substitution graph with
  cross-site stability testing~~ **Done, see "The substitution graph
  upgrade" above** (though see the follow-up bullet above about
  Mohenjo-daro sample-size confound).
- **Add real external control corpora** (Sumerian, Old Tamil, Vedic
  Sanskrit) to `analysis/entropy.py`'s comparisons via
  `external_control_entropy()`, replacing or supplementing the synthetic
  random/rigid controls, for a result directly comparable to Rao et
  al.'s original entropy figures.
- ~~Implement Kneser-Ney smoothing~~ **Done, see "The dependency-order
  curve" above.** Two follow-ups remain: (1) a discount-sensitivity check
  (does the order-3 finding survive discounts from ~0.5 to ~0.9, not just
  the default 0.75) before treating the +0.143 bits figure as precise
  rather than directional; (2) rerunning the top-90%-mass restoration
  metric with `KneserNeyModel` instead of add-alpha, for a fairer
  comparison to Yadav et al.'s published ~75% figure than this project's
  current add-alpha-based version allows.
- **Add a proper pytest test suite.** Every result in this project has so
  far been checked by manually rerunning each script and confirming a
  clean exit code plus a by-eye read of the printed numbers. That has
  worked, but it doesn't scale and depends on remembering to do it.
  Converting the sanity checks already implicit in this process (does
  main.py run on all three corpora, does the falsification harness still
  hit 100% self-test, does KneserNeyModel give non-identical perplexity
  across orders, the exact kind of check that caught that bug) into real
  automated tests would catch a regression the next time a shared module
  like analysis/ngram.py or data/loader.py changes, rather than relying
  on remembering to rerun everything by hand.
- **Keep a lightweight experiment log** (corpus, N, direction, features,
  seed, result, timestamp) for every run that produces a number quoted
  anywhere outside this repo, so any reported figure can be traced back
  to the exact run that produced it. Every `experiments/*.py` script's
  JSON output is a first, informal version of this.
- Add a proper PyTorch transformer once more real data is in hand, sized
  to the corpus (still likely small; a few thousand four-to-five-sign
  sequences is not much training data). Compare it against unigram
  through 4-gram baselines on an identical train/test split, not in
  isolation, so any improvement is measured rather than assumed.
- Build the CNN/YOLO seal-image segmentation pipeline (ASR-Net/MI-Net
  style) separately. That needs actual seal photographs, a different
  data-acquisition problem from the text-sequence work here.
- Sign-image visual similarity clustering against other Bronze Age
  scripts (Proto-Elamite, Sumerian) needs glyph image data per script,
  harder to source than text sequences. Treat as a stretch goal.

A few things intentionally NOT on this list yet: candidate-language
testing (Dravidian/Indo-Aryan/Munda hypothesis comparison), LLM-driven
hypothesis generation over sign functions, and anything framed as working
toward "decipherment." Those are reasonable long-run directions once the
falsification and structural-analysis layers above are considerably more
battle-tested than they are today, but reaching for them now would be
the same kind of premature confidence this project's own README argues
against elsewhere. The corpus-disagreement investigation above is a
concrete example of why that order matters: the more exciting-looking
result (a real corpus reads as "language-like") turned out to depend on
a bug in one feature, not on anything about the script. The next
result that looks exciting deserves the same scrutiny before being
treated as a finding.

## Citations

See `CITATIONS.md` for every data source, paper, and tool this project
relies on, including licensing notes on the two real-data repositories.
