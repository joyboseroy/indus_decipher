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

This codebase was developed iteratively with Claude (Anthropic) doing the
implementation, debugging, literature search, and drafting, under human
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
  corpora used to disagree" for why that matters.

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
(chance is 50%). That is the most specific finding in this project so
far: the real corpus contains sequential structure beyond what a bigram
(order-1) model explains. It does not by itself say what that higher-order
structure is, only that it exists.

Run it yourself with `python3 experiments/permutation_controls.py`.

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
  effect.** The cross-site stability table above shows every substitution
  class replicating worse at Harappa than at Mohenjo-daro, and
  Mohenjo-daro supplies more of the corpus. Before concluding anything
  site-specific, subsample Mohenjo-daro DOWN to Harappa's motif-labeled
  count and see whether the gap shrinks or persists. This is the same
  logic as "Resolved: why the two real corpora used to disagree" above,
  applied to the substitution-graph result instead of the falsification
  harness.
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
- **Implement Witten-Bell smoothing** in `analysis/ngram.py` alongside
  the current add-alpha smoothing, so the top-90%-mass restoration metric
  can be compared to Yadav et al.'s published ~75% figure on genuinely
  equal terms, rather than with the "large candidate-set size" caveat
  this toolkit currently has to attach to that number.
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
