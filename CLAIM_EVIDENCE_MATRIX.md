# Claim -> Evidence Matrix

What this project actually establishes, what it doesn't, and exactly
where to look for the evidence behind each row. If you take one claim
from this project and cite it without checking this table first, you
are more likely than not to overstate it -- that has happened to this
project's OWN interpretation of its own results at least twice (see the
"Downgraded from an earlier claim" column).

| Claim | Evidence | Corpora | Null/control used | Strength | Does NOT establish | Downgraded from an earlier claim? |
|---|---|---|---|---|---|---|
| Real corpus has non-random sequential structure beyond frequency and position | Adversarial null test | indus_website | Statistics-matched independent-draw null | Strong (100% discrimination) | Language; any specific mechanism | No |
| Structure extends beyond first-order (bigram) dependency | Permutation controls | indus_website | Bigram-order Markov null | Moderate -- see next row, this claim was itself revised | Which order the structure lives at | **Yes.** Originally stated as "beyond bigram," later shown the discriminating classifier can't actually measure order this way |
| Structure specifically at order-3 (trigram) exists | Kneser-Ney order-3 information gain | indus_website AND M77 (independently) | Bigram-order null + adversarial null, both negative where real data is positive | Strong -- replicated across two independently-sourced corpora, survives a discount-parameter sweep 0.5-0.9 | Language; morphology; that order 4+ has no structure (genuinely untested, not "absent") | No, but see next two rows for what it does NOT mean |
| Order-3 structure requires linguistic morphology | Synthetic continuum: civ_a (real morphology) does NOT reliably show positive gain; civ_e (zero linguistic motivation, hierarchical admin nesting) DOES | Synthetic only | 5-seed stability check at N=800 through N=33,000 | **This claim is FALSE, established directly** | -- | **Yes.** Order-3 gain does not track "linguistic" in either direction on these generators |
| Order-3 test has genuine sample-size limits around N~2,500 | Real Sumerian and Sanskrit BOTH lose their (otherwise strong) positive signal when subsampled to indus_website's size | ETCSL, DCS Sanskrit | Full-scale vs. matched-subsample comparison, same corpus | Strong -- replicated independently in two unrelated languages | That the Indus signal at N=2,543 is therefore linguistic (it isn't shown to be either way -- see previous row) | No, this REFINES the interpretation of the order-3 finding rather than reversing it |
| Order-3 finding is not explained by pooling different archaeological populations | Motif-stratified, site-stratified, and site+motif-stratified nulls all remain negative where real data is positive | indus_website | Stratified adversarial nulls (zero real dependency, but real per-stratum sign preferences) | Moderate-strong | That site/motif have NO effect at all (not tested, only that they're not SUFFICIENT to explain this one result) | No |
| Substitution/distributional communities are real, not an artifact of the mining method | 10/10 motif-corroborated communities show higher internal distributional similarity than random baseline | indus_website | Random-pair baseline, same corpus | Moderate (same-corpus test only) | -- | -- |
| ...and generalize across sites, not just within the discovery corpus | 10/11 communities discovered on Mohenjo-daro ALONE survive evaluation on Harappa's distributional statistics ALONE | indus_website (site-split) | Fully disjoint discovery/evaluation split | Strong | Semantic or grammatical meaning of the communities | -- |
| Harappa's weaker apparent class-stability was a sample-size artifact, not a real site difference | Mohenjo-daro subsampled to Harappa's exact motif-labeled count, 100 trials; Harappa's real score falls within that distribution for 5/5 communities | indus_website (site-split) | Matched-size resampling | Strong | -- | -- |
| Whole-sequence uniqueness is unremarkable, not evidence of deliberate registration/identifier design | Real uniqueness sits AT OR BELOW the entire range of a 200-trial null distribution (independent-draw, matched vocabulary/length) | indus_website AND M77 (independently) | Adversarial null, 200/50 trials | Strong, replicated across two corpora | That the system is therefore linguistic (a lower-than-chance uniqueness rate has several possible explanations, not just one) | -- |
| CISI's classification (language-like vs. mixed) depends materially on allograph granularity | Primary-sign granularity: language-like. Hierarchical (conservative, min. 3 attestations) AND full allograph granularity: both mixed | CISI (3 granularities) | Cross-granularity comparison on the same 104 inscriptions | Moderate (small corpus) | Which granularity is "correct" (genuinely unresolved) | -- |
| The M77 headline result replicates on the actual classic-literature corpus | Order-3 gain +0.158 (M77) vs. +0.143 (indus_website); uniqueness-vs-null result also replicates | M77 (independently obtained and verified) | Same nulls as indus_website, rebuilt from M77's own data | Strong | -- | -- |
| Falsification-harness classification (language-like/administrative/mixed) is itself informative | indus_website classifies language-like; M77 classifies mixed -- same corpus family, different digitizations, genuinely different classifier output | indus_website, M77 | Reference centroids from 3 synthetic civilizations | **Weak/unresolved** -- this disagreement is unexplained | Anything, currently -- this is an open question, not a finding |
| Structural patterns are not an artifact of one digitization project's specific choices | Published WUCS network statistics (reciprocity, connectivity, beginner/ender asymmetry) replicate qualitatively (4/4) on indus_website, computed independently | indus_website vs. WUCS (published) | Within-sequence shuffle randomization (WUCS's own method) | Moderate (published-statistics comparison, not raw-data replication) | -- | -- |
| Morphology (in the linguistic sense) | -- | -- | -- | **Not established** | -- | -- |
| Semantic categories / word meaning | -- | -- | -- | **Not established** | -- | -- |
| Administrative/registration-code system specifically | -- | -- | -- | **Not established** (and the uniqueness test above is one specific strike against it) | -- | -- |
| Underlying language family (Dravidian, Indo-Aryan, Munda, or otherwise) | -- | -- | -- | **Not tested** | -- | -- |
| Sign-to-sound or sign-to-meaning values (decipherment) | -- | -- | -- | **Not attempted, deliberately out of scope** | -- | -- |

## How to read the "Downgraded from an earlier claim" column

This project has revised its own headline claims downward twice, in
public, in the same document that states the current claim -- not
quietly. If you're citing a specific number or claim from this project,
check this column: a "Yes" means an earlier, stronger version of that
exact claim exists somewhere in this project's history and was
deliberately walked back, usually after a synthetic control or a second
corpus produced a result the original framing couldn't explain. The
README's "The synthetic continuum" and "Permutation controls" sections
tell both stories in full.
