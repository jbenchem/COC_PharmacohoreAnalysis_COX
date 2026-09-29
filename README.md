# COC pipeline — pharmacophore build

**Pinned data release: ChEMBL 37 (1 May 2026).**

A ligand-based modelling pipeline that featurises molecules as a **2D
pharmacophore signature** rather than a substructure fingerprint, and trains on
**one declared assay format** without re-arguing that choice.

**Start with `SETUP.md`.** Three commands from an empty GitHub repo to a screened
compound list.

---

## Why this package exists

The predecessor used ECFP4. It worked on its own chemistry — scaffold-split
margin +0.141 (95% CI +0.135 to +0.159), classifier AUC 0.845 against confirmed
inactives, y-scrambling clean — and then **refused 88.5% of a 2,915-compound
ethnopharmacological library**.

The refusal was investigated rather than accepted. Median applicability-domain
distance for the library was 0.751 against 0.299 for the training set itself, and
no threshold between 0.6 and 0.9 gave defensible coverage. Two explanations were
tested and rejected:

| hypothesis | test | result |
|---|---|---|
| glycosylation and size explain it | median MW and sugar-ring fraction, accepted vs refused | 341 vs 302 Da, 16% vs 17% — **rejected** |
| the library lacks stereochemistry | fully specified stereocentres | library 82%, training 79% — **rejected** |

What is left is **chemotype**. The training set is diaryl heterocycles,
sulfonamides and arylacetic acids; the library is terpenoids, alkaloids,
phenylpropanoids and polyketides. Different ring systems.

A substructure fingerprint asks *"is this exact atom environment present?"*, so an
unfamiliar scaffold sets unfamiliar bits and looks like nothing the model knows. A
pharmacophore asks *"is there a donor here, an acceptor this far along, a
hydrophobe beyond it?"* — which a terpenoid and a coxib can both satisfy. That
abstraction is the reason for this package.

**It is a hypothesis, not a fix.** Whether it holds is what the pipeline measures.
A lower refusal rate alone would prove nothing: a less discriminating similarity
measure produces a looser gate without any extra predictive power. Both the
refusal rate *and* the margin over null have to move.

Full evidence: `RESULTS-2026-09-29.md`.

---

## What changed from the ECFP4 pipeline

| | ECFP4 build | this build |
|---|---|---|
| Features | 2,048 Morgan bits + 217 descriptors | **4,096 folded Gobbi 2D pharmacophore bits** + 217 descriptors |
| Conformers needed | no | no |
| Stereochemistry needed | no | no |
| Assay format | chosen, then re-tested every run | **declared and required**, never re-tested |
| Splits | scaffold, temporal, format-held-out | **scaffold, temporal** |
| Characterisation | confound test + paired cross-format deltas | **representation coverage** + scaffold concentration |
| Control 2 | inactives pooled across formats — **a bug** | inactives **restricted to the training format** |

The dropped work is not lost, it is settled: structure predicts assay format at
AUC 0.887 against a 0.489 shuffled null, and two of three format transitions carry
per-compound divergence beyond measurement noise (excess SD 0.70 and 1.02 log
units) that no constant correction removes. Re-running that every build would be
re-proving a result, not testing one.

The Control 2 fix is not cosmetic. Pooling inactives across formats while the
model trained on one cost about 0.15 AUC of pure bookkeeping error — 0.599 pooled
against 0.749 filtered on the same model.

---

## Files

| File | What it does |
|---|---|
| **`pipeline.py`** | **The driver.** `census` → `build` → `screen` |
| **`targets/TEMPLATE.json`** | **The config.** Copy per target; every field carries a note on what it does and how it fails |
| `fetch_target.py` | ChEMBL web API → local SQLite subset, by UniProt accession |
| `stereo_status.py` | Stereochemical status per compound, with provenance on every derived value |
| `extract.py` | Curation cascade, standardisation, deduplication → `curated/*.csv` + `curation_log.md` |
| `featurise.py` | SMILES → folded Gobbi 2D pharmacophore signature + RDKit descriptors |
| `explore_space.py` | Four-panel data characterisation. Describes the data; validates nothing |
| `train_model.py` | RF + XGBoost vs four nulls across two splits, plus two negative controls |
| `train_classifier.py` | Active / not, using the measured inactives the potency model cannot, against a thresholded-regressor baseline |
| `screen.py` | Your own compound list against the built models. Caches to `models/`, so later lists screen in seconds |
| `feature_blocks.py` | Pharmacophore / descriptors / both — the block ablation, in one place |
| `census.sql`, `run_census.py` | Standalone whole-database census queries. Needs the full release |
| `RESULTS-2026-09-29.md` | The measured benchmark results this package is built on |

Nothing needs Python on your own machine.

---

## Reading the pharmacophore signature

The Gobbi factory defines six feature families — donor, acceptor, aromatic,
hydrophobe, acidic, basic — and enumerates **pairs and triplets** of them binned
by topological distance. 39,972 raw bits, folded to 4,096.

Two consequences worth knowing before you trust a number.

**Folding collides distinct terms onto one column.** The collision count is in
`featurisation_report.md`. It is applied identically to every molecule so it does
not bias a comparison, but a single column is not interpretable as one
pharmacophore feature.

**A molecule with no pharmacophore features sets no bits.** Biphenyl has two
aromatic centres, no donors or acceptors, and scores zero. Those molecules run on
descriptors alone, and any claim about the pharmacophore representation does not
apply to them. Panel D of the characterisation figure is that count, and
`featurise.py` warns when it is large.

Polyhydroxy compounds go the other way — glucose sets 276 raw bits against
celecoxib's 182 — so sugars may look more similar to each other under this
representation than under ECFP4. If the refusal rate drops, check whether it
dropped for that reason before claiming chemotype transfer.

---

## ChEMBL, licensing, and the full release

`census.sql` Parts C and D scan every target in the database and need the full
release rather than the API subset. A default Codespace is 2-core / 8 GB / 32 GB
disk and the SQLite release does not comfortably fit — pick a larger machine type,
or use a lab workstation. Check the file size on the FTP listing before starting a
download you have not budgeted disk for.

**ChEMBL is CC BY-SA.** Share-alike propagates to a curated dataset derived from
it and released. Work out what licence your derived dataset carries and state it.
