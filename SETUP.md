# Setup — from an empty GitHub repo to a screened compound list

**Three commands, one decision between the first and the second.** Everything
runs in the browser; nothing installs on your own machine.

```bash
python pipeline.py census --config targets/mytarget.json   # what exists
#   ... you choose one assay format ...
python pipeline.py build  --config targets/mytarget.json --repeats 5
python pipeline.py screen --config targets/mytarget.json --input mylist.csv
```

---

## 1. Make the repo

1. **github.com → New repository.** Name it, private is fine.
2. **Code → Codespaces → Create codespace on main.**
3. Drag `coc-pharm.zip` into the file explorer on the left.
4. In the terminal:

```bash
unzip -o coc-pharm.zip && rm coc-pharm.zip
pip install -r requirements.txt
python -c "
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
import sklearn, xgboost, scipy, joblib
print('ready —', Gobbi_Pharm2D.factory.GetSigSize(), 'pharmacophore bits available')"
```

> **Copy your edited configs aside before any future `unzip -o`.** It overwrites
> `targets/*.json` and your chosen `bao_format` goes with it. A file the zip does
> not ship — `targets/cox2-mine.json` — survives.

---

## 2. Describe your target

```bash
cp targets/TEMPLATE.json targets/mytarget.json
```

Set three things. Everything else has a working default and a `_note`.

```json
"name": "mytarget",
"targets": { "P35354": "COX-2" },
"target_type": "SINGLE PROTEIN"
```

**`targets` is UniProt accessions → short labels.** Accessions are stable across
ChEMBL releases; ChEMBL target IDs are not. The label becomes part of output
filenames, so keep it short. Two accessions builds two datasets side by side.

> **Whole-organism target?** Set `"target_type": "ORGANISM"` and, in `cascade`,
> set **both** `min_confidence_score` and `assay_type` to `null`. Organism-level
> assays score low by design, so leaving `min_confidence_score` at 8 silently
> deletes the entire dataset. This is the most common way to repoint the pipeline
> and get nothing back.

Leave `"filter": {"bao_format": null}` alone for now.

---

## 3. Census

```bash
python pipeline.py census --config targets/mytarget.json
```

Fetches everything ChEMBL holds for your target — every page cached, so a re-run
costs nothing and a crash resumes — then counts endpoint types and per-format
compound totals **after the same cascade the build will apply**. Writes
`census_<name>.md` and **stops**.

Pick one format and put it in the config:

```json
"filter": { "bao_format": "BAO_0000357" }
```

**This is required.** The build refuses without it. Pooling formats stacks
several different measurements under one column name, and the confound that makes
that unsafe was already measured: structure predicts assay format at AUC 0.887
against a 0.489 shuffled null, with excess SD of 0.70 and 1.02 log units on two
of three transitions. This pipeline does not re-argue that — it assumes it.

Read the assay descriptions before choosing. The largest format is not always the
right one: a purified-enzyme assay and a whole-blood assay measure different
things, and only you know which matches the question. Below roughly 500 compounds
expect the model to struggle against its nulls.

> The commonest mistake: putting the BAO code in `cascade.assay_type`. That field
> takes a single letter (`B`, `F`, `A`, `T`). The driver catches it, but it costs
> a run.

---

## 4. Build

```bash
python pipeline.py build --config targets/mytarget.json --repeats 5
```

| stage | what it does |
|---|---|
| **stereo** | Derives stereochemical status per compound. Trusts ChEMBL's `chirality` flag where it commits, derives from structure where blank — it was unpopulated for 83% of COX-2, and 5,493 compounds flagged "unknown" have no stereocentre at all |
| **extract** | Curation cascade, standardisation (salt strip → charge normalise → canonical tautomer), deduplication to one value per compound per format, `>` records kept as confirmed inactives. Writes `curation_log.md` |
| **featurise** | SMILES → folded Gobbi 2D pharmacophore signature + RDKit descriptors. No conformers, no stereochemistry |
| **characterise** | Four-panel figure: chemical space, potency distribution, scaffold concentration, **pharmacophore coverage**. Describes the data, validates nothing |
| **potency model** | RF + XGBoost against four nulls across scaffold and temporal splits, then two negative controls |
| **classifier** | Active / not, trained **with** the measured inactives, scored against a thresholded-regressor baseline so the second model has to justify existing |

`--repeats 5` reruns the scaffold cross-validation with the scaffold→fold
assignment reshuffled each time. The default of 1 measures one arbitrary
partition and understates uncertainty; use 5 for anything you intend to report.

### The ablation

```bash
python pipeline.py build --config targets/mytarget.json --blocks pharmacophore
```

Trains on the pharmacophore signature alone. `--blocks descriptors` does the
reverse. Outputs are suffixed so nothing overwrites the full run, splits and seed
are held fixed, and **the nulls do not move** — so margins are directly
comparable.

This matters more here than with a substructure fingerprint. A molecule that sets
**zero** pharmacophore bits is carried entirely by its descriptors, so
`--blocks pharmacophore` tells you what the signature is worth on its own and
`--blocks descriptors` tells you how much of the result never needed it. Panel D
of the characterisation figure is the count that decides whether you need to care.

---

## 5. Read the results, in this order

**`negative_controls.md` first.** It decides whether anything else is worth
reading.

- **Y-scrambling** — labels shuffled, model refit, margin must collapse to zero.
  It faces the same four nulls the real model did, so it is not passing against a
  softer bar. OECD validation principle 4; a panel will ask.
- **Confirmed inactives** — compounds ChEMBL records as `>`, never seen in
  training, **and restricted to the training assay format**. Cross-validation
  cannot answer "how often would this point me at something already known not to
  work", because it only ever scores compounds that had a measurable value.

**`curation_log.md`** — Table 1. Every cascade stage, what standardisation
collapsed, which groups disagreed with themselves, what was excluded and what it
cost.

**`featurisation_report.md`** — the fold collision count and, more importantly,
how many molecules set no pharmacophore bits at all.

**`table2.md`** — **read the margin over the best null, not the RMSE.** An RMSE of
0.7 means nothing until you know the null scored 0.75 or 1.4. Four nulls are
scored, and the **5-NN Tanimoto** one is the hard one: if the model cannot beat
it, the model is doing similarity lookup rather than learning structure–activity
relationships. Still useful, but a smaller claim — and one you should make
yourself rather than have a reviewer find.

**`table3_classification.md`** — active/not, both negative definitions, and
whether the classifier beats simply thresholding the potency model. This decides
the `--gate` flag in step 6.

**`figures/chemspace_*.png`** — panel D before the rest.

---

## 6. Screen your own compounds

One CSV, one column of SMILES. Tab-separated files are detected and read as such.

```bash
python pipeline.py screen --config targets/mytarget.json \
    --input mylist.csv --smiles-col SMILES --id-col ID --exclude-known
```

Column names are checked immediately, before anything expensive runs.

**The first run fits and saves.** Later runs load from `models/` — a cold run
took 19s and a warm one 5s on a small test set, and the saving is larger on real
data. The cache key covers the config fields that matter and the feature files'
size and modification time, so rebuilding the dataset or switching `--blocks`
forces a refit. `--refit` forces it manually.

### What it does, in order

1. **Standardise** through the identical cascade the training set went through.
2. **Domain gate.** `1 − mean Tanimoto to the 5 nearest training compounds`, on
   the pharmacophore signature, with the threshold at the **95th percentile of
   the training set's own neighbour distance**. Nothing chosen by hand. Past it,
   refuse — no score, because a number the model has no basis for is worse than
   silence.
3. **Classify** active or not.
4. **Predict pIC50**, only for what passed.

Every compound gets a row and a reason, refusals included.

> **This is a pharmacophore-similarity domain, not a substructure one.** A
> compound can be inside it while sharing no scaffold with anything in training —
> that is the entire point. It also means the refusal rate here is **not**
> comparable to an ECFP4 pipeline's number, only to another run of this one. To
> compare representations properly you need both refusal rate and margin over
> null, on the same compounds.

| flag | what it does |
|---|---|
| `--exclude-known` | drops compounds already in the training set from the shortlist and the enrichment test, matching on both the full InChIKey and its first block so a stereo-variant duplicate cannot slip past |
| `--ad-percentile N` | moves the threshold. 95 is the default |
| `--score-all` | scores out-of-domain compounds instead of refusing them, marked `OUT OF DOMAIN`. They get their **own table** in the report, never the shortlist |
| `--gate regressor` | uses the thresholded potency model, for when `table3_classification.md` says that won |
| `--jobs N` | processes for standardising the query list |

### Including the out-of-domain compounds

`--score-all` scores everything and gives the out-of-domain compounds a separate
section, with each one's distance beyond the threshold, so you can triage them by
your own chemistry or pick what goes to a structure-based method.

**Their scores have no support, and that is not the same as being weak.** With no
comparable training data the forest routes a query through splits on feature
combinations it has never seen, and the prediction lands wherever the tree
geometry sends it. The number is arbitrary. Rank within that table means nothing,
and a hit rate from it means less.

Report it as a separate table with the caveat, never merged into the shortlist.
Both counts belong in the write-up — a screen that discards what it could not
handle misreports its own coverage.

The report includes a refusal-rate-against-threshold table. The threshold is a
judgement call nobody can make objectively, so showing how the answer moves with
it is stronger than defending one value.

---

## What to expect from the numbers

**A scaffold-split RMSE below ~0.68 log units is a warning, not a triumph.** That
is the agreement between laboratories measuring the same compound (Kalliokoski
2013, *PLoS ONE* 8:e61007). Beating it means predicting better than the data can
be measured, which is leakage.

**Margins over null are small.** Beating a baseline by 0.1–0.2 log units on
held-out chemotypes is real work. Beating it by 1.0 means something is flattering
you.

**Some splits will fail, and that is why more than one runs.** A model that works
under scaffold cross-validation and fails temporally would not have been useful
prospectively — which is the only way it would ever be used.

---

## Repointing at a new target

A new JSON file and the same three commands.

```bash
cp targets/TEMPLATE.json targets/newtarget.json
# edit: name, targets, target_type
python pipeline.py census --config targets/newtarget.json
# read the formats, choose one, put it in the config
python pipeline.py build  --config targets/newtarget.json --repeats 5
python pipeline.py screen --config targets/newtarget.json --input mylist.csv
```

### Starting over, keeping the download

```bash
rm -rf curated features figures models table2*.md table3*.* \
       negative_controls*.md results_table2*.csv \
       screen_results.csv screen_report.md stereo_status*.csv
python pipeline.py build --config targets/mytarget.json --repeats 5
```

`*_subset.db` and `api_cache/` survive, so nothing refetches. Delete the `.db` to
force a fresh download.

---

## If something breaks

**Config won't parse** — the driver prints the broken line with a diagnosis.
Check any config edit with
`python -c "import json; json.load(open('targets/mytarget.json')); print('ok')"`.

**Fetch fails with connection errors** — ChEMBL rate-limits heavy use. The script
backs off, retries, and caches every page it got. Re-run; it resumes.

**`census` reports no records surviving the cascade** — `min_confidence_score` and
`assay_type` are the usual culprits. For ORGANISM targets both must be `null`.

**`build` refuses with "no assay format selected"** — that is deliberate. Run
`census` and choose one.

**Most molecules set zero pharmacophore bits** — this representation is wrong for
your dataset and the model is running on descriptors. Check panel D, then compare
`--blocks pharmacophore` against `--blocks descriptors` to see how much the
signature was contributing.

**`screen` refuses nearly everything** — usually the correct answer rather than a
bug. Check `ad_distance` in `screen_results.csv` against the threshold the report
prints, and read the refusal-rate table. If the library really is unrelated
chemistry, a high refusal rate is the finding.

**Anything else** — the halt names the stage and the command. Run that command on
its own to see the full error.
