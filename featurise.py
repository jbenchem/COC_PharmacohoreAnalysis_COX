#!/usr/bin/env python3
"""
SMILES -> feature matrix, using a 2D PHARMACOPHORE signature.

    python featurise.py --csv curated/cox2_regression.csv

Writes to features/:
    <stem>_fp.npy        uint8   (n, FOLD)   folded Gobbi pharmacophore bits
    <stem>_desc.npy      float32 (n, ~217)   RDKit 2D descriptors
    <stem>_meta.csv              (n, ...)    labels + split columns, NEVER features
    <stem>_columns.json                      every column name, in order
    featurisation_report.md                  what was computed and what is suspect

WHY PHARMACOPHORE RATHER THAN ECFP4
-----------------------------------
An ECFP4 model trained on COX-2 medicinal chemistry refused 88.5% of an
ethnopharmacological library. The gap was measured and it is CHEMOTYPE: the
training set is diaryl heterocycles and sulfonamides, the library is terpenoids,
alkaloids and phenylpropanoids. Different ring systems. Molecular weight and
glycosylation were tested and excluded as explanations.

A substructure fingerprint asks "is this exact atom environment present?", so an
unfamiliar scaffold sets unfamiliar bits and looks like nothing the model knows.
A pharmacophore asks "is there a donor here, an acceptor this far away, a
hydrophobe beyond it?" — which can be true of a terpenoid and a coxib alike.
That abstraction is the reason to try it.

WHAT IS COMPUTED
----------------
`Gobbi_Pharm2D` feature definitions: donors, acceptors, aromatic and
hydrophobic centres, acidic and basic groups. Pairs and triplets of those
features, binned by TOPOLOGICAL distance — bonds along the graph, not
Angstroms. So no conformer is generated and no stereochemistry is required,
which matters: 21% of this library carries unspecified stereocentres and one
compound has 28 of them, 2.7e8 stereoisomers.

The raw signature is 39,972 sparse bits. It is FOLDED to `FOLD` bits by taking
each on-bit index modulo FOLD. Folding creates collisions and the collision
count is reported below; measured fidelity on a small set was exact Tanimoto
0.350 against folded 0.385, which is acceptable and, being applied identically
to every molecule, does not bias a comparison.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
No column selection, no scaling, no variance filtering. It computes everything
and flags what looks wrong. Dropping zero-variance or correlated columns has to
be fitted on the TRAINING FOLD ONLY — doing it here, across the whole dataset,
leaks test information into the feature set. The model script does that inside
each fold.
"""

import argparse
import csv as _csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("ERROR: RDKit not installed. pip install rdkit")

FOLD = 4096          # folded width; overridden by --fold from the config

# Columns that describe the record, not the molecule. These govern splitting,
# aggregation and refusal. Feeding any of them to the model is the assay-artefact
# failure the thesis is about.
META_COLS = [
    "compound_chembl_id", "inchikey", "std_smiles", "murcko_scaffold",
    "pchembl_value", "n_measurements", "pchembl_spread", "spread_flag",
    "bao_format", "assay_format_label", "biological_system", "document_year",
    "max_phase", "stereo_status", "label", "target",
]

_W = {}


def _init():
    _W["factory"] = Gobbi_Pharm2D.factory
    _W["desc_names"] = [n for n, _ in Descriptors._descList]


def _one(smi):
    """Returns (folded bits, raw on-bit count, descriptor list) or None."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        sig = Generate.Gen2DFingerprint(mol, _W["factory"])
        on = list(sig.GetOnBits())
        v = np.zeros(FOLD, dtype=np.uint8)
        for i in on:
            v[i % FOLD] = 1
        d = Descriptors.CalcMolDescriptors(mol)
        return (v.tobytes(), len(on), [d.get(n, np.nan) for n in _W["desc_names"]])
    except Exception:
        return None


def main():
    global FOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles-col", default="std_smiles")
    ap.add_argument("--outdir", default="features")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0,
                    help="folded signature width. 0 keeps the module default "
                         "(%d). Passed by pipeline.py from "
                         "features.pharmacophore.folded_bits." % FOLD)
    args = ap.parse_args()
    if args.fold:
        if args.fold < 256:
            sys.exit(f"ERROR: --fold {args.fold} is too small; collisions would "
                     f"dominate. Use 1024 or more.")
        FOLD = args.fold

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    with open(args.csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        sys.exit(f"ERROR: {args.csv} has no rows")
    if args.smiles_col not in rows[0]:
        sys.exit(f"ERROR: no column {args.smiles_col!r}. "
                 f"Found: {', '.join(rows[0].keys())}")
    print(f"{stem}: {len(rows)} rows")

    desc_names = [n for n, _ in Descriptors._descList]
    raw_bits = Gobbi_Pharm2D.factory.GetSigSize()
    jobs = args.jobs or (os.cpu_count() or 1)
    smis = [r[args.smiles_col] for r in rows]

    t0 = time.time()
    print(f"  Gobbi 2D pharmacophore signature: {raw_bits} raw bits, "
          f"folded to {FOLD}")
    if jobs > 1:
        with Pool(jobs, initializer=_init) as pool:
            res = []
            for i, r in enumerate(pool.imap(_one, smis, chunksize=64), 1):
                res.append(r)
                if i % 1000 == 0:
                    print(f"  {i}/{len(smis)}")
    else:
        _init()
        res = [_one(s) for s in smis]
    print(f"  featurised in {time.time() - t0:.0f}s")

    keep_meta, fps, raws, descs, failed = [], [], [], [], []
    for r, out in zip(rows, res):
        if out is None:
            failed.append(r.get("compound_chembl_id", "?"))
            continue
        fps.append(np.frombuffer(out[0], dtype=np.uint8))
        raws.append(out[1])
        descs.append(out[2])
        keep_meta.append(r)
    if not keep_meta:
        sys.exit("ERROR: nothing featurised. Check the SMILES column.")

    X_fp = np.vstack(fps).astype(np.uint8)
    raws = np.array(raws)

    # Descriptors in float64, CLIPPED before the float32 cast. `Ipc` is a
    # product of graph-matrix eigenvalues and reaches ~1e40 on larger
    # molecules — past float32's 3.4e38 ceiling, so a naive cast turns it into
    # `inf` and every downstream finite-check then blames the wrong column.
    X64 = np.asarray(descs, dtype=np.float64)
    F32_MAX = np.finfo(np.float32).max
    overflow_cols = sorted({desc_names[j] for j in range(X64.shape[1])
                            if (np.abs(X64[:, j]) > F32_MAX).any()})
    n_overflow = int((np.abs(X64) > F32_MAX).sum())
    if n_overflow:
        print(f"  {n_overflow} descriptor value(s) exceed float32 range and "
              f"were clipped: {', '.join(overflow_cols)}")
    X_de = np.clip(X64, -F32_MAX, F32_MAX).astype(np.float32)

    if failed:
        print(f"  WARNING: {len(failed)} compound(s) failed to featurise: "
              f"{', '.join(failed[:5])}")
        print("  These were WRITTEN by RDKit during standardisation and cannot be")
        print("  READ BACK, or the signature generator rejected them. A round-trip")
        print("  failure means the stored structure is not what was standardised.")
        with open("featurisation_failures.txt", "w") as fh:
            fh.write("\n".join(failed) + "\n")

    # ---- diagnostics, reported not applied
    set_bits = X_fp.sum(axis=1)
    collisions = int((raws - set_bits).sum())
    empty = int((set_bits == 0).sum())
    dead = int((X_fp.sum(axis=0) == 0).sum())
    nonfinite = [desc_names[j] for j in range(X_de.shape[1])
                 if not np.isfinite(X_de[:, j]).all()]
    finite_cols = [j for j in range(X_de.shape[1])
                   if np.isfinite(X_de[:, j]).all()]
    zerovar = [desc_names[j] for j in finite_cols if np.ptp(X_de[:, j]) == 0]

    print(f"\nX width {FOLD + len(desc_names)}  "
          f"({FOLD} pharmacophore + {len(desc_names)} descriptor)")
    print(f"pharmacophore density {set_bits.mean():.1f} bits/molecule "
          f"(raw {raws.mean():.1f}), {dead} dead bits, "
          f"{collisions} fold collisions")
    print(f"descriptors: {len(nonfinite)} non-finite, {len(zerovar)} zero-variance")
    if empty:
        print(f"⚠️  {empty} molecule(s) set ZERO pharmacophore bits. The "
              f"signature encodes nothing for them — they carry no information "
              f"beyond their descriptors. Listed in the report.")

    np.save(f"{args.outdir}/{stem}_fp.npy", X_fp)
    np.save(f"{args.outdir}/{stem}_desc.npy", X_de)
    with open(f"{args.outdir}/{stem}_meta.csv", "w", newline="") as f:
        cols = [c for c in META_COLS if c in keep_meta[0]]
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(keep_meta)

    with open(f"{args.outdir}/{stem}_columns.json", "w") as f:
        json.dump({
            "fingerprint": {
                "names": [f"ph2d_{i}" for i in range(FOLD)],
                "kind": "gobbi_pharm2d_topological",
                "raw_bits": raw_bits, "folded_bits": FOLD,
                "min_points": Gobbi_Pharm2D.factory.minPointCount,
                "max_points": Gobbi_Pharm2D.factory.maxPointCount,
                "needs_conformer": False, "needs_stereochemistry": False,
                "dtype": "uint8",
                "note": "pairs and triplets of pharmacophore features binned by "
                        "topological distance, folded modulo FOLD; hashed, not "
                        "interpretable bit by bit",
            },
            "descriptors": {"names": desc_names, "dtype": "float32"},
            "total_width": FOLD + len(desc_names),
            "metadata_never_in_X": [c for c in META_COLS if c in keep_meta[0]],
            "suspect": {"non_finite": nonfinite, "zero_variance": zerovar,
                        "molecules_with_no_bits": empty},
        }, f, indent=2)

    report = f"""# Featurisation report — {stem}

**2D pharmacophore signature (Gobbi), topological distances.** No conformers, no
stereochemistry required.

| | |
|---|---|
| rows in | {len(rows)} |
| rows featurised | {len(keep_meta)} |
| failed | {len(failed)} |
| | *(round-trip failures: RDKit wrote them, RDKit cannot re-read them. Listed in `featurisation_failures.txt`.)* |
| **X width** | **{FOLD + len(desc_names)}** ({FOLD} pharmacophore + {len(desc_names)} descriptor) |

## Block A — Gobbi 2D pharmacophore

| | |
|---|---|
| raw signature width | {raw_bits} |
| folded to | {FOLD} |
| feature points per term | {Gobbi_Pharm2D.factory.minPointCount}–{Gobbi_Pharm2D.factory.maxPointCount} |
| mean raw on-bits | {raws.mean():.1f} |
| mean folded on-bits | {set_bits.mean():.1f} |
| fold collisions (total) | {collisions} |
| dead bits (never set) | {dead} of {FOLD} |
| molecules with **zero** bits | {empty} |

Folding collides distinct pharmacophore terms onto one column. The collision
count above is the cost. It is applied identically to every molecule, so it does
not bias a comparison between representations, but it does mean a single column
is not interpretable as one pharmacophore feature.

**Molecules setting zero bits carry no pharmacophore information at all.** They
are usually small or purely hydrocarbon — biphenyl, for instance, has two
aromatic features and no donors or acceptors, and scores nothing. If that count
is large, this representation is the wrong one for this dataset and the model
will be running on descriptors alone.

## Block B — RDKit 2D descriptors

| | |
|---|---|
| count | {len(desc_names)} |
| non-finite columns | {len(nonfinite)} |
| zero-variance columns | {len(zerovar)} |
| clipped for float32 overflow | {n_overflow} value(s){(' in ' + ', '.join(overflow_cols)) if overflow_cols else ''} |

Non-finite: {', '.join(nonfinite) if nonfinite else 'none'}

Zero-variance: {', '.join(zerovar[:20]) + (' …' if len(zerovar) > 20 else '') if zerovar else 'none'}

Reported, not dropped. Imputation and variance filtering happen inside each
training fold, because doing them here would fit them on data the model is
supposed to be tested against.
"""
    with open("featurisation_report.md", "w") as f:
        f.write(report)
    print(f"\nwrote {args.outdir}/{stem}_{{fp,desc}}.npy, _meta.csv, _columns.json")
    print("wrote featurisation_report.md")


if __name__ == "__main__":
    main()
