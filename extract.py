#!/usr/bin/env python3
"""
Build the curated dataset the model trains on, for any target.

    python extract.py --config targets/mytarget.json

Produces, per isoform:
    cox1_regression.csv / cox2_regression.csv    relation '=' only, one row per compound
    cox1_classification.csv / cox2_classification.csv   '=' plus '>' as inactives
    curation_log.md                              Table 1, all 9 stages, both isoforms

Nothing here is a model. This is the dataset and its audit trail.

WHAT THIS DOES, AND WHY EACH STEP EXISTS
----------------------------------------
1. Cascade          the 7 SQL stages, re-run in Python so the log is one artifact
2. STANDARDISATION  salt strip -> charge normalise -> canonical tautomer.
                    Stage 8. Diclofenac and diclofenac sodium are the same drug
                    and ChEMBL holds both; unstandardised their ECFP4 Tanimoto is
                    0.73 and the model sees two compounds with two labels. Every
                    dedup and every scaffold-split guarantee depends on this
                    collapse happening FIRST.
3. DEDUPLICATION    stage 9. Median pChEMBL per (compound, bao_format). Spread
                    > 1 log unit is flagged, not silently averaged — a compound
                    measured at 6.0 and 8.5 in the same format is a curation
                    problem, not a data point.
4. Metadata         bao_format (the format-held-out stratum), stereo_status,
                    murcko_scaffold (the scaffold split), document year (the
                    temporal split), censoring flag.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No fingerprints, no descriptors, no model. Featurisation is W3 and reads these
CSVs. Keeping extraction and featurisation apart means a featurisation change
does not re-run extraction, and the dataset stays one auditable thing.

MEASURED DECISIONS BAKED IN (see handover Section 0c)
-----------------------------------------------------
* IC50 only. Ki (27/3 compounds) and Kd (8/6) are rounding errors here.
* AC50 EXCLUDED, explicitly, not by accident. 136 COX-2 / 221 COX-1 compounds,
  usually HTS-derived, a different measurement paradigm. Recorded in the log.
* bao_format is the format stratum. The description regex left 45-52%
  unclassified; bao_format gives 5-7 clean strata.
* '>' records are the INACTIVE CLASS, not garbage. COX-1 is 34% censored, COX-2
  14%. Dropping them would leave a training set enriched for potent compounds by
  construction and an EF1% computed against almost no true negatives.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict, Counter
from statistics import median

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("ERROR: RDKit not installed. pip install rdkit")

import sqlite3

SPREAD_FLAG = 1.0     # log units; above this, flag the group rather than average it

# Both set from the config in main(); these are only fallbacks.
ACTIVE_CUTOFF = 6.0
STD_TYPES = ["IC50"]

_unch = rdMolStandardize.Uncharger()
_taut = rdMolStandardize.TautomerEnumerator()


# ------------------------------------------------------------ standardisation
def standardise(smiles):
    """Stage 8. Fixed order, same code path at training and prediction time.

    Returns (standardised_smiles, inchikey, murcko_scaffold) or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)   # strips salts / solvates
        mol = _unch.uncharge(mol)                    # charge normalise
        mol = _taut.Canonicalize(mol)                # canonical tautomer
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return (Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol), scaffold)


# -------------------------------------------------------------------- extract
BASE_SQL = """
WITH cox_targets AS (
    SELECT DISTINCT td.tid, cseq.accession
    FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession = ?
      AND td.target_type = ?
)
SELECT md.chembl_id          AS compound_chembl_id,
       cs.canonical_smiles   AS raw_smiles,
       md.chirality          AS chembl_chirality,
       md.max_phase          AS max_phase,
       act.standard_type     AS standard_type,
       act.standard_relation AS standard_relation,
       act.standard_value    AS standard_value,
       act.standard_units    AS standard_units,
       act.pchembl_value     AS pchembl_value,
       act.data_validity_comment AS validity,
       act.potential_duplicate   AS dup,
       a.chembl_id           AS assay_chembl_id,
       a.bao_format          AS bao_format,
       a.description         AS assay_description,
       a.assay_type          AS assay_type,
       a.confidence_score    AS confidence_score,
       d.year                AS document_year,
       d.chembl_id           AS document_chembl_id
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN cox_targets t          ON a.tid = t.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
LEFT JOIN compound_structures cs ON act.molregno = cs.molregno
LEFT JOIN docs d            ON a.doc_id = d.doc_id
"""


def format_label(description):
    """Human-readable label for Fig 2 ONLY. bao_format is the real stratum.
    cell-free is tested before the bare 'cell' catch, or '%cell%' would file a
    cell-free biochemical assay as cell-based."""
    d = (description or "").lower()
    if "whole blood" in d:
        return "whole blood"
    if "cell-free" in d or "cell free" in d or "acellular" in d:
        return "cell-free / biochemical"
    if "recombinant" in d:
        return "recombinant enzyme"
    if "microsom" in d:
        return "microsomal"
    if "platelet" in d:
        return "platelet"
    if "cell" in d:
        return "cell-based"
    return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--db")
    ap.add_argument("--stereo", default="stereo_status.csv",
                    help="output of stereo_status.py; joined on chembl_id if present")
    ap.add_argument("--outdir", default="curated")
    args = ap.parse_args()

    import json
    cfg = json.load(open(args.config))
    ACCESSION = cfg["targets"]
    casc = cfg.get("cascade", {})
    global ACTIVE_CUTOFF, STD_TYPES
    ACTIVE_CUTOFF = cfg.get("label", {}).get("active_cutoff", 6.0)
    STD_TYPES = cfg.get("endpoint", {}).get("standard_types", ["IC50"])
    MIN_CONF = casc.get("min_confidence_score", 8)
    ASSAY_TYPE = casc.get("assay_type", "B")
    args.db = args.db or f"{cfg['name']}_subset.db"
    if not os.path.exists(args.db):
        sys.exit(f"ERROR: no database at {args.db}. Run fetch_target.py first.")

    os.makedirs(args.outdir, exist_ok=True)
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    release = "UNKNOWN"
    try:
        release = con.execute("SELECT name FROM version").fetchone()[0]
    except Exception:
        pass

    # optional stereo join
    stereo = {}
    if os.path.exists(args.stereo):
        with open(args.stereo) as f:
            for r in csv.DictReader(f):
                stereo[r["chembl_id"]] = r
        print(f"joined stereo_status for {len(stereo)} compounds")
    else:
        print(f"NOTE: {args.stereo} not found — stereo_status will be blank. "
              f"Run stereo_status.py first.")

    log = [f"# Curation log — COX IC50 dataset\n",
           f"- **ChEMBL release**: {release}",
           f"- **Standardisation**: RDKit Cleanup -> FragmentParent -> Uncharger "
           f"-> TautomerEnumerator.Canonicalize (fixed order)",
           f"- **Deduplication**: median pChEMBL per (compound, bao_format); "
           f"spread > {SPREAD_FLAG} log unit flagged",
           f"- **standard_type**: IC50 ONLY. Ki and Kd are <30 compounds each on "
           f"this target. **AC50 excluded deliberately** — usually HTS-derived, a "
           f"different measurement paradigm; counted below, not used.",
           f"- **Censoring**: '=' for regression; '>' retained as the INACTIVE "
           f"class for classification and EF1%.\n"]

    summary = {}

    for accession, iso in ACCESSION.items():
        rows = [dict(r) for r in con.execute(
            BASE_SQL, (accession, cfg.get("target_type", "SINGLE PROTEIN")))]
        if not rows:
            sys.exit(f"ERROR: no rows for {accession}. Wrong database?")

        def ncomp(rs):
            return len({r["compound_chembl_id"] for r in rs})

        stages = [("0. all activity records", rows)]

        # AC50 counted TWO ways. The raw count answers "how much AC50 exists";
        # the cascaded count answers "what would excluding it actually cost",
        # which is the only one comparable to every other number in the log.
        # What ELSE is in this target's data that the standard_type filter
        # discards? Reported per type so repointing never silently drops a
        # whole endpoint you did not know existed.
        other_types = Counter(r["standard_type"] for r in rows
                              if r["standard_type"] not in STD_TYPES
                              and r["standard_type"])
        excluded_report = "; ".join(
            f"{t} ({ncomp([r for r in rows if r['standard_type'] == t])})"
            for t, _ in other_types.most_common(8)) or "none"
        ac50_raw = ncomp([r for r in rows
                          if r["standard_type"] not in STD_TYPES])
        ac50_cascaded = ncomp([
            r for r in rows
            if r["standard_type"] not in STD_TYPES
            and r["standard_relation"] == "="
            and r["pchembl_value"] is not None
            and r["validity"] is None
            and not r["dup"]
            and (r["confidence_score"] or 0) >= 8
            and (not ASSAY_TYPE or r["assay_type"] == ASSAY_TYPE)])

        s = [r for r in rows if r["standard_type"] in STD_TYPES]
        stages.append((f"1. + standard_type in {STD_TYPES}", s))
        s_all_rel = s                                    # keep for the inactives
        s = [r for r in s if r["standard_relation"] == "="]
        stages.append(("2. + relation = '='", s))
        s = [r for r in s if r["pchembl_value"] is not None]
        stages.append(("3. + pchembl_value present", s))
        s = [r for r in s if r["validity"] is None]
        stages.append(("4. + no validity comment", s))
        s = [r for r in s if not r["dup"]]
        stages.append(("5. + not potential duplicate", s))
        if MIN_CONF is not None:
            s = [r for r in s if (r["confidence_score"] or 0) >= MIN_CONF]
        stages.append((f"6. + confidence_score >= {MIN_CONF}", s))
        if ASSAY_TYPE:
            s = [r for r in s if r["assay_type"] == ASSAY_TYPE]
        stages.append((f"7. + assay_type = '{ASSAY_TYPE}'", s))

        # ---- stage 8: standardisation
        cache, std_rows, failed = {}, [], 0
        for r in s:
            smi = r["raw_smiles"]
            if not smi:
                failed += 1
                continue
            if smi not in cache:
                cache[smi] = standardise(smi)
            res = cache[smi]
            if res is None:
                failed += 1
                continue
            r = dict(r)
            r["std_smiles"], r["inchikey"], r["murcko_scaffold"] = res
            std_rows.append(r)
        stages.append(("8. + standardised (RDKit)", std_rows))

        # Salt-form collapse: distinct chembl_ids sharing one InChIKey.
        by_key = defaultdict(set)
        for r in std_rows:
            by_key[r["inchikey"]].add(r["compound_chembl_id"])
        collapsed = sum(len(v) - 1 for v in by_key.values() if len(v) > 1)

        # ---- stage 9: dedup to one value per (compound, bao_format)
        groups = defaultdict(list)
        for r in std_rows:
            groups[(r["inchikey"], r["bao_format"])].append(r)

        reg, flagged = [], 0
        for (key, bao), rs in groups.items():
            vals = [float(r["pchembl_value"]) for r in rs]
            spread = max(vals) - min(vals)
            if spread > SPREAD_FLAG:
                flagged += 1
            first = rs[0]
            st = stereo.get(first["compound_chembl_id"], {})
            reg.append({
                "compound_chembl_id": first["compound_chembl_id"],
                "inchikey": key,
                "std_smiles": first["std_smiles"],
                "murcko_scaffold": first["murcko_scaffold"],
                "pchembl_value": round(median(vals), 3),
                "n_measurements": len(vals),
                "pchembl_spread": round(spread, 3),
                "spread_flag": int(spread > SPREAD_FLAG),
                "endpoint_class": "biochemical",
                "unit_basis": "molar",
                "censored": "none",
                "bao_format": bao,
                "assay_format_label": format_label(first["assay_description"]),
                "biological_system": f"{iso} ({accession})",
                "document_year": first["document_year"],
                "max_phase": first["max_phase"],
                "chembl_chirality": first["chembl_chirality"],
                "stereo_status": st.get("stereo_status", ""),
                "stereo_is_derived": st.get("is_derived", ""),
                "keep_chirality_bits": st.get("keep_chirality_bits", ""),
            })
        stages.append((f"9. + deduplicated (compound x bao_format)", None))

        # ---- inactives from '>' records, same standardisation
        inact_src = [r for r in s_all_rel
                     if r["standard_relation"] == ">"
                     and r["standard_value"] is not None
                     and r["validity"] is None
                     and not r["dup"]
                     and (MIN_CONF is None or (r["confidence_score"] or 0) >= MIN_CONF)
                     and (not ASSAY_TYPE or r["assay_type"] == ASSAY_TYPE)]
        seen = {r["inchikey"] for r in std_rows}
        inact = {}
        for r in inact_src:
            smi = r["raw_smiles"]
            if not smi:
                continue
            if smi not in cache:
                cache[smi] = standardise(smi)
            res = cache[smi]
            if res is None:
                continue
            std_smi, key, scaf = res
            p = r["pchembl_value"]
            inact.setdefault(key, {
                "compound_chembl_id": r["compound_chembl_id"],
                "inchikey": key,
                "std_smiles": std_smi,
                "murcko_scaffold": scaf,
                "pchembl_value": float(p) if p is not None else "",
                "censored": "right",
                "bao_format": r["bao_format"],
                "assay_format_label": format_label(r["assay_description"]),
                "biological_system": f"{iso} ({accession})",
                "document_year": r["document_year"],
                "also_has_point_value": int(key in seen),
                "label": "inactive",
            })

        # ---- write
        lo = iso.replace("-", "").lower()
        def dump(name, recs):
            path = os.path.join(args.outdir, name)
            if not recs:
                return path, 0
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)
            return path, len(recs)

        p_reg, n_reg = dump(f"{lo}_regression.csv", reg)

        # Classification set: point-value rows plus '>' rows for compounds that
        # have no point value. One explicit field list so both sources line up —
        # mismatched keys here would be a silent column shift in the CSV.
        CLS_FIELDS = ["compound_chembl_id", "inchikey", "std_smiles",
                      "murcko_scaffold", "pchembl_value", "censored",
                      "bao_format", "assay_format_label", "biological_system",
                      "document_year", "label"]

        def cls_row(r, label, censored):
            row = {k: r.get(k, "") for k in CLS_FIELDS}
            row["label"] = label
            row["censored"] = censored
            return row

        cls = [cls_row(r, "active" if float(r["pchembl_value"]) >= ACTIVE_CUTOFF
                       else "weak", "none") for r in reg]
        cls += [cls_row(r, "inactive", "right") for r in inact.values()
                if not r["also_has_point_value"]]
        p_cls, n_cls = dump(f"{lo}_classification.csv", cls)

        uniq_compounds = len({r["inchikey"] for r in reg})
        summary[iso] = dict(
            stages=stages, reg_rows=n_reg, uniq=uniq_compounds,
            flagged=flagged, failed=failed, collapsed=collapsed,
            inactives=len(inact), cls_rows=n_cls,
            ac50_raw=ac50_raw, ac50_cascaded=ac50_cascaded,
            excluded_report=excluded_report,
            scaffolds=len({r["murcko_scaffold"] for r in reg}),
            formats=len({r["bao_format"] for r in reg}),
        )

        print(f"\n{iso}: {n_reg} regression rows, {uniq_compounds} unique compounds, "
              f"{len(inact)} inactives, {n_cls} classification rows")
        print(f"  -> {p_reg}\n  -> {p_cls}")

    con.close()

    # ---------------------------------------------------------------- the log
    log.append("\n## Table 1 — curation cascade (distinct compounds)\n")
    log.append("| stage | " + " | ".join(ACCESSION.values()) + " |")
    log.append("|---|" + "---|" * len(ACCESSION))
    first = next(iter(summary))
    names = [n for n, _ in summary[first]["stages"]]
    for i, name in enumerate(names):
        cells = []
        for iso in ACCESSION.values():
            _, rs = summary[iso]["stages"][i]
            if rs is None:
                cells.append(f"{summary[iso]['uniq']}")
            else:
                cells.append(str(len({r["compound_chembl_id"] for r in rs})))
        log.append(f"| {name} | " + " | ".join(cells) + " |")

    log.append("\n## Standardisation and deduplication detail\n")
    log.append("| | " + " | ".join(ACCESSION.values()) + " |")
    log.append("|---|" + "---|" * len(ACCESSION))
    for label, key in [
        ("SMILES that failed to standardise", "failed"),
        ("distinct ChEMBL IDs collapsed onto a shared InChIKey (salt forms)", "collapsed"),
        ("(compound x format) groups with spread > 1 log unit — FLAGGED", "flagged"),
        ("unique compounds after dedup", "uniq"),
        ("regression rows (compound x bao_format)", "reg_rows"),
        ("distinct Murcko scaffolds", "scaffolds"),
        ("distinct bao_format strata", "formats"),
        ("right-censored ('>') compounds available as inactives", "inactives"),
        ("classification rows (actives + inactives)", "cls_rows"),
        ("compounds excluded by the standard_type filter — raw", "ac50_raw"),
        ("compounds excluded by standard_type — **through the same cascade**",
         "ac50_cascaded"),
        ("excluded endpoint types present in this target's data",
         "excluded_report"),
    ]:
        log.append(f"| {label} | " +
                   " | ".join(str(summary[i][key]) for i in ACCESSION.values()) + " |")

    log.append("""
## Decisions recorded here because a filter should never decide silently

**Endpoint types excluded.** Two counts are given because only one is
comparable. The raw count is every compound carrying an excluded endpoint type.
The **cascaded** count is what would survive the identical filters, and that is
what the exclusion actually costs — quote that one. The row above it names which
types were dropped and how many compounds each holds, so repointing at a new
target cannot silently discard a whole endpoint you did not know was there.

**Ki and Kd excluded.** Under 30 compounds each on this target. Pooling would
have been wrong and is also moot.

**Spread-flagged groups are kept, with the flag.** A compound measured at 6.0 and
8.5 in the same assay format is a curation problem. The median is used and
`spread_flag=1` marks it, so the rows can be excluded in a sensitivity check
rather than being quietly averaged into the training set.

**One row per (compound, bao_format), not per compound.** A compound measured in
two formats appears twice, with two labels. That is deliberate: those
cross-format compounds are what make the format-transfer penalty directly
measurable rather than inferred. Any model must split on compound
(`inchikey`), never on row, or the same molecule lands on both sides.

**`>` records are inactives, not discarded data.** Kept in the classification
set only where the compound has no point value, so an active is never relabelled
inactive. `also_has_point_value` recorded the overlap before filtering.

**Stereo status carries provenance.** `stereo_is_derived=1` means we computed the
status because ChEMBL left `chirality` unpopulated. See `stereo_status_derived.csv`.
""")

    with open("curation_log.md", "w") as f:
        f.write("\n".join(log))
    print("\nwrote curation_log.md — this is Table 1 plus the audit trail")
    print(f"wrote {args.outdir}/*.csv")
    print("\nNEXT: W3 featurisation reads these CSVs. Split on `inchikey`, "
          "stratify on `bao_format`.")


if __name__ == "__main__":
    main()
