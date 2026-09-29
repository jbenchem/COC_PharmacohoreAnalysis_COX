#!/usr/bin/env python3
"""
Derive stereo_status correctly: STRUCTURE first, ChEMBL flag second.

    python stereo_status.py --db cox_subset.db

Why this exists
---------------
ChEMBL's molecule_dictionary.chirality is CURATION METADATA, not a computed
property. -1 ("unknown") means nobody recorded it. It does NOT mean the molecule
has stereocentres whose configuration is unknown. 2 ("achiral") is only set when
a curator asserted it.

So a flat achiral molecule — celecoxib, aspirin, diclofenac, indomethacin —
routinely carries -1 simply because the field was never curated. Deriving
stereo_status from the flag alone therefore files thousands of achiral compounds
as "unknown stereochemistry", which is a statement about ChEMBL's curation
backlog, not about chemistry.

The structure is ground truth for WHETHER stereocentres exist. The flag is
evidence for WHAT THE ASSAY MEASURED. Those are different questions and only the
second one needs the flag.

Two modes
---------
--mode trust   (DEFAULT)
    Trust the flag wherever ChEMBL committed to one; only derive from structure
    where it did not. Measured on the ChEMBL 37 COX set, the committed flags
    agree with the structures 95-98% of the time:
        flag 1 -> 729/768 assigned  (95%)
        flag 2 -> 915/943 achiral   (97%)
        flag 0 -> 423/431 drawn without stereo (98%)
    while flag -1 is an EMPTY FIELD covering 79% of the set, 72% of which is
    simply achiral. So:
        flag  1 -> assigned        KEEP bits
        flag  0 -> racemic         strip
        flag  2 -> achiral         strip (nothing to encode)
        flag -1 -> derive from the structure with RDKit
    Costs 8 compounds of disagreement out of 10,061 and removes a class of
    judgement calls. This is the policy the pipeline uses.

--mode audit
    Derive EVERY compound from the structure and cross-tabulate against the
    flag, ignoring what the flag claims. Slower, and it is what produces the
    curation-log numbers: how many compounds the flag mislabels, how many
    conflicts exist, how far the achiral undercount goes. Run it once, put the
    numbers in the log, then use trust mode for the pipeline.

Decision table (structure-derived branches)
-------------------------------------------
  no stereocentres         -> achiral        (chirality bits irrelevant)
  centres all assigned     -> assigned       (KEEP) if flag 1
                              conflict       (strip) if flag 0
                              flag_disagrees (strip) if flag 2
                              structure_only (strip) if flag -1
  centres unassigned       -> unspecified    (strip)

Only `assigned` earns chirality bits in the fingerprint.

Writes stereo_status.csv (one row per compound, with a keep_chirality_bits
column to join into curation on chembl_id).
"""

import argparse
import csv
import sqlite3
import sys
from collections import Counter

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("ERROR: RDKit not installed. pip install rdkit")

FLAG_NAME = {0: "racemic mixture", 1: "single stereoisomer",
             2: "achiral", -1: "unknown / not reported"}

# stereo_status -> (keep chirality bits?, one-line reason)
POLICY = {
    "achiral":        (False, "no stereocentres — nothing for chirality bits to encode"),
    "assigned":       (True,  "one stereoisomer was tested and the configuration is known"),
    "racemic":        (False, "ChEMBL records that a racemate was tested"),
    "conflict":       (False, "structure asserts one enantiomer, ChEMBL says a racemate was tested"),
    "structure_only": (False, "structure asserts stereo, ChEMBL never recorded what was tested"),
    "flag_disagrees": (False, "structure has assigned centres but ChEMBL calls it achiral"),
    "unspecified":    (False, "stereocentres present, configuration not drawn"),
    "unparseable":    (False, "SMILES did not parse"),
    "no_structure":   (False, "no SMILES in the release"),
}

# Flags ChEMBL committed to, and what trust mode maps them to directly.
TRUSTED = {1: "assigned", 0: "racemic", 2: "achiral"}


def from_structure(smiles, flag):
    """Derive status from the structure. Used for every compound in audit mode,
    and only for flag -1 in trust mode."""
    if not smiles:
        return "no_structure", 0, 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparseable", 0, 0

    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                        useLegacyImplementation=False)
    n = len(centres)
    assigned = sum(1 for _, tag in centres if tag != "?")

    # No centres means the flag is moot, whatever it says.
    if n == 0:
        return "achiral", 0, 0
    if assigned < n:
        return "unspecified", n, assigned

    # All centres assigned. The flag is the only evidence about what was assayed.
    if flag == 1:
        return "assigned", n, assigned
    if flag == 0:
        return "conflict", n, assigned
    if flag == 2:
        return "flag_disagrees", n, assigned
    return "structure_only", n, assigned


def classify(smiles, flag, mode):
    """Return (stereo_status, n_centres, n_assigned, source, note).

    `source` and `note` exist so that NOTHING derived is mistaken for something
    ChEMBL reported. Every row says where its status came from, and every
    derived row says what was observed to justify it. A value we computed and a
    value the source asserted must never be indistinguishable downstream.
    """
    if mode == "trust" and flag in TRUSTED:
        if not smiles:
            return ("no_structure", 0, 0, "chembl_reported",
                    f"ChEMBL chirality={flag}; no SMILES to verify against")
        # Still parse, so n_stereocentres is populated for the curation log —
        # but the STATUS comes from the flag, which is what trust mode means.
        _, n, assigned = from_structure(smiles, flag)
        return (TRUSTED[flag], n, assigned, "chembl_reported",
                f"ChEMBL chirality={flag} taken as reported "
                f"(RDKit sees {n} centre(s), {assigned} assigned)")

    status, n, assigned = from_structure(smiles, flag)
    if flag == -1:
        note = (f"ChEMBL chirality UNPOPULATED (-1); RDKit found {n} "
                f"stereocentre(s), {assigned} assigned -> DERIVED '{status}'")
    else:
        note = (f"audit mode: derived from structure regardless of "
                f"ChEMBL chirality={flag}; {n} centre(s), {assigned} assigned")
    return status, n, assigned, "rdkit_derived", note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cox_subset.db")
    ap.add_argument("--out", default="stereo_status.csv")
    ap.add_argument("--mode", choices=["trust", "audit"], default="trust",
                    help="trust (default): believe flags 0/1/2, derive only "
                         "flag -1. audit: derive everything from the structure "
                         "and cross-tabulate — this is what the curation log needs.")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    rows = con.execute("""
        SELECT md.chembl_id, md.chirality, cs.canonical_smiles
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
    """).fetchall()
    con.close()

    if not rows:
        sys.exit(f"ERROR: no molecules in {args.db}")

    if args.mode == "trust":
        print(f"Classifying {len(rows)} compounds — MODE: trust")
        print("  flags 0/1/2 taken as given; only flag -1 derived from the "
              "structure.\n  Run --mode audit for the curation-log numbers.\n")
    else:
        print(f"Classifying {len(rows)} compounds — MODE: audit")
        print("  every compound derived from the structure, flag ignored "
              "except as evidence\n  of what was assayed.\n")

    status_count = Counter()
    cross = Counter()
    source_count = Counter()
    out = []
    for chembl_id, flag, smiles in rows:
        flag = -1 if flag is None else flag
        status, n_centres, n_assigned, source, note = classify(
            smiles, flag, args.mode)
        status_count[status] += 1
        cross[(flag, status)] += 1
        source_count[source] += 1
        out.append({
            "chembl_id": chembl_id,
            "chembl_chirality": flag,
            "chembl_chirality_meaning": FLAG_NAME.get(flag, f"undocumented ({flag})"),
            "n_stereocentres": n_centres,
            "n_assigned": n_assigned,
            "stereo_status": status,
            # ---- provenance. Never let a derived value pass as a reported one.
            "status_source": source,
            "is_derived": int(source == "rdkit_derived"),
            "derivation_note": note,
            "keep_chirality_bits": int(POLICY[status][0]),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    total = len(rows)

    print("=" * 74)
    print(f"stereo_status — mode: {args.mode}")
    print("=" * 74)
    derived_by_status = Counter(r["stereo_status"] for r in out if r["is_derived"])
    print(f"{'status':<17}{'n':>7}{'%':>7}{'derived*':>10}  {'bits':<6} reason")
    for status, n in status_count.most_common():
        keep, reason = POLICY[status]
        d = derived_by_status.get(status, 0)
        print(f"{status:<17}{n:>7}{100*n/total:>6.1f}%{d:>10}  "
              f"{'KEEP' if keep else 'strip':<6} {reason}")
    print("\n* derived = WE computed this from the structure because ChEMBL left")
    print("  `chirality` unpopulated. It is not a reported value. Every such row")
    print("  carries is_derived=1 and a derivation_note saying what was observed.")

    keep_n = sum(n for s, n in status_count.items() if POLICY[s][0])
    print(f"\nChirality bits carry real information for {keep_n} of {total} "
          f"compounds ({100*keep_n/total:.1f}%).")
    print(f"\nProvenance: {source_count['chembl_reported']} reported by ChEMBL, "
          f"{source_count['rdkit_derived']} derived by us.")

    print("\n" + "=" * 74)
    print("CROSS-TAB — ChEMBL flag vs assigned status")
    if args.mode == "trust":
        print("(flags 0/1/2 map 1:1 by definition here — that is what trust mode")
        print(" does. The interesting block is flag -1.)")
    print("=" * 74)
    for flag in sorted({f for f, _ in cross}, key=lambda f: -sum(
            n for (ff, _), n in cross.items() if ff == flag)):
        sub = {s: n for (f, s), n in cross.items() if f == flag}
        print(f"\nflag {flag:>3} = {FLAG_NAME.get(flag, 'undocumented')} "
              f"({sum(sub.values())} compounds)")
        for s, n in sorted(sub.items(), key=lambda kv: -kv[1]):
            print(f"    {s:<17}{n:>7}")

    # ---- findings for the curation log
    print("\n" + "=" * 74)
    print("FOR THE CURATION LOG")
    print("=" * 74)
    unreported = sum(n for (f, _), n in cross.items() if f == -1)
    print(f"ChEMBL left `chirality` unpopulated for {unreported} of {total} "
          f"compounds ({100*unreported/total:.0f}%).")
    print(f"Resolved from the structures:")
    for s, n in sorted(((s, n) for (f, s), n in cross.items() if f == -1),
                       key=lambda kv: -kv[1]):
        print(f"    {s:<17}{n:>7}")
    ach = cross.get((-1, "achiral"), 0)
    if ach:
        print(f"  -> {ach} of them have NO stereocentres at all. Filing those as")
        print(f"     'unknown stereochemistry' would describe ChEMBL's curation")
        print(f"     backlog, not the chemistry. They are achiral.")

    if args.mode == "audit":
        print(f"\n{status_count.get('conflict', 0)} compounds: structure asserts one "
              f"enantiomer, ChEMBL says a RACEMATE was tested.")
        print("  -> Fingerprinting these with chirality on invents a species that")
        print("     was never assayed, and attaches a mixture's potency to it.")
        print(f"{status_count.get('flag_disagrees', 0)} compounds: structure has "
              f"assigned centres, ChEMBL calls it achiral.")
    else:
        print("\nConflicts between a committed flag and the structure are NOT")
        print("reported in trust mode, by design — the flag wins. Run")
        print("`--mode audit` once and put those counts in the log.")

    # ---- separate file holding ONLY what we derived, so the override is auditable
    derived = [r for r in out if r["is_derived"]]
    if derived:
        dpath = args.out.replace(".csv", "_derived.csv")
        if dpath == args.out:
            dpath = args.out + ".derived.csv"
        with open(dpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(derived[0].keys()))
            w.writeheader()
            w.writerows(derived)
        print(f"\nWrote {dpath} — {len(derived)} compounds whose stereo_status WE")
        print("assigned, each with the observation that justifies it. This file is")
        print("the audit trail: it is the difference between the dataset as ChEMBL")
        print("reports it and the dataset as we are modelling it. Cite it in the")
        print("curation log and keep it under version control.")

    print(f"\nWrote {args.out}. Join into curation on chembl_id; use")
    print("keep_chirality_bits to decide useChirality per compound.")


if __name__ == "__main__":
    main()
