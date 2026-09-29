#!/usr/bin/env python3
"""
W1 census runner.

    python run_census.py --db /path/to/chembl_35.db
    python run_census.py --db /path/to/chembl_35.db --only cox_cascade
    python run_census.py --dsn "postgresql://user@host/chembl_35"

Reads census.sql, runs each query, writes one CSV per query plus a single
markdown report. Does nothing else — no network, no mutation, no hidden
state. Re-running it on the same release reproduces the same outputs.

The schema assertion runs first and stops the whole thing if any column the
queries depend on is missing from your release.
"""

import argparse
import csv
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

SQL_FILE = "census.sql"
OUT_DIR = "census_out"

# Every column the queries in census.sql depend on.
REQUIRED = {
    "activities": ["assay_id", "molregno", "standard_type", "standard_relation",
                   "standard_value", "standard_units", "pchembl_value",
                   "data_validity_comment", "potential_duplicate"],
    "assays": ["assay_id", "tid", "doc_id", "description", "assay_type",
               "confidence_score", "bao_format", "variant_id"],
    "target_dictionary": ["tid", "chembl_id", "pref_name", "organism", "target_type"],
    "molecule_dictionary": ["molregno", "chembl_id", "chirality", "max_phase"],
    "compound_structures": ["molregno", "canonical_smiles"],
    "docs": ["doc_id", "year"],
    "target_components": ["tid", "component_id"],
    "component_sequences": ["component_id", "accession"],
    "variant_sequences": ["variant_id", "mutation", "accession"],
}

# Queries that are inputs to a later step rather than report tables.
NOT_REPORTED = {"cox_smiles_for_stereo_check"}

# Handled outside the main loop.
SKIP = {"release_version"}


def parse_sql(path):
    """Split census.sql on '-- @query: name' markers, preserving order."""
    text = open(path).read()
    parts = re.split(r"^--\s*@query:\s*(\w+)\s*$", text, flags=re.M)
    out = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1].strip()
        body = "\n".join(l for l in body.splitlines()
                         if not l.strip().startswith("--")).strip().rstrip(";")
        if body:
            out.append((parts[i], body))
    return out


def connect(args):
    if args.dsn:
        import psycopg2
        return psycopg2.connect(args.dsn), "postgres"
    if not os.path.exists(args.db):
        sys.exit(f"ERROR: no database at {args.db}")
    import sqlite3
    return sqlite3.connect(args.db), "sqlite"


def run(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def assert_schema(conn, flavour):
    """Stop now if the release renamed or dropped anything we depend on."""
    present = {}
    cur = conn.cursor()
    if flavour == "sqlite":
        for tbl in REQUIRED:
            try:
                cur.execute(f"PRAGMA table_info({tbl})")
                present[tbl] = {r[1].lower() for r in cur.fetchall()}
            except Exception:
                present[tbl] = set()
    else:
        cur.execute("""SELECT LOWER(table_name), LOWER(column_name)
                       FROM information_schema.columns""")
        for t, c in cur.fetchall():
            present.setdefault(t, set()).add(c)

    problems = []
    for tbl, cols in REQUIRED.items():
        have = present.get(tbl, set())
        if not have:
            problems.append(f"  table '{tbl}' not found")
            continue
        missing = [c for c in cols if c.lower() not in have]
        if missing:
            problems.append(f"  {tbl}: missing {', '.join(missing)}")
    return problems


def stereo_crosscheck(rows, cols):
    """ChEMBL's chirality flag vs what RDKit finds in the structure."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None

    i_chir, i_smi = cols.index("chirality"), cols.index("canonical_smiles")
    MEANING = {0: "racemic mixture", 1: "single stereoisomer",
               2: "achiral", -1: "unknown"}
    table = {}
    unparseable = 0

    for r in rows:
        mol = Chem.MolFromSmiles(r[i_smi])
        if mol is None:
            unparseable += 1
            continue
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                            useLegacyImplementation=False)
        if not centres:
            rdkit_says = "no stereocentres"
        elif all(c[1] != "?" for c in centres):
            rdkit_says = "all centres assigned"
        elif any(c[1] != "?" for c in centres):
            rdkit_says = "partially assigned"
        else:
            rdkit_says = "centres present, none assigned"
        key = (MEANING.get(r[i_chir], f"undocumented ({r[i_chir]})"), rdkit_says)
        table[key] = table.get(key, 0) + 1

    return table, unparseable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="chembl.db", help="ChEMBL SQLite file")
    ap.add_argument("--dsn", help="PostgreSQL DSN (overrides --db)")
    ap.add_argument("--only", help="run a single named query")
    ap.add_argument("--skip-schema-check", action="store_true")
    args = ap.parse_args()

    queries = parse_sql(SQL_FILE)
    sql_hash = hashlib.sha256(open(SQL_FILE, "rb").read()).hexdigest()[:12]
    os.makedirs(OUT_DIR, exist_ok=True)

    conn, flavour = connect(args)
    print(f"connected: {flavour}")

    # ---- release version
    release = "UNKNOWN — no version table"
    try:
        _, v = run(conn, "SELECT name, creation_date, comments FROM version")
        if v:
            release = " | ".join(str(x) for x in v[0])
    except Exception:
        pass
    print(f"release:   {release}")

    # ---- schema assertion
    if not args.skip_schema_check:
        problems = assert_schema(conn, flavour)
        if problems:
            print("\nSCHEMA CHECK FAILED — these queries will not run:")
            print("\n".join(problems))
            print("\nFix census.sql for this release, then re-run. "
                  "(--skip-schema-check to override.)")
            sys.exit(1)
        print("schema:    OK, all required columns present")

    # ---- run
    report = [
        "# W1 census report\n",
        f"- **ChEMBL release**: {release}",
        f"- **Run (UTC)**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        f"- **census.sql sha256**: `{sql_hash}`",
        f"- **Engine**: {flavour}\n",
        "> Counts are DISTINCT COMPOUNDS unless a column says `n_records`.\n",
    ]
    stereo_rows = stereo_cols = None
    failures = []

    for name, sql in queries:
        if args.only and name != args.only:
            continue
        if name in SKIP:
            continue
        try:
            cols, rows = run(conn, sql)
        except Exception as e:
            print(f"  {name:<32} FAILED: {str(e)[:70]}")
            report.append(f"\n## {name}\n\n```\nFAILED: {e}\n```\n")
            failures.append((name, str(e)))
            continue

        print(f"  {name:<32} {len(rows):>7} rows")
        with open(f"{OUT_DIR}/{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)

        if name == "cox_smiles_for_stereo_check":
            stereo_rows, stereo_cols = rows, cols

        if name not in NOT_REPORTED:
            report.append(f"\n## {name}\n")
            report.append("| " + " | ".join(cols) + " |")
            report.append("|" + "---|" * len(cols))
            for r in rows[:40]:
                report.append("| " + " | ".join(
                    "" if v is None else str(v)[:60] for v in r) + " |")
            if len(rows) > 40:
                report.append(f"\n*{len(rows) - 40} more rows in "
                              f"`{OUT_DIR}/{name}.csv`*")

    # ---- stereo cross-check
    if stereo_cols is None:
        report.append("\n## stereo cross-check\n\n**NOT RUN** — "
                      "`cox_smiles_for_stereo_check` did not execute.\n")
        failures.append(("stereo_crosscheck", "source query did not run"))
    elif not stereo_rows:
        report.append("\n## stereo cross-check\n\n**NO INPUT ROWS** — the query "
                      "ran and returned nothing. This is NOT 'no disagreements'; "
                      "it means there are no curated COX compounds with SMILES. "
                      "Check the cascade before reading anything else here.\n")
        failures.append(("stereo_crosscheck", "source query returned 0 rows"))
    else:
        res = stereo_crosscheck(stereo_rows, stereo_cols)
        if res is None:
            report.append("\n## stereo cross-check\n\nSkipped — RDKit not installed.\n")
        else:
            table, bad = res
            report.append("\n## stereo cross-check — ChEMBL flag vs RDKit structure\n")
            report.append("| ChEMBL `chirality` | RDKit finds | compounds |")
            report.append("|---|---|---|")
            for (flag, rd), n in sorted(table.items(), key=lambda kv: -kv[1]):
                report.append(f"| {flag} | {rd} | {n} |")
            if bad:
                report.append(f"\n{bad} SMILES failed to parse.")
            report.append(
                "\n**Read this before setting `useChirality`.** Rows where ChEMBL "
                "says *racemic mixture* but RDKit finds *all centres assigned* are "
                "the trap: the structure asserts one enantiomer, the assay measured "
                "a mixture. Turning chirality on invents a species that was never "
                "tested. Count them, then decide — and put the count in the "
                "curation log.\n")

    if failures:
        report.insert(6, "\n> ⚠️ **THIS RUN IS INCOMPLETE.** "
                         f"{len(failures)} step(s) failed: "
                         + ", ".join(f"`{n}`" for n, _ in failures)
                         + ". Do not quote any number from this report until "
                           "they are fixed.\n")

    open("census_report.md", "w").write("\n".join(report))
    conn.close()
    print(f"\nwrote census_report.md and {OUT_DIR}/*.csv")

    if failures:
        print(f"\n{'=' * 60}\nINCOMPLETE RUN — {len(failures)} step(s) failed:")
        for n, e in failures:
            print(f"  {n}: {e[:100]}")
        print("Report written, but marked incomplete. Exiting 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
