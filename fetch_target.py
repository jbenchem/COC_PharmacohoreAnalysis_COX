#!/usr/bin/env python3
"""
Build a local ChEMBL subset for ANY target, over the ChEMBL web API.

    python fetch_target.py --config targets/mytarget.json

Targets are named by UniProt accession in the config — accessions are stable
across ChEMBL releases, ChEMBL target IDs are not.

Output: <name>_subset.db — a SQLite file using ChEMBL's REAL table and column
names, so census.sql and run_census.py run against it UNCHANGED.

Why a subset and not the full release
-------------------------------------
The full ChEMBL SQLite release is tens of GB unpacked. A default GitHub
Codespace has 32 GB of disk. COX-1 + COX-2 is a few thousand compounds —
minutes over the API, ~20 MB on disk. Use this for W1 Parts A and B.
Parts C and D (the AMR census) scan every target in the database and DO
need the full release on a bigger machine. See README.md.

What this is not
----------------
It is not a replacement for the full release, and the subset is NOT a
citable ChEMBL distribution. It records the release version the API
reports, so numbers stay traceable — but for the thesis methods section,
re-run the census against a pinned full release and confirm the counts.

Resumability: every API page is cached under ./api_cache/. Re-running costs
nothing and refetches nothing. Delete that folder to force a fresh pull.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from urllib.parse import urlencode

import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
CACHE = "api_cache"
DB = "cox_subset.db"

HEADERS = {"Accept": "application/json",
           "User-Agent": "COC-census/1.0 (PhD candidature; contact via repo)"}

# Anything the API did not return, recorded so the run cannot claim completeness.
MISSING_LOG = {}


# --------------------------------------------------------------- transport
def get(path, params=None, tries=4):
    """GET one page, with an on-disk cache and polite retries."""
    params = dict(params or {})
    params.setdefault("format", "json")
    url = f"{BASE}/{path}?{urlencode(sorted(params.items()))}"

    os.makedirs(CACHE, exist_ok=True)
    # hashlib, not hash() — Python randomises string hashing per process, so
    # hash() would give a different filename every run and cache nothing.
    key = hashlib.sha256(url.encode()).hexdigest()[:24] + ".json"
    cached = os.path.join(CACHE, key)
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)

    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            if r.status_code == 200:
                data = r.json()
                with open(cached, "w") as f:
                    json.dump(data, f)
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"    HTTP {r.status_code}, retry {attempt}/{tries} "
                      f"in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            sys.exit(f"\nERROR: HTTP {r.status_code} from {url}\n{r.text[:300]}")
        except requests.RequestException as e:
            print(f"    {type(e).__name__}, retry {attempt}/{tries} in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    sys.exit(f"\nERROR: gave up on {url} after {tries} attempts.\n"
             "If this is a network block rather than a server problem, you "
             "need the full release route in README.md instead.")


def paginate(path, params, collection, label):
    """Walk every page of a list endpoint.

    Stops on total_count, NOT on a short page. A short page is not proof of
    end-of-data: if the server caps `limit` below what we asked for, or returns
    a short page mid-stream, a break-on-short-page loop exits early, prints
    "200 / 8000" and returns 2.5% of the data with exit code 0. Every downstream
    count is then quietly wrong. So: read total_count, keep going until we have
    it, and fail loudly if we cannot.
    """
    out, offset, limit, total = [], 0, 1000, None
    while True:
        page = get(path, {**params, "limit": limit, "offset": offset})
        chunk = page.get(collection, [])
        meta = page.get("page_meta", {}) or {}
        if total is None:
            total = meta.get("total_count")
        out.extend(chunk)
        print(f"    {label}: {len(out)}" + (f" / {total}" if total is not None else ""))

        if total is not None and len(out) >= total:
            break
        if not chunk:
            # No progress. Either we are genuinely done, or the server stopped
            # serving. Those are different and only total_count distinguishes them.
            if total is not None and len(out) < total:
                sys.exit(f"\nERROR: {label} stalled at {len(out)} of {total} "
                         f"records — the server returned an empty page before "
                         f"the end.\nThis would silently truncate the census. "
                         f"Re-run (the cache keeps what was fetched); if it "
                         f"stalls again at the same place, reduce `limit` in "
                         f"paginate() and try once more.")
            break
        if total is None and len(chunk) < limit:
            # No total_count to check against. Accept, but say so — a census
            # built on an unverifiable page walk is not a census you can cite.
            print(f"    WARNING: {label} had no total_count in page_meta; "
                  f"stopped on a short page at {len(out)}. Count UNVERIFIED.")
            break
        offset += len(chunk)
        if offset > 400_000:
            sys.exit(f"\nERROR: {label} exceeded 400k records. The filter is "
                     f"almost certainly wrong. Check it rather than raising "
                     f"this cap.")
    return out


def batched(path, collection, id_field, ids, label, chunk=40):
    """Fetch many records by ID, in chunks small enough for a URL.

    Reconciles what came back against what was asked for. Every downstream join
    is an INNER JOIN, so a record that silently fails to arrive removes its
    activities from every count without raising anything.
    """
    ids = sorted(set(i for i in ids if i))
    out = []
    for i in range(0, len(ids), chunk):
        block = ids[i:i + chunk]
        out.extend(paginate(path, {f"{id_field}__in": ",".join(block)},
                            collection, f"{label} {i + len(block)}/{len(ids)}"))

    got = {r.get(id_field) for r in out}
    missing = [i for i in ids if i not in got]
    if missing:
        print(f"\n  ⚠️  {label}: asked for {len(ids)}, got {len(got)}. "
              f"{len(missing)} MISSING.")
        print(f"      first few: {', '.join(missing[:8])}")
        print(f"      Every join downstream is an INNER JOIN, so activities on "
              f"these records would vanish from every count with no error.")
        MISSING_LOG[label] = missing
    return out


# ------------------------------------------------------------------ schema
SCHEMA = """
CREATE TABLE version            (name TEXT, creation_date TEXT, comments TEXT);
CREATE TABLE target_dictionary  (tid INTEGER PRIMARY KEY, chembl_id TEXT,
                                 pref_name TEXT, organism TEXT, target_type TEXT);
CREATE TABLE target_components  (tid INTEGER, component_id INTEGER);
CREATE TABLE component_sequences(component_id INTEGER PRIMARY KEY, accession TEXT);
CREATE TABLE molecule_dictionary(molregno INTEGER PRIMARY KEY, chembl_id TEXT,
                                 chirality INTEGER, max_phase REAL);
CREATE TABLE compound_structures(molregno INTEGER PRIMARY KEY, canonical_smiles TEXT);
CREATE TABLE docs               (doc_id INTEGER PRIMARY KEY, chembl_id TEXT, year INTEGER);
CREATE TABLE variant_sequences  (variant_id INTEGER PRIMARY KEY, mutation TEXT,
                                 accession TEXT);
CREATE TABLE assays             (assay_id INTEGER PRIMARY KEY, chembl_id TEXT,
                                 tid INTEGER, doc_id INTEGER, description TEXT,
                                 assay_type TEXT, confidence_score INTEGER,
                                 bao_format TEXT, variant_id INTEGER);
CREATE TABLE activities         (activity_id INTEGER PRIMARY KEY, assay_id INTEGER,
                                 molregno INTEGER, standard_type TEXT,
                                 standard_relation TEXT, standard_value REAL,
                                 standard_units TEXT, pchembl_value REAL,
                                 data_validity_comment TEXT,
                                 potential_duplicate INTEGER);
CREATE INDEX ix_act_assay ON activities(assay_id);
CREATE INDEX ix_act_mol   ON activities(molregno);
CREATE INDEX ix_assay_tid ON assays(tid);
"""


class Keys:
    """ChEMBL string IDs -> stable integers, because the real schema joins
    on integers and census.sql expects that."""

    def __init__(self):
        self.maps = {}

    def __call__(self, kind, chembl_id):
        if chembl_id is None:
            return None
        m = self.maps.setdefault(kind, {})
        if chembl_id not in m:
            m[chembl_id] = len(m) + 1
        return m[chembl_id]


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the api_cache and refetch everything")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    ACCESSIONS = cfg["targets"]          # {"P35354": "COX-2", ...}
    TARGET_TYPE = cfg.get("target_type", "SINGLE PROTEIN")
    args.out = args.out or f"{cfg['name']}_subset.db"
    if not ACCESSIONS:
        sys.exit("ERROR: config has no `targets`. Add at least one "
                 "\"UNIPROT_ACCESSION\": \"label\" pair.")

    if args.fresh and os.path.isdir(CACHE):
        import shutil
        shutil.rmtree(CACHE)

    print(f"ChEMBL subset builder — {cfg['name']}\n" + "=" * 60)

    # -- release version, so the output is traceable
    status = get("status")
    release = status.get("chembl_db_version", "UNKNOWN")
    print(f"ChEMBL release reported by API: {release}")
    print("Record this in every output. It is what makes the counts citable.\n")

    keys = Keys()
    targets, comp_rows, seq_rows = [], [], []

    # -- 1. resolve the two targets by accession
    print("[1/5] Resolving COX targets by UniProt accession")
    for acc, label in ACCESSIONS.items():
        found = paginate("target",
                         {"target_components__accession": acc,
                          "target_type": TARGET_TYPE},
                         "targets", f"  {acc} {label}")
        if not found:
            sys.exit(f"ERROR: no '{TARGET_TYPE}' target found for accession "
                     f"{acc}.\nCheck it at https://www.uniprot.org/uniprotkb/"
                     f"{acc} and confirm ChEMBL has bioactivity for it.")
        for t in found:
            tid = keys("target", t["target_chembl_id"])
            targets.append((tid, t["target_chembl_id"], t.get("pref_name"),
                            t.get("organism"), t.get("target_type")))
            for c in t.get("target_components", []):
                cacc = c.get("accession")
                cid = keys("component", cacc or f"noacc-{tid}")
                comp_rows.append((tid, cid))
                seq_rows.append((cid, cacc))
        print(f"  {acc} -> {[t['target_chembl_id'] for t in found]}  {label}")

    target_ids = [t[1] for t in targets]

    # -- 2. all activities for those targets
    print(f"\n[2/5] Fetching activities for {len(target_ids)} target(s)")
    acts = []
    for tcid in target_ids:
        acts.extend(paginate("activity", {"target_chembl_id": tcid},
                             "activities", f"  {tcid}"))
    print(f"  {len(acts)} activity records")
    if not acts:
        sys.exit("ERROR: no activities returned. Nothing to build.")

    assay_ids = {a.get("assay_chembl_id") for a in acts}
    mol_ids = {a.get("molecule_chembl_id") for a in acts}
    doc_ids = {a.get("document_chembl_id") for a in acts}
    print(f"  {len(assay_ids)} assays, {len(mol_ids)} compounds, "
          f"{len(doc_ids)} documents")

    # -- 3. assay metadata (confidence_score lives here, not on activity)
    print(f"\n[3/5] Fetching assay metadata ({len(assay_ids)} assays)")
    assays = batched("assay", "assays", "assay_chembl_id", assay_ids, "  assays")

    # -- 4. molecules (chirality + SMILES) and documents (year)
    print(f"\n[4/5] Fetching molecules ({len(mol_ids)}) and documents ({len(doc_ids)})")
    mols = batched("molecule", "molecules", "molecule_chembl_id", mol_ids, "  molecules")
    docs = batched("document", "documents", "document_chembl_id", doc_ids, "  documents")

    # -- 5. write SQLite
    print(f"\n[5/5] Writing {args.out}")
    if os.path.exists(args.out):
        os.remove(args.out)
    con = sqlite3.connect(args.out)
    con.executescript(SCHEMA)

    con.execute("INSERT INTO version VALUES (?,?,?)",
                (release, time.strftime("%Y-%m-%d"),
                 f"SUBSET for {cfg['name']} ({', '.join(ACCESSIONS)}) built "
                 f"from the ChEMBL web API by fetch_target.py. NOT the full "
                 f"release — whole-database queries will return nothing."))

    con.executemany("INSERT INTO target_dictionary VALUES (?,?,?,?,?)", targets)
    con.executemany("INSERT OR IGNORE INTO target_components VALUES (?,?)",
                    set(comp_rows))
    con.executemany("INSERT OR IGNORE INTO component_sequences VALUES (?,?)",
                    set(seq_rows))

    mol_rows, struct_rows = [], []
    for m in mols:
        mr = keys("molecule", m["molecule_chembl_id"])
        chir = as_int(m.get("chirality"))
        mol_rows.append((mr, m["molecule_chembl_id"],
                         -1 if chir is None else chir,
                         as_float(m.get("max_phase"))))
        smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
        struct_rows.append((mr, smi))
    con.executemany("INSERT OR REPLACE INTO molecule_dictionary VALUES (?,?,?,?)",
                    mol_rows)
    con.executemany("INSERT OR REPLACE INTO compound_structures VALUES (?,?)",
                    struct_rows)

    con.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?)",
                    [(keys("doc", d["document_chembl_id"]),
                      d["document_chembl_id"], as_int(d.get("year")))
                     for d in docs])

    # Assays can cite a document the activity rows never mentioned. Those doc_ids
    # must exist in `docs` or cox_year_distribution (an INNER JOIN) silently drops
    # those compounds and biases the temporal-split feasibility call.
    doc_known = {d["document_chembl_id"] for d in docs}
    extra_docs = {a.get("document_chembl_id") for a in assays
                  if a.get("document_chembl_id")} - doc_known
    if extra_docs:
        print(f"  fetching {len(extra_docs)} extra document(s) cited by assays "
              f"but not by any activity")
        docs += batched("document", "documents", "document_chembl_id",
                        extra_docs, "  extra documents")
        con.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?)",
                        [(keys("doc", d["document_chembl_id"]),
                          d["document_chembl_id"], as_int(d.get("year")))
                         for d in docs])

    assay_rows, var_rows = [], []
    for a in assays:
        vid = None
        mut = a.get("variant_sequence") or {}
        if mut.get("mutation"):
            vid = keys("variant", f"{mut.get('accession')}:{mut.get('mutation')}")
            var_rows.append((vid, mut.get("mutation"), mut.get("accession")))
        assay_rows.append((
            keys("assay", a["assay_chembl_id"]), a["assay_chembl_id"],
            keys("target", a.get("target_chembl_id")),
            keys("doc", a.get("document_chembl_id")),
            a.get("description"), a.get("assay_type"),
            as_int(a.get("confidence_score")), a.get("bao_format"), vid))
    con.executemany("INSERT OR REPLACE INTO variant_sequences VALUES (?,?,?)",
                    set(var_rows))
    con.executemany(
        "INSERT OR REPLACE INTO assays VALUES (?,?,?,?,?,?,?,?,?)", assay_rows)

    con.executemany(
        "INSERT INTO activities VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(i + 1,
          keys("assay", a.get("assay_chembl_id")),
          keys("molecule", a.get("molecule_chembl_id")),
          a.get("standard_type"), a.get("standard_relation"),
          as_float(a.get("standard_value")), a.get("standard_units"),
          as_float(a.get("pchembl_value")), a.get("data_validity_comment"),
          # NOT `or 0` — that turns a missing flag into a confident 0 and the
          # IS NULL branch of cascade stage 5 would then never fire, so the
          # subset could not reproduce the full release's stage-5 drop.
          as_int(a.get("potential_duplicate")))
         for i, a in enumerate(acts)])

    con.commit()

    # -- what landed
    print("\n" + "=" * 60)
    print("Row counts written:")
    for t in ("target_dictionary", "assays", "activities",
              "molecule_dictionary", "compound_structures", "docs",
              "variant_sequences"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<24}{n:>8}")

    print("\nCompounds per target (all activity types, unfiltered):")
    for row in con.execute("""
            SELECT td.chembl_id, td.pref_name,
                   COUNT(DISTINCT act.molregno) AS n
            FROM activities act
            JOIN assays a ON act.assay_id = a.assay_id
            JOIN target_dictionary td ON a.tid = td.tid
            GROUP BY td.chembl_id, td.pref_name ORDER BY n DESC"""):
        print(f"  {row[0]:<16}{str(row[1])[:38]:<40}{row[2]:>7}")

    print("\nChirality flags present (this is the stereo decision):")
    MEAN = {0: "racemic mixture", 1: "single stereoisomer",
            2: "achiral", -1: "unknown / not reported"}
    for chir, n in con.execute("""
            SELECT md.chirality, COUNT(DISTINCT md.molregno)
            FROM molecule_dictionary md
            GROUP BY md.chirality ORDER BY 2 DESC"""):
        print(f"  {chir:>3}  {MEAN.get(chir, 'UNDOCUMENTED'):<26}{n:>7}")

    # -- orphan check: anything an INNER JOIN would silently drop
    print("\nIntegrity checks (each must be 0):")
    orphans = {
        "activities with no assay row":
            "SELECT COUNT(*) FROM activities act LEFT JOIN assays a "
            "ON act.assay_id=a.assay_id WHERE a.assay_id IS NULL",
        "activities with no molecule row":
            "SELECT COUNT(*) FROM activities act LEFT JOIN molecule_dictionary md "
            "ON act.molregno=md.molregno WHERE md.molregno IS NULL",
        "assays with no docs row":
            "SELECT COUNT(*) FROM assays a LEFT JOIN docs d ON a.doc_id=d.doc_id "
            "WHERE a.doc_id IS NOT NULL AND d.doc_id IS NULL",
        "molecules with no structure":
            "SELECT COUNT(*) FROM molecule_dictionary md LEFT JOIN "
            "compound_structures cs ON md.molregno=cs.molregno "
            "WHERE cs.canonical_smiles IS NULL",
    }
    bad = 0
    for label, q in orphans.items():
        n = con.execute(q).fetchone()[0]
        if n == 0:
            flag = ""
        elif "no structure" in label:
            # Not fatal and not a count problem: a molecule with no SMILES still
            # has its activities counted. It is only excluded from the RDKit
            # stereo cross-check, which needs a structure to parse.
            flag = "   (not counted in the stereo cross-check; counts unaffected)"
        else:
            flag = "   <-- LOSES DATA IN EVERY COUNT"
            bad += n
        print(f"  {label:<36}{n:>7}{flag}")

    con.close()

    if MISSING_LOG or bad:
        print("\n" + "=" * 60)
        print("SUBSET IS INCOMPLETE. Do not quote counts from it.")
        for label, missing in MISSING_LOG.items():
            print(f"  {label}: {len(missing)} record(s) never returned")
        if bad:
            print(f"  {bad} orphaned activity/assay row(s) — these disappear "
                  f"from every census count via INNER JOIN")
        print("\nRe-run the script. The cache keeps what already arrived, so a "
              "re-run only retries what failed. If the same records are missing "
              "twice, they are genuinely absent from the API and you should note "
              "that in the curation log before using this subset.")
        sys.exit(1)

    print(f"\nAll integrity checks passed. Wrote {args.out}.")


if __name__ == "__main__":
    main()
