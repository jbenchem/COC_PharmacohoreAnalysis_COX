#!/usr/bin/env python3
"""
Screen an external compound list. Two gates, then a prediction.

    python screen.py --config targets/cox2.json --input cmfoa.csv \\
        --smiles-col SMILES --id-col ID --exclude-known

    # with prior annotations, for the enrichment test:
    python screen.py --config targets/cox2.json --input cmfoa.csv \\
        --smiles-col SMILES --id-col ID \\
        --annotation-col "Potential Traditional Uses" \\
        --annotation-positive anti-inflam

THE DECISION ORDER, AND WHY IT IS THIS WAY
------------------------------------------
    1. PARSE         the SMILES, then standardise it through the SAME cascade
                     the training set went through. A query standardised
                     differently from the training data is compared against
                     something it does not match.

    2. DOMAIN GATE   1 - mean Tanimoto to the 5 nearest training compounds.
                     Beyond the threshold, REFUSE. No score is produced, because
                     a number the model has no basis for is worse than silence —
                     it looks like an answer. `--score-all` overrides this and
                     scores everything, with out-of-domain rows marked.

    3. CLASSIFY      active / not. Trained WITH the measured inactives, which the
                     potency model never sees.

    4. PREDICT       pIC50, but ONLY for compounds the classifier called active.
                     Potency for something predicted inactive is a number nobody
                     should act on.

Every compound gets a row and a reason, including the refused ones. A screen
that silently drops what it could not handle has told you the wrong thing about
its own coverage.

THE MODEL IS SAVED
------------------
The first run fits both models, calibrates the domain threshold, and writes the
whole fitted state to `models/`. Later runs load it, so screening a new list is
seconds of setup rather than minutes.

The cache key is a hash of the config fields that matter, the feature files'
size and modification time, and the featurisation parameters. Change any of
them — rebuild the dataset, edit `drop_descriptors`, switch `--blocks` — and
the key changes and the model is refitted. `--refit` forces it.

That matters more than it sounds: a stale model silently scoring against last
week's dataset is the kind of error that survives all the way into a figure.

WHAT THIS CANNOT DO
-------------------
It produces a RANKED SHORTLIST, not evidence of activity. Nothing here has been
tested. The honest claim is "these are the compounds most worth an experiment",
never "these work".
"""

import argparse
import csv as _csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("ERROR: RDKit not installed. pip install rdkit")

try:
    import joblib
except ImportError:
    sys.exit("ERROR: joblib not installed. pip install joblib")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import feature_blocks as FB

AD_K = 5              # nearest training neighbours the domain distance averages
CACHE_VERSION = "3"   # bump when the fitted-state layout changes

_unch = rdMolStandardize.Uncharger()
_taut = rdMolStandardize.TautomerEnumerator()


def standardise(smiles):
    """Identical cascade to extract.py. Must stay identical."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _unch.uncharge(mol)
        mol = _taut.Canonicalize(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


# --------------------------------------------------------- parallel worker
# Module level, with globals set by an initialiser, because multiprocessing
# pickles the function by name and re-sending the descriptor list with every
# task would cost more than the work.
_W = {}


def _init_worker(desc_names, fold):
    _W["desc_names"] = desc_names
    _W["fold"] = fold
    _W["factory"] = Gobbi_Pharm2D.factory


def _featurise_one(smiles):
    """SMILES -> primitives. Returns picklable types only; an RDKit mol does
    not survive the trip between processes reliably."""
    smi = (smiles or "").strip()
    if not smi:
        return None
    mol = standardise(smi)
    if mol is None:
        return None
    try:
        fold = _W["fold"]
        fp = np.zeros(fold, dtype=np.uint8)
        for i in Generate.Gen2DFingerprint(mol, _W["factory"]).GetOnBits():
            fp[i % fold] = 1
        d = Descriptors.CalcMolDescriptors(mol)
        desc = [d.get(n, np.nan) for n in _W["desc_names"]]
        return (Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol),
                fp.tobytes(), desc)
    except Exception:
        return None


# ------------------------------------------------------------- the cache key
def _stat(path):
    try:
        s = os.stat(path)
        return f"{os.path.basename(path)}:{s.st_size}:{int(s.st_mtime)}"
    except OSError:
        return f"{os.path.basename(path)}:missing"


def cache_key(cfg, stem, featdir, blocks, seed, ad_pct):
    """Everything that would change the fitted model or the threshold.

    Deliberately includes the feature files' mtime and size. Rebuilding the
    dataset must invalidate the model, and comparing contents would mean
    reading 60 MB to decide whether to read 60 MB.
    """
    parts = [
        CACHE_VERSION, stem, blocks, str(seed), str(AD_K), str(ad_pct),
        json.dumps(cfg.get("label") or {}, sort_keys=True),
        json.dumps(cfg.get("filter") or {}, sort_keys=True),
        json.dumps((cfg.get("features") or {}).get("drop_descriptors") or [],
                   sort_keys=True),
    ]
    for kind in ("regression", "classification"):
        for suffix in ("_fp.npy", "_desc.npy", "_meta.csv", "_columns.json"):
            parts.append(_stat(f"{featdir}/{stem}_{kind}{suffix}"))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# -------------------------------------------------------------- fit or load
def build_state(cfg, stem, featdir, blocks, seed, ad_pct):
    """Fit both models and calibrate the domain threshold. Slow path."""
    try:
        rfp = np.load(f"{featdir}/{stem}_regression_fp.npy")
        rde = np.load(f"{featdir}/{stem}_regression_desc.npy")
        rcols = json.load(open(f"{featdir}/{stem}_regression_columns.json"))
        rmeta = list(_csv.DictReader(
            open(f"{featdir}/{stem}_regression_meta.csv")))
        cfp = np.load(f"{featdir}/{stem}_classification_fp.npy")
        cde = np.load(f"{featdir}/{stem}_classification_desc.npy")
        cmeta = list(_csv.DictReader(
            open(f"{featdir}/{stem}_classification_meta.csv")))
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nRun `pipeline.py build` first.")

    cutoff = cfg["label"]["active_cutoff"]
    want_fmt = (cfg.get("filter") or {}).get("bao_format")
    drop = set((cfg.get("features") or {}).get("drop_descriptors") or [])
    desc_names = rcols["descriptors"]["names"]
    keep = [j for j, n in enumerate(desc_names) if n not in drop]

    def rows_for(meta):
        if not want_fmt:
            return list(range(len(meta)))
        return [i for i, r in enumerate(meta) if r.get("bao_format") == want_fmt]

    ridx, cidx = rows_for(rmeta), rows_for(cmeta)
    if not ridx or not cidx:
        sys.exit(f"ERROR: format filter {want_fmt} leaves no rows.")

    Xr = FB.assemble(rfp[ridx], rde[ridx], keep, blocks)
    yr = np.array([float(rmeta[i]["pchembl_value"]) for i in ridx])

    def is_active(r):
        p = r.get("pchembl_value")
        try:
            return 1 if p not in ("", None) and float(p) >= cutoff else 0
        except ValueError:
            return 0

    Xc = FB.assemble(cfp[cidx], cde[cidx], keep, blocks)
    yc = np.array([is_active(cmeta[i]) for i in cidx])

    print(f"  features: {FB.describe(blocks, rfp.shape[1], len(keep))}")
    print(f"  training on {len(ridx)} potency rows, {len(cidx)} classification "
          f"rows ({yc.sum()} active)" +
          (f", format {want_fmt}" if want_fmt else ""))

    def fit_prep(X):
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        return med, np.where(np.isfinite(X), X, med).std(axis=0) > 0

    med_r, live_r = fit_prep(Xr)
    med_c, live_c = fit_prep(Xc)

    t0 = time.time()
    print("  fitting potency model...", end="", flush=True)
    reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                random_state=seed).fit(
        np.where(np.isfinite(Xr), Xr, med_r)[:, live_r], yr)
    print(f" {time.time() - t0:.0f}s")
    t0 = time.time()
    print("  fitting classifier...", end="", flush=True)
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                 class_weight="balanced",
                                 random_state=seed).fit(
        np.where(np.isfinite(Xc), Xc, med_c)[:, live_c], yc)
    print(f" {time.time() - t0:.0f}s")

    # ---- applicability domain, calibrated on the training set's own geometry
    # Always on the PHARMACOPHORE SIGNATURE, whatever --blocks selects for the
    # model, so refusal rates stay comparable across ablations. Note this is a
    # pharmacophore-similarity domain, not a substructure one: a compound can be
    # inside it while sharing no scaffold with anything in training. That is the
    # intent, and it is why the refusal rate here is not comparable to an ECFP4
    # pipeline's — only to another run of this one.
    train_fp = np.asarray(rfp[ridx], dtype=np.uint8)
    pop = train_fp.astype(np.float32).sum(1)
    rng = np.random.default_rng(seed)
    cal = (rng.choice(len(train_fp), 2000, replace=False)
           if len(train_fp) > 2000 else np.arange(len(train_fp)))
    t0 = time.time()
    print(f"  calibrating domain threshold on {len(cal)} of {len(train_fp)} "
          f"compounds...", end="", flush=True)
    self_d = _ad_distance(train_fp[cal], train_fp, pop, exclude_self=True)
    ad_threshold = float(np.percentile(self_d, ad_pct))
    print(f" {time.time() - t0:.0f}s")

    return {
        "reg": reg, "clf": clf,
        "med_r": med_r, "live_r": live_r, "med_c": med_c, "live_c": live_c,
        "desc_names": desc_names, "keep": keep, "blocks": blocks,
        "fpspec": rcols["fingerprint"],
        "fold": int(rcols["fingerprint"]["folded_bits"]),
        "train_fp": train_fp, "train_pop": pop,
        # Two key sets. The full InChIKey includes the stereo layer, so the
        # same substance drawn WITHOUT stereochemistry gets a different key and
        # would slip past the contamination check. The first block encodes
        # connectivity only, so it catches stereo-variant duplicates — which is
        # most of what a natural-product library will collide with.
        "train_keys": {r["inchikey"] for r in rmeta},
        "train_skeletons": {r["inchikey"].split("-")[0] for r in rmeta
                            if r.get("inchikey")},
        "ad_threshold": ad_threshold,
        "ad_median": float(np.median(self_d)), "ad_pct": ad_pct,
        "cutoff": cutoff, "want_fmt": want_fmt,
        "n_reg": len(ridx), "n_clf": len(cidx), "n_active": int(yc.sum()),
    }


def _ad_distance(query_fp, train_fp, train_pop, exclude_self=False, block=256):
    """1 - mean Tanimoto to the AD_K nearest training compounds, vectorised.

    Tanimoto on bit vectors is |A&B| / (|A| + |B| - |A&B|), and the intersection
    for every pair is one matrix product. Blocked so the |query| x |train| matrix
    never has to exist all at once.
    """
    A = np.asarray(train_fp, dtype=np.float32)
    out = np.empty(len(query_fp), dtype=float)
    for i in range(0, len(query_fp), block):
        B = np.asarray(query_fp[i:i + block], dtype=np.float32)
        inter = B @ A.T
        union = B.sum(1)[:, None] + train_pop[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        k = AD_K + 1 if exclude_self else AD_K
        k = min(k, sim.shape[1])
        top = -np.partition(-sim, k - 1, axis=1)[:, :k]
        top = np.sort(top, axis=1)[:, ::-1]
        if exclude_self:
            top = top[:, 1:]        # drop the self-match at similarity 1.0
        out[i:i + block] = 1.0 - top.mean(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True, help="CSV of compounds to screen")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--annotation-col", default=None,
                    help="optional column of prior annotations, for the "
                         "enrichment test")
    ap.add_argument("--annotation-positive", default=None,
                    help="value in --annotation-col that marks the class you "
                         "expect to be enriched")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--featdir", default="features")
    ap.add_argument("--modeldir", default="models")
    ap.add_argument("--out", default="screen_results.csv")
    ap.add_argument("--gate", choices=["classifier", "regressor"],
                    default="classifier",
                    help="what decides active/not. Use `regressor` only if "
                         "table3_classification.md says the thresholded potency "
                         "model beat the classifier.")
    ap.add_argument("--exclude-known", action="store_true",
                    help="drop query compounds already in the training set from "
                         "the shortlist and the enrichment test. Handover §9b "
                         "item 4 — without it the model is congratulated for "
                         "remembering.")
    ap.add_argument("--ad-percentile", type=float, default=95.0,
                    help="the domain threshold is this percentile of the "
                         "training set's own neighbour distance. Higher is more "
                         "permissive. 95 is the default; report the refusal "
                         "rate across a range rather than defending one value.")
    ap.add_argument("--score-all", action="store_true",
                    help="score out-of-domain compounds too, marked "
                         "OUT OF DOMAIN, instead of refusing them. Their "
                         "predictions have no support and must be reported "
                         "separately, never merged into the shortlist.")
    ap.add_argument("--refit", action="store_true",
                    help="ignore any cached model and refit from scratch.")
    ap.add_argument("--jobs", type=int, default=0,
                    help="processes for standardising and featurising the "
                         "query list. 0 = all cores.")
    ap.add_argument("--seed", type=int, default=0)
    FB.add_arg(ap)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    stem = args.stem or list(cfg["targets"].values())[0].replace("-", "").lower()
    blocks = FB.resolve(cfg, args.blocks)

    # ---- READ AND CHECK THE INPUT FIRST -------------------------------------
    # Before anything expensive. Being told a column name is wrong after the
    # models have fitted is the kind of small cruelty that makes people stop
    # running things.
    if not os.path.exists(args.input):
        sys.exit(f"ERROR: no such file: {args.input}")
    with open(args.input, newline="", encoding="utf-8-sig") as f:
        head = f.readline()
        f.seek(0)
        delim = "\t" if head.count("\t") > head.count(",") else ","
        qrows = list(_csv.DictReader(f, delimiter=delim))
    if delim == "\t":
        print(f"{args.input}: TAB-separated, reading it as such")
    if not qrows:
        sys.exit(f"ERROR: {args.input} has no rows")
    for col, flag in ((args.smiles_col, "--smiles-col"),
                      (args.id_col, "--id-col"),
                      (args.annotation_col, "--annotation-col")):
        if col and col not in qrows[0]:
            sys.exit(f"ERROR: no column {col!r} in {args.input}.\n"
                     f"Columns found: {', '.join(qrows[0].keys())}\n"
                     f"Pass the right one with {flag}.")
    if args.annotation_col and not args.annotation_positive:
        sys.exit("ERROR: --annotation-col needs --annotation-positive, the "
                 "value that marks the class you expect to be enriched.")
    print(f"{args.input}: {len(qrows)} compounds to screen")

    # ---- fitted state: load if the key matches, otherwise fit and save ------
    os.makedirs(args.modeldir, exist_ok=True)
    key = cache_key(cfg, stem, args.featdir, blocks, args.seed,
                    args.ad_percentile)
    path = os.path.join(args.modeldir, f"{stem}_{blocks}_{key}.joblib")
    if os.path.exists(path) and not args.refit:
        t0 = time.time()
        S = joblib.load(path)
        print(f"loaded fitted model from {path} ({time.time() - t0:.0f}s)")
        print(f"  features: {FB.describe(blocks, S['train_fp'].shape[1], len(S['keep']))}")
        print(f"  trained on {S['n_reg']} potency rows, {S['n_clf']} "
              f"classification rows ({S['n_active']} active)")
    else:
        print("fitting models (no cached model for this data and config)")
        S = build_state(cfg, stem, args.featdir, blocks, args.seed,
                        args.ad_percentile)
        joblib.dump(S, path, compress=3)
        print(f"  saved to {path} — later runs on this data load it in seconds")

    cutoff, want_fmt = S["cutoff"], S["want_fmt"]
    ad_threshold, keep = S["ad_threshold"], S["keep"]
    print(f"  AD threshold = {ad_threshold:.3f} "
          f"({S['ad_pct']:g}th percentile of training-set distance; "
          f"median {S['ad_median']:.3f})")
    # Featurisation parameters come from the COLUMNS FILE, which records what
    # featurise.py actually did, not from the config, which may have been edited
    # since. A mismatch would fold the queries to a different width from the
    # training set and every Tanimoto distance below would compare two different
    # encodings.
    cfgfp = (cfg.get("features") or {}).get("pharmacophore") or {}
    if "folded_bits" in cfgfp and cfgfp["folded_bits"] != S["fpspec"]["folded_bits"]:
        print(f"  WARNING: config says folded_bits="
              f"{cfgfp['folded_bits']} but the training features were built "
              f"with {S['fpspec']['folded_bits']}. Using the features. Re-run "
              f"`build` if the config is the correct one.")

    # ---- featurise the whole query list, in parallel ------------------------
    smis = [(r.get(args.smiles_col) or "").strip() for r in qrows]
    jobs = args.jobs or (os.cpu_count() or 1)
    t0 = time.time()
    print(f"\nstandardising and featurising {len(smis)} compounds on "
          f"{jobs} process(es)...")
    initargs = (S["desc_names"], S["fold"])
    if jobs > 1:
        with Pool(jobs, initializer=_init_worker, initargs=initargs) as pool:
            done, feats = 0, []
            for res in pool.imap(_featurise_one, smis, chunksize=32):
                feats.append(res)
                done += 1
                if done % 500 == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"  {done}/{len(smis)}  "
                          f"({rate:.0f}/s, ~{(len(smis)-done)/rate/60:.0f} min left)")
    else:
        _init_worker(*initargs)
        feats = [_featurise_one(s) for s in smis]
    print(f"  featurised in {time.time() - t0:.0f}s")

    ok = [i for i, f in enumerate(feats) if f is not None]
    n_bad = len(feats) - len(ok)
    if not ok:
        sys.exit("ERROR: not one SMILES parsed. Wrong column, or the file is "
                 "mangled — check it in a text editor.")

    F32_MAX = np.finfo(np.float32).max
    Q_fp = np.stack([np.frombuffer(feats[i][2], dtype=np.uint8) for i in ok])
    Q_de = np.clip(np.asarray([feats[i][3] for i in ok], dtype=np.float64),
                   -F32_MAX, F32_MAX).astype(np.float32)
    Q = FB.assemble(Q_fp, Q_de, keep, S["blocks"])

    # ---- domain distance, vectorised ---------------------------------------
    t0 = time.time()
    print("computing applicability-domain distances...")
    dist = _ad_distance(Q_fp, S["train_fp"], S["train_pop"])
    print(f"  {time.time() - t0:.0f}s")
    in_dom = dist <= ad_threshold

    # ---- one batched prediction, not 5,821 single ones ---------------------
    scored = np.ones(len(ok), dtype=bool) if args.score_all else in_dom
    p_active = np.full(len(ok), np.nan)
    pred = np.full(len(ok), np.nan)
    if scored.any():
        t0 = time.time()
        print(f"predicting for {int(scored.sum())} compounds...")
        Xs = Q[scored]
        Xr = np.where(np.isfinite(Xs), Xs, S["med_r"])[:, S["live_r"]]
        pred[scored] = S["reg"].predict(Xr)
        if args.gate == "classifier":
            Xc = np.where(np.isfinite(Xs), Xs, S["med_c"])[:, S["live_c"]]
            p_active[scored] = S["clf"].predict_proba(Xc)[:, 1]
        print(f"  {time.time() - t0:.0f}s")

    # ---- assemble ----------------------------------------------------------
    out, counts = [], Counter()
    pos = {i: j for j, i in enumerate(ok)}
    for n, q in enumerate(qrows):
        qid = q.get(args.id_col) if args.id_col else f"row{n + 1}"
        rec = {"id": qid, "input_smiles": smis[n], "std_smiles": "",
               "in_training_set": 0, "ad_distance": "", "in_domain": "",
               "p_active": "", "predicted_pchembl": "", "rank_score": "",
               "training_match": "", "decision": "", "reason": ""}
        if args.annotation_col:
            rec["annotation"] = q.get(args.annotation_col, "")

        if n not in pos:
            rec.update(decision="REFUSED",
                       reason=("empty SMILES cell" if not smis[n]
                               else "SMILES did not parse or standardise"))
            counts["unparseable"] += 1
            out.append(rec); continue

        j = pos[n]
        rec["std_smiles"] = feats[n][0]
        ik = feats[n][1]
        exact = ik in S["train_keys"]
        skel = ik.split("-")[0] in S.get("train_skeletons", set())
        rec["in_training_set"] = int(exact or skel)
        rec["training_match"] = ("exact" if exact else
                                "same skeleton, different stereochemistry"
                                if skel else "")
        rec["ad_distance"] = round(float(dist[j]), 4)
        rec["in_domain"] = int(bool(in_dom[j]))

        if not scored[j]:
            rec.update(decision="REFUSED",
                       reason=f"outside applicability domain "
                              f"(distance {dist[j]:.3f} > {ad_threshold:.3f})")
            counts["refused_out_of_domain"] += 1
            out.append(rec); continue

        if args.gate == "regressor":
            passed = pred[j] >= cutoff
            why = f"predicted pIC50 {pred[j]:.2f} vs cutoff {cutoff}"
            rec["rank_score"] = round(float(pred[j]), 3)
        else:
            rec["p_active"] = round(float(p_active[j]), 4)
            passed = p_active[j] >= 0.5
            why = f"classifier {p_active[j]:.2f}"
            rec["rank_score"] = rec["p_active"]

        ood = "" if in_dom[j] else " [OUT OF DOMAIN — no support for this score]"
        if not passed:
            rec.update(decision="PREDICTED INACTIVE" + ("" if in_dom[j]
                                                        else " (OUT OF DOMAIN)"),
                       reason=f"{why} — below the active line. No potency "
                              f"reported, because a value for something "
                              f"predicted inactive is not actionable{ood}")
            counts["predicted_inactive" if in_dom[j] else "ood_inactive"] += 1
            out.append(rec); continue

        rec["predicted_pchembl"] = round(float(pred[j]), 3)
        rec.update(decision="PREDICTED ACTIVE" + ("" if in_dom[j]
                                                  else " (OUT OF DOMAIN)"),
                   reason=f"{why}, predicted pIC50 {pred[j]:.2f}{ood}")
        counts["predicted_active" if in_dom[j] else "ood_active"] += 1
        out.append(rec)

    # ---- contamination -----------------------------------------------------
    n_known = sum(r["in_training_set"] for r in out)
    n_skel = sum(1 for r in out
                 if r["training_match"].startswith("same skeleton"))
    if args.exclude_known:
        for r in out:
            if r["in_training_set"]:
                r["decision"] = "EXCLUDED"
                r["reason"] = ("already in the training set — excluded from the "
                               "shortlist and the enrichment test "
                               "(--exclude-known)")
        counts = Counter()
        for r in out:
            d, dom = r["decision"], r["in_domain"]
            if d == "EXCLUDED":
                counts["excluded_known"] += 1
            elif d.startswith("PREDICTED ACTIVE"):
                counts["predicted_active" if dom == 1 else "ood_active"] += 1
            elif d.startswith("PREDICTED INACTIVE"):
                counts["predicted_inactive" if dom == 1 else "ood_inactive"] += 1
            elif "outside applicability" in r["reason"]:
                counts["refused_out_of_domain"] += 1
            else:
                counts["unparseable"] += 1

    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    # ---- the report --------------------------------------------------------
    total = len(out)
    md = [f"# Screen — {args.input} against {cfg['name']}\n",
          f"- **Compounds screened**: {total}",
          f"- **Model**: {S['n_reg']} potency rows / {S['n_clf']} "
          f"classification rows" + (f", assay format {want_fmt}" if want_fmt else ""),
          f"- **Feature blocks**: `{S['blocks']}`",
          f"- **Active cutoff**: pIC50 >= {cutoff}",
          f"- **Applicability domain**: 1 - mean Tanimoto to {AD_K} nearest "
          f"training compounds; threshold {ad_threshold:.3f} "
          f"({S['ad_pct']:g}th percentile of the training set's own neighbour "
          f"distance, median {S['ad_median']:.3f})",
          f"- **Gate**: `{args.gate}`\n"]
    if args.score_all:
        md.append("⚠️ `--score-all` was used. Out-of-domain compounds were "
                  "scored rather than refused. Those scores have **no support** "
                  "— the model has no comparable training data for them — and "
                  "they are marked `OUT OF DOMAIN` in every row. Report them "
                  "separately or not at all; never merge them into the "
                  "shortlist.\n")

    md += ["## Outcome\n", "| | n | % |", "|---|---:|---:|"]
    labels = [("predicted_active", "**PREDICTED ACTIVE** — shortlist"),
              ("predicted_inactive", "predicted inactive"),
              ("ood_active", "out of domain, scored active (no support)"),
              ("ood_inactive", "out of domain, scored inactive (no support)"),
              ("refused_out_of_domain", "REFUSED — outside the domain"),
              ("unparseable", "REFUSED — SMILES did not parse"),
              ("excluded_known", "EXCLUDED — already in training")]
    print(f"\n{'=' * 72}\nOUTCOME\n{'=' * 72}")
    for k, label in labels:
        if not counts[k]:
            continue
        md.append(f"| {label} | {counts[k]} | {100 * counts[k] / total:.1f}% |")
        plain = label.replace("**", "")
        print(f"  {plain:<50}{counts[k]:>6}  {100 * counts[k] / total:>5.1f}%")

    refused = counts["refused_out_of_domain"] + counts["unparseable"]
    if refused:
        md.append(f"\n**{100 * refused / total:.0f}% of this library was "
                  f"refused.** That is the headline number, not the shortlist "
                  f"size. A model trained on one target's medicinal-chemistry "
                  f"series has no basis for most of an unrelated library, and "
                  f"saying so is the point of the domain gate.\n")

    if n_bad:
        md.append(f"{n_bad} compound(s) could not be parsed or standardised. "
                  f"They are listed in `{args.out}` with a reason rather than "
                  f"dropped.\n")

    if args.exclude_known:
        md.append(f"\n**{n_known} compound(s) excluded as already present in "
                  f"the training set** (`--exclude-known`). They are listed in "
                  f"`{args.out}` marked `EXCLUDED` but take no part in the "
                  f"shortlist or the enrichment test. Report this number — it "
                  f"is the contamination check, not a footnote.\n")
        if n_skel:
            md.append(f"Of those, **{n_skel} matched on connectivity but not "
                      f"stereochemistry** — the same skeleton drawn with "
                      f"different or absent stereo annotation. Matching only "
                      f"full InChIKeys would have missed them, and since the "
                      f"fingerprint ignores chirality the model genuinely has "
                      f"seen those structures.\n")
        print(f"\n  --exclude-known: {n_known} already in training, excluded")
    elif n_known:
        md.append(f"\n⚠️ **{n_known} of these compounds are already in the "
                  f"training set** and are still included below. Re-run with "
                  f"`--exclude-known` before quoting any of this.\n")
        print(f"\n  ⚠️ {n_known} query compound(s) already in the training set. "
              f"Re-run with --exclude-known before quoting the shortlist.")

    gate_label = ("classifier confidence" if args.gate == "classifier"
                  else "predicted pIC50")
    shortlist = sorted([r for r in out if r["decision"] == "PREDICTED ACTIVE"],
                       key=lambda r: -r["rank_score"])
    if shortlist:
        md.append(f"\n## Shortlist — top 40 by {gate_label}\n")
        md.append("| rank | id | p(active) | predicted pIC50 | AD distance |")
        md.append("|---:|---|---:|---:|---:|")
        for i, r in enumerate(shortlist[:40], 1):
            pa = f"{r['p_active']:.3f}" if r["p_active"] != "" else "—"
            md.append(f"| {i} | {r['id']} | {pa} | {r['predicted_pchembl']} "
                      f"| {r['ad_distance']:.3f} |")

    # ---- the out-of-domain list, kept SEPARATE on purpose -----------------
    ood = sorted([r for r in out
                  if r["decision"].startswith("PREDICTED ACTIVE (OUT OF DOMAIN)")],
                 key=lambda r: -r["rank_score"])
    if ood:
        md.append(f"\n## Out of domain, scored — {len(ood)} compounds "
                  f"(`--score-all`)\n")
        md.append("⚠️ **These scores have no support.** The model has no "
                  "comparable training data for these compounds, so the forest "
                  "routed each one through splits on feature combinations it "
                  "has never seen and the prediction landed wherever the tree "
                  "geometry sent it. That is an arbitrary number, not a weak "
                  "one. They are listed because a screen that discards what it "
                  "could not handle misreports its own coverage — **not** "
                  "because the ranking within this table means anything.\n")
        md.append("Use it to triage by your own chemistry, or to pick what goes "
                  "to a structure-based method that has no training domain. Do "
                  "not merge it with the shortlist above, and do not quote a hit "
                  "rate from it.\n")
        md.append("| rank | id | p(active) | predicted pIC50 | AD distance | "
                  "beyond threshold by |")
        md.append("|---:|---|---:|---:|---:|---:|")
        for i, r in enumerate(ood[:40], 1):
            pa = f"{r['p_active']:.3f}" if r["p_active"] != "" else "—"
            md.append(f"| {i} | {r['id']} | {pa} | {r['predicted_pchembl']} "
                      f"| {r['ad_distance']:.3f} "
                      f"| +{r['ad_distance'] - ad_threshold:.3f} |")
        far = [r for r in ood if r["ad_distance"] > ad_threshold + 0.15]
        if far:
            md.append(f"\n{len(far)} of these sit more than 0.15 beyond the "
                      f"threshold. At that distance the nearest training "
                      f"compound shares almost nothing with the query, and the "
                      f"score is closer to the training mean than to a "
                      f"prediction.\n")

    # ---- refusal rate across thresholds -----------------------------------
    # The threshold is a judgement call nobody can make objectively. Showing how
    # the answer moves with it is stronger than defending one value.
    parsed = [r for r in out if r["ad_distance"] != ""]
    if parsed:
        d = np.array([r["ad_distance"] for r in parsed])
        md.append("\n## Refusal rate against domain threshold\n")
        md.append("| threshold | refused | % of parsed |")
        md.append("|---:|---:|---:|")
        for t in sorted({0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7,
                         round(ad_threshold, 3)}):
            n = int((d > t).sum())
            mark = "  ← in use" if abs(t - round(ad_threshold, 3)) < 1e-9 else ""
            md.append(f"| {t:.3f}{mark} | {n} | {100 * n / len(d):.1f}% |")
        md.append(f"\nMedian query distance {np.median(d):.3f}; the training "
                  f"set's own median is {S['ad_median']:.3f}. The gap between "
                  f"those two is how far this library sits from the model's "
                  f"experience.\n")

    # ---- enrichment -------------------------------------------------------
    if args.annotation_col and args.annotation_positive:
        def hit(r):
            return args.annotation_positive.lower() in str(
                r.get("annotation", "")).lower()

        elig = [r for r in out if r["rank_score"] != ""
                and r["decision"] != "EXCLUDED" and r["in_domain"] == 1]
        p, q = [r for r in elig if hit(r)], [r for r in elig if not hit(r)]
        md.append("\n## Enrichment against prior annotation\n")
        if len(p) < 10 or len(q) < 10:
            md.append(f"Too few annotated compounds survived the domain gate "
                      f"({len(p)} positive, {len(q)} other) for this "
                      f"comparison to mean anything.\n")
            print(f"\n  enrichment test skipped — {len(p)} annotated, "
                  f"{len(q)} other in domain")
        else:
            from scipy.stats import mannwhitneyu
            a = np.array([float(r["rank_score"]) for r in p])
            b = np.array([float(r["rank_score"]) for r in q])
            u = mannwhitneyu(a, b, alternative="greater")
            auc = float(u.statistic / (len(a) * len(b)))
            md += [
                f"Compounds annotated **{args.annotation_positive}** "
                f"(n={len(a)}) versus the rest (n={len(b)}), ranked by "
                f"{gate_label}. Every in-domain compound is included whether or "
                f"not it passed the gate — testing only the survivors would "
                f"condition on the outcome.\n",
                f"| | median {gate_label} | n |", "|---|---:|---:|",
                f"| {args.annotation_positive} | {np.median(a):.3f} | {len(a)} |",
                f"| other | {np.median(b):.3f} | {len(b)} |",
                f"\n**AUC {auc:.3f}**, Mann-Whitney one-sided p = {u.pvalue:.2g}\n",
                "This is the test worth having, because **the annotation is "
                "independent of the training data**. The model learned from "
                "ChEMBL measurements; the annotation came from somewhere else "
                "entirely. An enrichment here is two unrelated lines of "
                "evidence agreeing, not a model recognising its own training "
                "set.\n",
                "⚠️ The annotation is recorded at the **source-organism** level, "
                "not the compound level. A plant used for inflammation holds "
                "hundreds of compounds and this one may not be why it worked. "
                "State that in the caption.\n"]
            print(f"\n{'=' * 72}\nENRICHMENT AGAINST PRIOR ANNOTATION\n{'=' * 72}")
            print(f"  {args.annotation_positive:<24} median {np.median(a):.3f}  (n={len(a)})")
            print(f"  {'other':<24} median {np.median(b):.3f}  (n={len(b)})")
            print(f"  AUC {auc:.3f}, p = {u.pvalue:.2g}")
            print("  -> enriched. Two independent sources agree."
                  if auc > 0.6 and u.pvalue < 0.05 else
                  "  -> no enrichment. Report it: the model may not transfer to "
                  "this chemistry, the annotation may be too noisy, or the "
                  "active constituents may not be in this list. Being unable to "
                  "separate those is itself a finding.")

    md.append("\n## What this is and is not\n")
    md.append("A **ranked shortlist**. Nothing here has been tested. The "
              "defensible claim is *these are the compounds most worth an "
              "experiment*, never *these work*.\n")
    md.append("Refused compounds are listed with a reason rather than dropped. "
              "A screen that silently discards what it could not handle "
              "misreports its own coverage.\n")
    with open("screen_report.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {args.out} and screen_report.md")


if __name__ == "__main__":
    main()
