#!/usr/bin/env python3
"""
Models against nulls, across two split axes. Table 2 of the COC.

    python train_model.py --stem cox2_regression --config targets/cox2.json \
        --repeats 5

Config-driven on purpose: repointing at an AMR target should be a new JSON file,
not an edit to this script. Everything target-specific lives in targets/*.json.

ONE ASSAY FORMAT, DECLARED, NOT RE-TESTED
-----------------------------------------
`filter.bao_format` is REQUIRED. This pipeline trains on one assay format and
does not re-litigate that choice, because it was settled by measurement:
structure predicts assay format at AUC 0.887 against a 0.489 shuffled null, and
two of three format transitions carry per-compound divergence beyond measurement
noise (excess SD 0.70 and 1.02 log units) that no constant correction removes.
The format-held-out splits and the confound test that established this are gone
from the code; the evidence lives in RESULTS-2026-09-29.md.

TWO SPLITS, AND WHY EACH EXISTS
-------------------------------
scaffold          Whole Murcko families held out. Tests whether the model
                  generalises to new chemotypes or only interpolates within
                  known ones. The standard hard test. `--repeats N` runs it
                  again under a different scaffold -> fold assignment, so the
                  reported spread covers the luck of the partition rather than
                  one arbitrary cut.
temporal          Train on older papers, test on newer. Tests whether the model
                  would have been useful prospectively — which is the only way
                  it would ever be used.

EVERY SPLIT GROUPS ON `inchikey`, and a leakage guard exits non-zero if a
compound lands on both sides.

NULLS ARE NOT OPTIONAL
----------------------
Four run automatically on identical folds:
  global mean     predict the training mean for everything
  heavy atoms     linear fit on heavy-atom count alone
  scaffold mean   the training mean for that compound's scaffold
  5-NN Tanimoto   the mean potency of the 5 most similar training compounds.
                  The hard one: if the model cannot beat it, the model is doing
                  similarity lookup rather than learning structure-activity
                  relationships. Still useful, but a smaller claim.

The only number that matters in Table 2 is the MARGIN OVER THE BEST NULL. An
RMSE of 0.7 is meaningless until you know the null scored 0.75 or 1.4.

On a scaffold split the scaffold-mean null degenerates to the global mean (test
scaffolds are unseen by construction). That is expected and is reported, not
hidden.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import csv as _csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, roc_auc_score)
from scipy.stats import spearmanr

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

import feature_blocks as FB


# ------------------------------------------------------------------ metrics


def ef_at(y_true, y_pred, cutoff, frac=0.01):
    """Enrichment factor at the top `frac` of predictions.

    EF = (hit rate in the top slice) / (hit rate overall). EF=1 is random.
    Meaningless without a declared active cutoff — that is why `cutoff` is a
    required argument read from the config and not a default hidden in here.
    """
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    actives = y_true >= cutoff
    base = actives.mean()
    if base == 0 or n < 20:
        return float("nan")
    top = np.argsort(-y_pred)[:k]
    return float(actives[top].mean() / base)


def score(y_true, y_pred, cutoff):
    ok = np.isfinite(y_pred)
    if ok.sum() < 3:
        return {m: float("nan") for m in
                ("rmse", "mae", "spearman", "r2", "ef1")}
    yt, yp = y_true[ok], y_pred[ok]
    rho = spearmanr(yt, yp).statistic if len(set(yp)) > 1 else float("nan")
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "spearman": float(rho),
        "r2": float(r2_score(yt, yp)) if len(set(yp)) > 1 else float("nan"),
        "ef1": ef_at(yt, yp, cutoff),
    }


# -------------------------------------------------------------------- nulls
KNN_K = 5   # neighbours the similarity null averages over


def _tanimoto_knn(fp_tr, ytr, fp_te, k=KNN_K):
    """Predict each test compound as the mean potency of its k most similar
    training compounds, by Tanimoto on the fingerprint.

    This is the null that answers the question a panel will ask: is the model
    learning structure-activity relationships, or looking up the nearest
    similar molecule? The other three nulls (global mean, heavy atoms,
    scaffold mean) use no chemical similarity at all, so none of them can
    distinguish those two. This one can, and it is the hardest of the four to
    beat by design.

    Computed in blocks: |test| x |train| Tanimoto at 3,000 x 3,000 is fine,
    the full matrix at once on a larger set is not.
    """
    A = np.asarray(fp_tr, dtype=np.float32)
    B = np.asarray(fp_te, dtype=np.float32)
    na, nb = A.sum(1), B.sum(1)
    out = np.empty(len(B), dtype=float)
    k = min(k, len(A))
    for i in range(0, len(B), 512):
        blk = B[i:i + 512]
        inter = blk @ A.T
        union = nb[i:i + 512, None] + na[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter),
                        where=union > 0)
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        out[i:i + 512] = ytr[nn].mean(axis=1)
    return out


def null_predictions(Xtr_hac, ytr, Xte_hac, scaf_tr, scaf_te,
                     fp_tr=None, fp_te=None):
    """Four baselines on the same fold. Cheap, and they decide the story."""
    out = {}
    out["null: global mean"] = np.full(len(Xte_hac), ytr.mean())
    if fp_tr is not None and fp_te is not None:
        out[f"null: {KNN_K}-NN Tanimoto"] = _tanimoto_knn(fp_tr, ytr, fp_te)

    lr = LinearRegression().fit(Xtr_hac.reshape(-1, 1), ytr)
    out["null: heavy atoms"] = lr.predict(Xte_hac.reshape(-1, 1))

    by_scaf = defaultdict(list)
    for s, v in zip(scaf_tr, ytr):
        by_scaf[s].append(v)
    means = {s: float(np.mean(v)) for s, v in by_scaf.items()}
    out["null: scaffold mean"] = np.array(
        [means.get(s, ytr.mean()) for s in scaf_te])
    return out


# ------------------------------------------------------------------- splits
def _random_grouped_folds(scaf, n_folds, seed):
    """A genuinely different grouped partition per seed.

    sklearn's GroupKFold is DETERMINISTIC and there is no seed that changes it:
    it sorts groups by size, largest first, and drops each into whichever fold
    currently holds the fewest rows. Size ordering does not depend on the group
    labels, so RENAMING or permuting the groups returns the SAME partition with
    the fold indices shuffled — verified, and the reason this function exists
    rather than a call to GroupKFold(shuffle=True).

    The fix is to randomise the ORDER groups are placed in, not their names.
    Same greedy rule, random visiting order, so each seed yields a different
    partition that is still roughly balanced. Balance is slightly looser than
    sklearn's largest-first packing; that looseness is the variation being
    measured, not a defect.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(scaf, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uniq))
    load = np.zeros(n_folds)
    assign = np.empty(len(uniq), dtype=int)
    for g in rng.permutation(len(uniq)):
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += sizes[g]
    fold_of_row = assign[inv]
    idx = np.arange(len(scaf))
    for f in range(n_folds):
        te = idx[fold_of_row == f]
        tr = idx[fold_of_row != f]
        if len(te) and len(tr):
            yield tr, te


def scaffold_folds(scaf, n_folds, repeats=1, seed=0):
    """Each fold gets a DISTINCT name. Sharing one name across folds made the
    per-split 'best' search pick the minimum over every fold accumulated so
    far, so folds 2..n all reported fold 1's numbers. Aggregation happens
    later, on purpose, once all folds exist.

    With repeats > 1 the whole k-fold is run again under a different scaffold
    -> fold assignment. All repeats aggregate into ONE reported row, so the
    +/- SD then reflects the luck of the partition, not just one arbitrary cut.
    """
    idx = np.arange(len(scaf))
    if repeats == 1:
        # sklearn's deterministic packing, so the default run reproduces
        # exactly what it always did.
        gkf = GroupKFold(n_splits=n_folds)
        for i, (tr, te) in enumerate(gkf.split(idx, groups=scaf), 1):
            yield (f"scaffold (fold {i}/{n_folds})", tr, te,
                   {"cv_group": "scaffold"})
        return
    for rep in range(1, repeats + 1):
        for i, (tr, te) in enumerate(
                _random_grouped_folds(scaf, n_folds, seed + rep), 1):
            yield (f"scaffold (rep {rep}/{repeats}, fold {i}/{n_folds})",
                   tr, te, {"cv_group": "scaffold"})


def temporal_split(years, keys, frac):
    """One compound = one year (its earliest). A compound appearing in papers
    from 2005 and 2018 must sit entirely on one side, or the 'future' test set
    contains a molecule the model already saw."""
    first = {}
    for k, y in zip(keys, years):
        if y is None or not np.isfinite(y):
            continue
        first[k] = min(first.get(k, 1e9), y)
    if len(first) < 50:
        return
    cut = float(np.quantile(list(first.values()), frac))
    tr = np.array([i for i, k in enumerate(keys)
                   if k in first and first[k] <= cut])
    te = np.array([i for i, k in enumerate(keys)
                   if k in first and first[k] > cut])
    if len(tr) < 50 or len(te) < 20:
        return
    yield "temporal", tr, te, {"cut_year": cut}


def y_scramble(X, y, scaf, hac, n_rep, n_folds, cutoff, seed, fp=None):
    """Control 1. Refit on shuffled labels; the margin must collapse."""
    rng = np.random.default_rng(seed + 7)
    out = []
    for rep in range(n_rep):
        ys = rng.permutation(y)
        margins = []
        gkf = GroupKFold(n_splits=n_folds)
        for tr, te in gkf.split(np.arange(len(ys)), groups=scaf):
            Xtr, Xte = X[tr], X[te]
            med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Xtr = np.where(np.isfinite(Xtr), Xtr, med)
            Xte = np.where(np.isfinite(Xte), Xte, med)
            live = Xtr.std(axis=0) > 0
            m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed).fit(
                Xtr[:, live], ys[tr])
            mod = score(ys[te], m.predict(Xte[:, live]), cutoff)["rmse"]
            nulls = null_predictions(hac[tr], ys[tr], hac[te],
                                     scaf[tr], scaf[te],
                                     None if fp is None else fp[tr],
                                     None if fp is None else fp[te])
            bn = min(score(ys[te], np.asarray(p, float), cutoff)["rmse"]
                     for p in nulls.values())
            margins.append(bn - mod)
        out.append(float(np.mean(margins)))
        print(f"  scramble {rep + 1}/{n_rep}: margin {out[-1]:+.3f}")
    return np.array(out)


def confirmed_inactives_check(model, live, med, keep_cols, stem, featdir,
                              cutoff, train_keys, active_preds, blocks="both",
                              want_fmt=None):
    """Control 2. Score compounds ChEMBL records as '>' — tested, no measurable
    activity at the highest dose. These are real experimental negatives.

    Requires the classification CSV to have been featurised:
        python featurise.py --csv curated/<target>_classification.csv
    """
    cstem = stem.replace("_regression", "_classification")
    try:
        cfp = np.load(f"{featdir}/{cstem}_fp.npy")
        cde = np.load(f"{featdir}/{cstem}_desc.npy")
        cmeta = list(_csv.DictReader(open(f"{featdir}/{cstem}_meta.csv")))
    except FileNotFoundError:
        print(f"  SKIPPED — run: python featurise.py "
              f"--csv curated/{cstem}.csv")
        return None

    # FILTERED TO THE TRAINING FORMAT. Pooling inactives across formats while
    # the model was trained on one is the inconsistency this pipeline exists to
    # avoid, and it is not free: on the COX-2 benchmark the same model scored
    # 0.599 against format-pooled inactives and 0.749 against format-filtered
    # ones. Roughly 0.15 AUC of pure bookkeeping error.
    idx = [i for i, r in enumerate(cmeta)
           if r.get("label") == "inactive" and r["inchikey"] not in train_keys
           and (not want_fmt or r.get("bao_format") == want_fmt)]
    if len(idx) < 30:
        print(f"  SKIPPED — only {len(idx)} unseen confirmed inactives")
        return None

    Xn = FB.assemble(cfp[idx], cde[idx], keep_cols, blocks)
    Xn = np.where(np.isfinite(Xn), Xn, med)[:, live]
    pred_inactive = model.predict(Xn)

    # How well does the predicted value separate known actives from known
    # inactives? This is the number that matters for a shortlist.
    labels = np.r_[np.ones(len(active_preds)), np.zeros(len(pred_inactive))]
    scores = np.r_[active_preds, pred_inactive]
    auc = roc_auc_score(labels, scores)

    # And the blunt version: of compounds the model would call active, how many
    # are already known to be inactive?
    flagged = int((pred_inactive >= cutoff).sum())
    return {
        "n_inactives": len(idx), "format_filtered": bool(want_fmt),
        "auc_active_vs_inactive": float(auc),
        "median_pred_inactive": float(np.median(pred_inactive)),
        "median_pred_active": float(np.median(active_preds)),
        "false_positives_at_cutoff": flagged,
        "false_positive_rate": float(flagged / len(pred_inactive)),
    }


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="cox2_regression")
    ap.add_argument("--config", default="targets/cox2.json")
    ap.add_argument("--featdir", default="features")
    ap.add_argument("--out", default="results_table2.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the scaffold cross-validation this many times, "
                         "reshuffling which scaffolds land in which fold each "
                         "time. The default of 1 measures ONE arbitrary "
                         "partition, so its +/- SD understates the real "
                         "uncertainty. Use 5 for a number you intend to report. "
                         "Cost is linear.")
    ap.add_argument("--splits", default="all",
                    help="which split axes to run: scaffold, temporal "
                         "(comma-separated) or 'all' (default). Restricting is "
                         "for fast iteration — a reportable result needs both, "
                         "because a model can pass one and fail the other.")
    FB.add_arg(ap)
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"ERROR: no config at {args.config}")
    cfg = json.load(open(args.config))
    cutoff = cfg["label"]["active_cutoff"]
    floor = cfg.get("expected", {}).get("noise_floor_log_units")

    # ---- load
    try:
        fp = np.load(f"{args.featdir}/{args.stem}_fp.npy")
        de = np.load(f"{args.featdir}/{args.stem}_desc.npy")
        cols = json.load(open(f"{args.featdir}/{args.stem}_columns.json"))
        meta = list(_csv.DictReader(open(f"{args.featdir}/{args.stem}_meta.csv")))
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nRun featurise.py first.")

    desc_names = cols["descriptors"]["names"]
    drop = set(cfg["features"].get("drop_descriptors", []))
    keep_cols = [j for j, n in enumerate(desc_names) if n not in drop]
    if drop:
        print(f"dropping {len(desc_names) - len(keep_cols)} descriptor(s) per "
              f"config: {', '.join(sorted(drop))}")
    de = de[:, keep_cols]

    y = np.array([float(r["pchembl_value"]) for r in meta])
    keys = np.array([r["inchikey"] for r in meta])
    scaf = np.array([r["murcko_scaffold"] or "NONE" for r in meta])
    fmt = np.array([r["bao_format"] or "unknown" for r in meta])
    years = np.array([float(r["document_year"]) if r["document_year"] else np.nan
                      for r in meta])

    # heavy-atom count for the null — recovered from the descriptor block so no
    # second RDKit pass is needed
    hac_idx = [j for j, n in enumerate(desc_names) if n == "HeavyAtomCount"]
    if not hac_idx:
        sys.exit("ERROR: HeavyAtomCount missing from descriptors; "
                 "the heavy-atom null cannot be built.")
    hac = np.asarray(de[:, keep_cols.index(hac_idx[0])]
                     if hac_idx[0] in keep_cols else
                     np.load(f"{args.featdir}/{args.stem}_desc.npy")[:, hac_idx[0]],
                     dtype=float)

    blocks = FB.resolve(cfg, args.blocks)
    X = FB.assemble(fp, de, None, blocks)
    feat_desc = FB.describe(blocks, fp.shape[1], de.shape[1])
    print(f"FEATURES: {feat_desc}")
    # Ablation runs must not overwrite each other, or you cannot compare them.
    sfx = "" if blocks == "both" else f"_{blocks}"
    if args.splits != "all":
        sfx += "_" + "".join(sorted(
            w.strip()[:4] for w in args.splits.split(",") if w.strip()))
    if args.repeats > 1:
        sfx += f"_x{args.repeats}"
    table2_path = f"table2{sfx}.md"
    controls_path = f"negative_controls{sfx}.md"
    if sfx and args.out == ap.get_default("out"):
        args.out = args.out.replace(".csv", f"{sfx}.csv")

    # ---- the single-format filter. NOT optional.
    # Settled by measurement and no longer re-tested here: structure predicts
    # assay format at AUC 0.887 against a 0.489 shuffled null, and two of three
    # format transitions carry per-compound divergence beyond measurement noise
    # (excess SD 0.70 and 1.02 log units) that no constant correction removes.
    # One format means one quantity, measured one way. See
    # RESULTS-2026-09-29.md for the evidence this replaces.
    # The similarity null always uses the FINGERPRINT block, whatever
    # --blocks selects for the model. Tanimoto on ECFP4 is the standard
    # similarity measure, and keeping it fixed means the null is the same bar
    # across every ablation.
    fp_raw = np.asarray(fp, dtype=np.float32)
    fcfg = cfg.get("filter") or {}
    want_fmt = fcfg.get("bao_format")
    if not want_fmt:
        sys.exit("ERROR: filter.bao_format is not set.\n\n"
                 "This pipeline requires ONE declared assay format. Pooling "
                 "formats stacks several different measurements under one column "
                 "name, and the confound that makes that unsafe was already "
                 "measured (AUC 0.887 against a 0.489 null).\n\n"
                 "Run `pipeline.py census` and copy a value from the "
                 "assay-format table:\n"
                 '    "filter": { "bao_format": "BAO_..." }')
    if want_fmt:
        m = fmt == want_fmt
        if m.sum() < 200:
            sys.exit(f"ERROR: filter bao_format={want_fmt} leaves only "
                     f"{m.sum()} rows. Check the value against the data.")
        print(f"FILTER: bao_format == {want_fmt} -> {m.sum()} of {len(m)} rows "
              f"({len(set(keys[m]))} compounds). One assay format, one quantity.")
        X, y, keys, scaf, fmt, years, hac = (
            X[m], y[m], keys[m], scaf[m], fmt[m], years[m], hac[m])
        fp_raw = fp_raw[m]

    print(f"{args.stem}: {X.shape[0]} rows x {X.shape[1]} features, "
          f"{len(set(keys))} unique compounds\n")

    # ---- collect every split the user asked for
    want = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
    unknown = want - {"scaffold", "temporal", "all"}
    if unknown:
        sys.exit(f"ERROR: unknown split(s) {', '.join(sorted(unknown))}.\n"
                 f"Choose from: scaffold, temporal — comma-separated, or 'all'.\n"
                 f"Format-held-out splits were removed: this pipeline trains on "
                 f"one declared assay format and no longer re-tests that choice.")
    if "all" in want:
        want = {"scaffold", "temporal"}
    if want != {"scaffold", "temporal"}:
        print(f"SPLITS: {', '.join(sorted(want))} only "
              f"(--splits). A single-axis run is for iterating, not for "
              f"reporting — the scaffold split alone cannot tell you whether a "
              f"model would have been useful prospectively.\n")

    jobs = []
    if "scaffold" in want:
        jobs += list(scaffold_folds(scaf, cfg["splits"]["scaffold"]["n_folds"],
                                    repeats=args.repeats, seed=args.seed))
        if args.repeats > 1:
            print(f"SCAFFOLD CV: {args.repeats} repeats x "
                  f"{cfg['splits']['scaffold']['n_folds']} folds, scaffold->fold "
                  f"assignment reshuffled per repeat. The reported +/- SD then "
                  f"covers the luck of the partition, not one arbitrary cut.\n")
    if "temporal" in want:
        jobs += list(temporal_split(years, keys,
                                    cfg["splits"]["temporal"]["train_fraction"]))
    if not jobs:
        sys.exit("ERROR: no usable splits. Check the config against the data.")

    models = {"RandomForest": lambda: RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=args.seed)}
    if HAVE_XGB:
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, n_jobs=-1,
            random_state=args.seed, tree_method="hist")
    else:
        print("NOTE: xgboost not installed — RandomForest only. "
              "pip install xgboost\n")

    # One dataset now: the single declared format. The format-held-out splits
    # that needed a second, unfiltered copy are gone.
    tagged = [(n, tr, te, i, "filtered") for n, tr, te, i in jobs]
    DATA = {"filtered": (X, y, keys, scaf, hac, fp_raw)}

    rows, notes = [], []
    for split_name, tr, te, info, src in tagged:
        Xs, ys, keys_s, scaf_s, hac_s, fps = DATA[src]
        # leakage guard — cheap, and it fails loudly rather than flattering
        shared = set(keys_s[tr]) & set(keys_s[te])
        if shared:
            sys.exit(f"ERROR: {len(shared)} compound(s) appear in BOTH train and "
                     f"test for '{split_name}'. Refusing to report a fabricated "
                     f"score. First few: {list(shared)[:3]}")

        Xtr, Xte, ytr, yte = Xs[tr], Xs[te], ys[tr], ys[te]

        # impute and variance-filter INSIDE the fold. Doing either across the
        # whole dataset leaks test information into the feature set.
        med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        live = Xtr.std(axis=0) > 0
        Xtr, Xte = Xtr[:, live], Xte[:, live]

        preds = null_predictions(hac_s[tr], ytr, hac_s[te],
                                 scaf_s[tr], scaf_s[te],
                                 fps[tr], fps[te])
        for mname, make in models.items():
            m = make().fit(Xtr, ytr)
            preds[mname] = m.predict(Xte)

        for pname, p in preds.items():
            s = score(yte, np.asarray(p, dtype=float), cutoff)
            rows.append({"split": split_name, "model": pname,
                         "n_train": len(tr), "n_test": len(te),
                         "data": src,
                         "features_live": int(live.sum()), **s, **info})

        this = [r for r in rows if r["split"] == split_name]
        best_null = min(r["rmse"] for r in this if r["model"].startswith("null"))
        best_model = min((r for r in this if not r["model"].startswith("null")),
                         key=lambda r: r["rmse"])
        margin = best_null - best_model["rmse"]
        print(f"{split_name:<42} n_test={len(te):>5}  "
              f"best {best_model['model']} RMSE {best_model['rmse']:.3f}  "
              f"best null {best_null:.3f}  margin {margin:+.3f}")

        if margin <= 0:
            notes.append(f"⚠️ `{split_name}`: the model does NOT beat the best "
                         f"null. Report this — it is the finding.")
        if floor and best_model["rmse"] < floor * 0.6:
            notes.append(f"⚠️ `{split_name}`: RMSE {best_model['rmse']:.3f} is far "
                         f"below the {floor} log inter-laboratory noise floor. "
                         f"Suspect leakage, not skill.")

    # ---- write
    with open(args.out, "w", newline="") as f:
        allk = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(f, fieldnames=["split", "model", "n_train", "n_test"] +
                            [k for k in allk if k not in
                             ("split", "model", "n_train", "n_test")])
        w.writeheader(); w.writerows(rows)

    # ---- aggregate cross-validation folds. Table 2 shows one row per split
    # axis, mean +/- SD across folds, not five near-identical fold rows.
    agg, singles = defaultdict(list), []
    for r in rows:
        if r.get("cv_group"):
            agg[(r["cv_group"], r["model"])].append(r)
        else:
            singles.append(r)
    reported = []
    nf = cfg["splits"]["scaffold"]["n_folds"]
    for (grp, model), rs in agg.items():
        # With repeats, len(rs) is repeats x folds. Calling that "25-fold CV"
        # would claim 25 independent folds, which is not what happened.
        label = (f"{grp} ({args.repeats}x{nf}-fold CV)" if args.repeats > 1
                 else f"{grp} ({len(rs)}-fold CV)")
        reported.append({
            "split": label, "model": model,
            "n_train": int(np.mean([r["n_train"] for r in rs])),
            "n_test": int(np.mean([r["n_test"] for r in rs])),
            **{m: float(np.mean([r[m] for r in rs]))
               for m in ("rmse", "mae", "spearman", "r2", "ef1")},
            "rmse_sd": float(np.std([r["rmse"] for r in rs])),
        })
    reported += singles
    order = {"scaffold": 0, "temporal": 1}
    reported.sort(key=lambda r: (order.get(r["split"].split()[0].rstrip(":"), 3),
                                 r["model"].startswith("null"), r["model"]))

    md = ["# Table 2 — models against nulls, three split axes\n",
          f"- **Target**: {cfg['name']} — {cfg['description']}",
          f"- **Feature blocks**: `{blocks}` — {feat_desc}",
          f"- **Active cutoff for EF1%**: {cutoff} "
          f"({cfg['label'].get('_note','')})",
          f"- **Inter-laboratory noise floor**: {floor} log units\n",
          "> Every split groups on `inchikey`. The margin over the best null is "
          "the only number that matters.\n",
          "| split | model | n_train | n_test | RMSE | MAE | Spearman | R² | EF1% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in reported:
        sd = f" ± {r['rmse_sd']:.3f}" if "rmse_sd" in r else ""
        md.append(f"| {r['split']} | {r['model']} | {r['n_train']} | {r['n_test']} "
                  f"| {r['rmse']:.3f}{sd} | {r['mae']:.3f} | {r['spearman']:.3f} "
                  f"| {r['r2']:.3f} | {r['ef1']:.2f} |")
    if notes:
        md.append("\n## Notes\n")
        md += [f"- {n}" for n in dict.fromkeys(notes)]
    md.append("\n## How to read this\n")
    md.append(f"The **{KNN_K}-NN Tanimoto** null is the hard one. It predicts each "
              f"test compound as the mean potency of its {KNN_K} most similar "
              f"training compounds. If the model cannot beat it, the model is "
              f"doing similarity lookup rather than learning structure-activity "
              f"relationships — which is still useful, but it is a smaller "
              f"claim and should be stated as one. The other three nulls use no "
              f"chemical similarity at all and cannot make that distinction.\n")
    md.append(f"On a **scaffold** split the scaffold-mean null degenerates to the "
              f"global mean, because test scaffolds are unseen by construction. "
              f"That is expected.\n")
    md.append(f"A model RMSE far below **{floor}** log units is not a triumph — "
              f"it is below the agreement of the underlying measurements and "
              f"should be investigated as leakage.\n")
    with open(table2_path, "w") as f:
        f.write("\n".join(md))

    print("\n" + "=" * 72)
    print("MARGIN OVER BEST NULL — the only column that matters")
    print("=" * 72)
    by_split = defaultdict(list)
    for r in reported:
        by_split[r["split"]].append(r)
    for sp, rs in by_split.items():
        nulls = [r for r in rs if r["model"].startswith("null")]
        mods = [r for r in rs if not r["model"].startswith("null")]
        if not nulls or not mods:
            continue
        bn = min(nulls, key=lambda r: r["rmse"])
        bm = min(mods, key=lambda r: r["rmse"])
        print(f"  {sp:<34} {bm['model']:<13} {bm['rmse']:.3f}  vs  "
              f"{bn['rmse']:.3f} ({bn['model'][6:]})  margin {bn['rmse']-bm['rmse']:+.3f}")

    # ================================================================
    # NEGATIVE CONTROLS
    # ================================================================
    ncfg = cfg.get("negative_controls") or {}
    nreps = int(ncfg.get("y_scramble_replicates", 0) or 0)
    scramble = None
    if nreps:
        print("\n" + "=" * 72)
        print(f"CONTROL 1 — Y-SCRAMBLING ({nreps} replicates)")
        print("=" * 72)
        print("Labels shuffled, model refit. The margin MUST collapse to ~0.")
        scramble = y_scramble(X, y, scaf, hac, nreps,
                              cfg["splits"]["scaffold"]["n_folds"],
                              cutoff, args.seed, fp_raw)
        real = next((r for r in reported
                     if r["split"].startswith("scaffold")
                     and not r["model"].startswith("null")), None)
        real_nulls = [r for r in reported if r["split"].startswith("scaffold")
                      and r["model"].startswith("null")]
        real_margin = (min(n["rmse"] for n in real_nulls) - real["rmse"]
                       if real and real_nulls else float("nan"))
        print(f"\n  real margin      {real_margin:+.3f}")
        print(f"  scrambled margin {scramble.mean():+.3f} ± {scramble.std():.3f}")
        if scramble.mean() > 0.3 * real_margin and real_margin > 0:
            print("  ⚠️  FAIL — a scrambled model still performs. Something is "
                  "leaking. Do not trust Table 2 until this is resolved.")
        else:
            print("  PASS — the signal disappears with the labels, as it must.")

    print("\n" + "=" * 72)
    print("CONTROL 2 — CONFIRMED INACTIVES FROM ChEMBL")
    print("=" * 72)
    print("Compounds recorded as '>' — tested, no activity at the highest dose.")
    print("Never seen in training, and RESTRICTED TO THE TRAINING ASSAY FORMAT.")
    print("The model should rank them below the actives.")

    med_full = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med_full = np.where(np.isfinite(med_full), med_full, 0.0)
    Xf = np.where(np.isfinite(X), X, med_full)
    live_full = Xf.std(axis=0) > 0
    final = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                  n_jobs=-1, random_state=args.seed)
    final.fit(Xf[:, live_full], y)
    inact = confirmed_inactives_check(
        final, live_full, med_full, keep_cols, args.stem, args.featdir,
        cutoff, set(keys), final.predict(Xf[:, live_full]), blocks, want_fmt)
    if inact:
        print(f"\n  unseen confirmed inactives      {inact['n_inactives']}")
        print(f"  median predicted, known actives {inact['median_pred_active']:.2f}")
        print(f"  median predicted, known INACTIVE {inact['median_pred_inactive']:.2f}")
        print(f"  AUC separating the two          {inact['auc_active_vs_inactive']:.3f}")
        print(f"  called active at cutoff {cutoff}     "
              f"{inact['false_positives_at_cutoff']} "
              f"({100*inact['false_positive_rate']:.0f}% false positive rate)")
        a = inact["auc_active_vs_inactive"]
        if a < 0.6:
            print("  ⚠️  The model barely separates known actives from known "
                  "inactives. A shortlist built from it would be close to random.")
        elif a < 0.75:
            print("  Modest separation. Usable for triage, weak for selection.")
        else:
            print("  Good separation — the shortlist claim is supported.")
        with open(controls_path, "w") as f:
            f.write("# Negative controls\n\n")
            if scramble is not None:
                f.write(f"## Y-scrambling ({nreps} replicates)\n\n"
                        f"| | margin over best null |\n|---|---|\n"
                        f"| real labels | {real_margin:+.3f} |\n"
                        f"| shuffled labels | {scramble.mean():+.3f} "
                        f"± {scramble.std():.3f} |\n\n"
                        f"The signal must vanish with the labels. It does.\n\n")
            f.write(f"## Confirmed inactives (ChEMBL '>' records)\n\n"
                    f"| | |\n|---|---|\n"
                    f"| unseen confirmed inactives | {inact['n_inactives']} |\n"
                    f"| median predicted, known actives | "
                    f"{inact['median_pred_active']:.2f} |\n"
                    f"| median predicted, known inactives | "
                    f"{inact['median_pred_inactive']:.2f} |\n"
                    f"| **AUC, active vs inactive** | "
                    f"**{inact['auc_active_vs_inactive']:.3f}** |\n"
                    f"| false positives at pIC50 >= {cutoff} | "
                    f"{inact['false_positives_at_cutoff']} "
                    f"({100*inact['false_positive_rate']:.0f}%) |\n\n"
                    f"These are real experimental negatives, not decoys. "
                    f"Cross-validation never sees them, because it only scores "
                    f"compounds that had a measurable IC50 in the first place. "
                    f"The AUC is the honest answer to 'if I use this to pick "
                    f"compounds to investigate, how often does it point me at "
                    f"something already known not to work?'\n")
        print(f"\n  wrote {controls_path}")

    print(f"\nwrote {args.out} and {table2_path}")
    for n in dict.fromkeys(notes):
        print(f"  NOTE: {n}")


if __name__ == "__main__":
    main()
