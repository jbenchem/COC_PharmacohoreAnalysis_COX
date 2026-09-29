#!/usr/bin/env python3
"""
Active / not-active classification, plus the thresholded-regressor baseline.

    python train_classifier.py --stem cox2 --config targets/cox2.json

Needs both featurised, which `pipeline.py build` does:
    features/<stem>_regression_*      the potency model's data
    features/<stem>_classification_*  the same compounds PLUS measured inactives

TWO DEFINITIONS OF "NEGATIVE", BOTH SCORED
-------------------------------------------
    all        everything below the cutoff — measured-weak AND recorded-inactive.
               Matches the practical question: is this worth investigating?
    confirmed  only the '>' records: compounds someone tested and found nothing
               in. Stricter, and contains no compound whose weakness was
               inferred from a number near the cutoff.

Both run by default. The difference between them says what the model is keying
on: a HIGHER score against confirmed-only is expected, because merely-weak
compounds sit near the boundary and are the hard cases. A LOWER score there
would mean the model keys on something the confirmed inactives share with the
actives, and the larger set was hiding it.

WHY THIS EXISTS ALONGSIDE THE REGRESSION
----------------------------------------
You can classify for free by thresholding a potency model: predict pIC50, call
anything above the cutoff active. So why train a second model?

Because THE REGRESSION CAN NEVER SEE AN INACTIVE. ChEMBL records them as '>'
— tested, nothing measurable at the highest dose — so they carry no value to
fit and regression training excludes every one of them. On COX-2 that is 891
compounds, ~27% more data, and they are precisely the compounds the question is
about.

The two also optimise different things. Regression minimises error across the
whole range, spending effort on whether something is 4.2 or 4.5 — a distinction
nobody cares about. Classification puts its effort at the decision boundary.

So three models are scored on the SAME held-out compounds:

    classifier              trained with the inactives
    regressor, thresholded  the free baseline — does the extra model earn itself?
    null (majority class)   predict the commonest label for everything

If the classifier does not beat the thresholded regressor, use the regressor and
say so. That is a real result and it saves a model.

NEGATIVES
---------
Default `--negatives all`: everything below the active cutoff counts as negative,
whether it was measured weak or recorded as inactive. That matches the practical
question — "is this worth investigating" — where a measured-weak compound and an
untestable one are the same answer.

`--negatives confirmed` uses ONLY the '>' records. Stricter, smaller, and it
answers the narrower question of whether the model separates actives from
compounds someone tested and found nothing.
"""

import argparse
import csv as _csv
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, balanced_accuracy_score)

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

import feature_blocks as FB


def ef_at(y_true, scores, frac=0.01):
    """Enrichment at the top `frac`. EF=1 is random."""
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    base = y_true.mean()
    if base == 0 or n < 20:
        return float("nan")
    return float(y_true[np.argsort(-scores)[:k]].mean() / base)


def score_binary(y_true, scores, thresh=0.5):
    pred = (scores >= thresh).astype(int)
    out = {"n_test": len(y_true), "n_positive": int(y_true.sum())}
    if len(set(y_true)) < 2:
        return {**out, **{m: float("nan") for m in
                          ("auc", "avg_precision", "precision", "recall",
                           "balanced_acc", "ef1")}}
    out.update({
        "auc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "ef1": ef_at(y_true, scores),
    })
    return out


def load(stem, featdir):
    fp = np.load(f"{featdir}/{stem}_fp.npy")
    de = np.load(f"{featdir}/{stem}_desc.npy")
    cols = json.load(open(f"{featdir}/{stem}_columns.json"))
    meta = list(_csv.DictReader(open(f"{featdir}/{stem}_meta.csv")))
    return fp, de, cols, meta


def prep(Xtr, Xte):
    """Impute and variance-filter INSIDE the fold. Across the whole dataset it
    leaks test information into the feature set."""
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    live = Xtr.std(axis=0) > 0
    return Xtr[:, live], Xte[:, live]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="cox2",
                    help="base name; _regression and _classification are appended")
    ap.add_argument("--config", required=True)
    ap.add_argument("--featdir", default="features")
    ap.add_argument("--negatives", choices=["all", "confirmed", "both"],
                    default="both",
                    help="both (default) runs each definition and reports the "
                         "difference, which says what the model is keying on")
    ap.add_argument("--out", default="table3_classification.csv")
    ap.add_argument("--seed", type=int, default=0)
    FB.add_arg(ap)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    cutoff = cfg["label"]["active_cutoff"]
    want_fmt = (cfg.get("filter") or {}).get("bao_format")
    blocks = FB.resolve(cfg, args.blocks)
    # Ablation runs must not overwrite each other.
    sfx = "" if blocks == "both" else f"_{blocks}"
    table3_path = f"table3_classification{sfx}.md"
    if sfx and args.out == ap.get_default("out"):
        args.out = args.out.replace(".csv", f"{sfx}.csv")

    try:
        cfp, cde, ccols, cmeta = load(f"{args.stem}_classification", args.featdir)
        rfp, rde, rcols, rmeta = load(f"{args.stem}_regression", args.featdir)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nRun `pipeline.py build` first — both the "
                 f"regression and classification sets must be featurised.")

    drop = set(cfg["features"].get("drop_descriptors", []))
    keep = [j for j, n in enumerate(ccols["descriptors"]["names"]) if n not in drop]

    # ---- labels
    def positive(r):
        p = r.get("pchembl_value")
        try:
            return 1 if p not in ("", None) and float(p) >= cutoff else 0
        except ValueError:
            return 0

    NEG_MODES = (["all", "confirmed"] if args.negatives == "both"
                 else [args.negatives])
    all_summaries, all_rows, all_headers = {}, [], {}

  # ---- one pass per definition of "negative"
    for NEG in NEG_MODES:
      idx = list(range(len(cmeta)))
      if want_fmt:
          idx = [i for i in idx if cmeta[i].get("bao_format") == want_fmt]
          if len(idx) < 200:
              sys.exit(f"ERROR: format filter {want_fmt} leaves {len(idx)} rows.")
      if NEG == "confirmed":
          idx = [i for i in idx
                 if positive(cmeta[i]) or cmeta[i].get("label") == "inactive"]

      y = np.array([positive(cmeta[i]) for i in idx])
      keys = np.array([cmeta[i]["inchikey"] for i in idx])
      scaf = np.array([cmeta[i]["murcko_scaffold"] or "NONE" for i in idx])
      lab = Counter(cmeta[i].get("label", "?") for i in idx)
      X = FB.assemble(cfp[idx], cde[idx], keep, blocks)

      hac_j = [j for j, n in enumerate(ccols["descriptors"]["names"])
               if n == "HeavyAtomCount"]
      hac = cde[idx][:, hac_j[0]].astype(float) if hac_j else np.zeros(len(idx))

      print(f"{args.stem}: {len(idx)} rows, {len(set(keys))} compounds, "
            f"{y.sum()} active ({100*y.mean():.0f}%), {len(y)-y.sum()} negative")
      print(f"  label mix: {dict(lab)}")
      print(f"  features: {FB.describe(blocks, cfp.shape[1], len(keep))}")
      print(f"  negatives: {NEG} "
            f"({'measured-weak and confirmed-inactive' if NEG=='all' else 'confirmed inactive only'})")
      if want_fmt:
          print(f"  format filter: {want_fmt}")
      if y.sum() < 30 or (len(y) - y.sum()) < 30:
          sys.exit("ERROR: too few in one class to model.")

      # ---- the free baseline: a potency regressor, thresholded.
      # Scored on the SAME compounds. It cannot be trained on the inactives, which
      # is the whole point of the comparison.
      reg_key_to_y = {r["inchikey"]: float(r["pchembl_value"]) for r in rmeta
                      if (not want_fmt or r.get("bao_format") == want_fmt)}
      from sklearn.ensemble import RandomForestRegressor
      rkeep = [j for j, n in enumerate(rcols["descriptors"]["names"])
               if n not in drop]
      ridx = [i for i, r in enumerate(rmeta)
              if (not want_fmt or r.get("bao_format") == want_fmt)]
      Xr = FB.assemble(rfp[ridx], rde[ridx], rkeep, blocks)
      yr = np.array([float(rmeta[i]["pchembl_value"]) for i in ridx])
      rkeys = np.array([rmeta[i]["inchikey"] for i in ridx])

      # Both models get the SAME class weighting, or the comparison is between
      # weighting schemes rather than algorithms. RandomForest takes
      # class_weight="balanced"; XGBoost's equivalent is scale_pos_weight set to
      # the negative:positive ratio. Leaving XGBoost unweighted on a ~15%-active
      # set would let it lean toward the majority and lose for the wrong reason.
      pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
      print(f"  class weighting: balanced "
            f"(scale_pos_weight = {pos_weight:.2f} for XGBoost)")
      models = {"classifier: RandomForest": lambda: RandomForestClassifier(
          n_estimators=500, min_samples_leaf=2, n_jobs=-1,
          class_weight="balanced", random_state=args.seed)}
      if HAVE_XGB:
          models["classifier: XGBoost"] = lambda: XGBClassifier(
              n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.6, n_jobs=-1, random_state=args.seed,
              tree_method="hist", eval_metric="logloss",
              scale_pos_weight=pos_weight)
      else:
          print("  NOTE: xgboost not installed — RandomForest only")

      rows = []
      n_folds = cfg["splits"]["scaffold"]["n_folds"]
      gkf = GroupKFold(n_splits=n_folds)
      for fold, (tr, te) in enumerate(gkf.split(np.arange(len(y)), groups=scaf), 1):
          if set(keys[tr]) & set(keys[te]):
              sys.exit(f"ERROR: compound in both train and test, fold {fold}. "
                       f"Refusing to report a fabricated score.")
          if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
              continue
          Xtr, Xte = prep(X[tr], X[te])
          preds = {}

          # null: majority class, constant score
          preds["null: majority class"] = np.full(len(te), float(y[tr].mean()))
          # null: molecule size alone
          lr = LogisticRegression(max_iter=1000).fit(hac[tr].reshape(-1, 1), y[tr])
          preds["null: heavy atoms"] = lr.predict_proba(hac[te].reshape(-1, 1))[:, 1]

          for name, make in models.items():
              m = make().fit(Xtr, y[tr])
              preds[name] = m.predict_proba(Xte)[:, 1]

          # thresholded regressor — trained ONLY on compounds with values, and
          # only on those whose scaffolds are in this fold's training set
          tr_scaf = set(scaf[tr])
          te_keys = set(keys[te])
          rtr = [i for i in range(len(yr)) if rkeys[i] not in te_keys]
          if len(rtr) > 100:
              Xrtr, Xrte_full = prep(Xr[rtr], Xr)
              rm = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                         n_jobs=-1, random_state=args.seed)
              rm.fit(Xrtr, yr[rtr])
              # score the classification test compounds by predicted potency
              Xte_forreg = FB.assemble(cfp[[idx[i] for i in te]],
                                       cde[[idx[i] for i in te]], rkeep, blocks
                                     ).astype(np.float32)
              _, Xte_r = prep(Xr[rtr], Xte_forreg)
              preds["baseline: regressor, thresholded"] = rm.predict(Xte_r)

          for pname, p in preds.items():
              thr = cutoff if pname.startswith("baseline") else 0.5
              s = score_binary(y[te], np.asarray(p, float), thr)
              rows.append({"fold": fold, "model": pname, "n_train": len(tr), **s})

          best = max((r for r in rows if r["fold"] == fold
                      and not r["model"].startswith("null")),
                     key=lambda r: r["auc"])
          bn = max((r for r in rows if r["fold"] == fold
                    and r["model"].startswith("null")), key=lambda r: r["auc"])
          print(f"  fold {fold}/{n_folds}  n_test={len(te):>5}  "
                f"best {best['model'][:28]:<30} AUC {best['auc']:.3f}  "
                f"null {bn['auc']:.3f}")

      if not rows:
          sys.exit("ERROR: no usable folds.")

      # ---- aggregate
      agg = defaultdict(list)
      for r in rows:
          agg[r["model"]].append(r)
      summary = []
      for model, rs in agg.items():
          summary.append({
              "model": model, "folds": len(rs),
              **{m: float(np.mean([r[m] for r in rs]))
                 for m in ("auc", "avg_precision", "precision", "recall",
                           "balanced_acc", "ef1")},
              "auc_sd": float(np.std([r["auc"] for r in rs])),
          })
      summary.sort(key=lambda r: -r["auc"])

      for r in rows:
          r["negatives"] = NEG
      all_rows.extend(rows)
      all_summaries[NEG] = summary

      md = [f"\n## Negatives: {NEG}\n",
            f"- **Active cutoff**: pIC50 >= {cutoff}",
            f"- **Negatives**: {NEG}",
            f"- **Assay format**: {want_fmt or 'all (pooled)'}",
            f"- **Feature blocks**: `{blocks}` — "
            f"{FB.describe(blocks, cfp.shape[1], len(keep))}",
            f"- **Split**: scaffold, {n_folds}-fold, grouped on `inchikey`",
            f"- **Class balance**: {y.sum()} active / {len(y)-y.sum()} negative\n",
            "> The classifier trains WITH the measured inactives. The thresholded "
            "regressor cannot — ChEMBL records inactives as '>' with no value to "
            "fit, so potency models discard them. That is the comparison.\n",
            "| model | AUC | avg precision | precision | recall | balanced acc | EF1% |",
            "|---|---|---|---|---|---|---|"]
      for r in summary:
          md.append(f"| {r['model']} | **{r['auc']:.3f}** ± {r['auc_sd']:.3f} "
                    f"| {r['avg_precision']:.3f} | {r['precision']:.3f} "
                    f"| {r['recall']:.3f} | {r['balanced_acc']:.3f} "
                    f"| {r['ef1']:.2f} |")

      clf = [r for r in summary if r["model"].startswith("classifier")]
      base = [r for r in summary if r["model"].startswith("baseline")]
      md.append("\n## Does the classifier earn its place?\n")
      print(f"\n{'=' * 72}\nDOES THE CLASSIFIER EARN ITS PLACE?\n{'=' * 72}")
      if clf and base:
          d = clf[0]["auc"] - base[0]["auc"]
          line = (f"Best classifier AUC {clf[0]['auc']:.3f} vs thresholded "
                  f"regressor {base[0]['auc']:.3f} — difference {d:+.3f}.")
          if d > 0.03:
              verdict = ("The classifier wins. The 891 measured inactives carry "
                         "information that potency regression discards, and a "
                         "model that sees them separates actives better.")
          elif d < -0.03:
              verdict = ("The thresholded regressor wins. Use it and drop the "
                         "second model — a simpler pipeline that performs as well "
                         "is the better answer.")
          else:
              verdict = ("No meaningful difference. Report both and use the "
                         "regressor, since it is one model rather than two.")
          print(f"  {line}\n  {verdict}")
          md += [line + "\n", verdict + "\n"]
      else:
          md.append("Baseline not computed — comparison unavailable.\n")

      all_headers[NEG] = md

    # ================================================================
    # ONE REPORT, BOTH DEFINITIONS OF "NEGATIVE"
    # ================================================================
    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    doc = [f"# Classification — active vs not, {args.stem}\n",
           f"- **Active cutoff**: pIC50 >= {cutoff}",
           f"- **Assay format**: {want_fmt or 'all (pooled)'}",
           f"- **Feature blocks**: `{blocks}` — "
           f"{FB.describe(blocks, cfp.shape[1], len(keep))}",
           f"- **Split**: scaffold, {n_folds}-fold, grouped on `inchikey`\n",
           "Two definitions of *negative* are scored, because they answer "
           "different questions:\n",
           "- **all** — everything below the cutoff, whether measured weak or "
           "recorded inactive. Matches the practical question: is this worth "
           "investigating?",
           "- **confirmed** — only the `>` records: compounds someone tested "
           "and found nothing in. Stricter, and free of any compound whose "
           "weakness was inferred from a number.\n",
           "> The classifier trains WITH the measured inactives. The "
           "thresholded regressor cannot — ChEMBL records inactives as '>' "
           "with no value to fit, so potency models discard them. That is the "
           "comparison.\n"]
    for NEG in NEG_MODES:
        doc += all_headers[NEG]

    if len(NEG_MODES) == 2:
        doc.append("\n## What the two definitions say together\n")
        print(f"\n{'=' * 72}\nALL vs CONFIRMED NEGATIVES\n{'=' * 72}")
        for tag in ("classifier", "baseline"):
            a = next((r for r in all_summaries["all"]
                      if r["model"].startswith(tag)), None)
            c = next((r for r in all_summaries["confirmed"]
                      if r["model"].startswith(tag)), None)
            if not (a and c):
                continue
            d = c["auc"] - a["auc"]
            line = (f"**{a['model']}** — AUC {a['auc']:.3f} against all "
                    f"negatives, {c['auc']:.3f} against confirmed only "
                    f"({d:+.3f}).")
            print(f"  {a['model']:<36} all {a['auc']:.3f}  "
                  f"confirmed {c['auc']:.3f}  ({d:+.3f})")
            doc.append(line)
        doc.append(
            "\nA **higher** score against confirmed-only means the model "
            "separates actives from genuinely untestable compounds more "
            "cleanly than from merely weak ones — which is what you would "
            "expect, since weak compounds sit near the cutoff and are the hard "
            "cases. A **lower** score there is worth investigating: it would "
            "mean the model is keying on something the confirmed inactives "
            "share with the actives, and the larger `all` set was hiding it.\n")

    doc.append("\n## Reading these\n")
    doc.append("**AUC** — probability a random active scores above a random "
               "negative. 0.5 is a coin flip.\n")
    doc.append("**EF1%** — how many more actives sit in the top 1% than chance "
               "would put there. This is the shortlist number.\n")
    doc.append("**Precision** — of the compounds called active, what fraction "
               "were. **Recall** — of the true actives, what fraction were "
               "found. A shortlist wants precision; a screen wants recall.\n")
    with open(table3_path, "w") as f:
        f.write("\n".join(doc))

    print(f"\n{'=' * 72}")
    for NEG in NEG_MODES:
        print(f"  negatives = {NEG}")
        for r in all_summaries[NEG]:
            print(f"    {r['model']:<36} AUC {r['auc']:.3f} ± "
                  f"{r['auc_sd']:.3f}   EF1% {r['ef1']:.2f}")
    print(f"\nwrote {args.out} and {table3_path}")


if __name__ == "__main__":
    main()
