#!/usr/bin/env python3
"""
End-to-end driver. Three commands, one decision between the first two.

    python pipeline.py census --config targets/mytarget.json
        Fetches everything ChEMBL has for your target and reports which assay
        formats exist, with counts. Then STOPS, because choosing a format is a
        judgement only you can make — the codes and their sizes differ for
        every target and cannot be known in advance.

    [ you put one bao_format into the config ]

    python pipeline.py build --config targets/mytarget.json
        stereo -> extract -> featurise -> characterise -> potency model ->
        controls -> classifier. Runs straight through. Halts only on a real
        error, and says what to do.

    python pipeline.py screen --config targets/mytarget.json --input mylist.csv
        Your own compounds through the built models: standardise, applicability
        -domain gate, classify, then predict potency for what passes. Optional
        enrichment test against a prior annotation column.

PHARMACOPHORE FEATURES, ONE ASSAY FORMAT
---------------------------------------
Molecules are featurised as a folded Gobbi 2D PHARMACOPHORE signature — pairs and
triplets of donor / acceptor / aromatic / hydrophobe / acidic / basic features
binned by topological distance — plus the RDKit descriptor block. No conformers
and no stereochemistry are required, which matters when a fifth of the compounds
of interest carry unspecified stereocentres.

The reason is measured: an ECFP4 model refused 88.5% of an ethnopharmacological
library, and the gap was chemotype, not size or decoration. A substructure
fingerprint asks whether an exact atom environment is present, so an unfamiliar
scaffold looks like nothing. A pharmacophore asks whether a donor sits a certain
distance from a hydrophobe, which a terpenoid and a coxib can both satisfy.

`filter.bao_format` is REQUIRED and is not re-tested. The confound test and the
format-held-out splits are gone: that argument was settled (AUC 0.887 against a
0.489 null; excess SD 0.70 and 1.02 log units on two of three transitions) and
the evidence is in RESULTS-2026-09-29.md.

THE ABLATION
------------
    python pipeline.py build --config targets/mytarget.json --blocks pharmacophore

Trains on the pharmacophore signature alone, with no RDKit descriptors, and
writes to suffixed files so it does not overwrite the full run. Do the same on
`screen`. It matters here because a molecule that sets ZERO pharmacophore bits is
carried entirely by its descriptors — see panel D of the characterisation figure.
`feature_blocks.py` has how to read the three outcomes.

Everything else lives in the config. Repointing at a new protein is a new JSON
file and these same three commands.

WHY THE STOP EXISTS
-------------------
You cannot name the assay format up front. BAO codes vary by target — COX-2 has
5, COX-1 has 7, a bacterial target will have different ones. The same compound
measured in purified enzyme and in whole blood differs by 3-10x, so pooling
formats trains the model on several quantities stacked under one column name.
The census counts what is actually there; you pick; everything after is
automatic.
"""

import argparse
import json
import os
import subprocess
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, what):
    """Run a stage. Stop the whole pipeline on failure, loudly and usefully."""
    print(f"\n{'=' * 72}\n{what}\n{'=' * 72}")
    r = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\nPIPELINE HALTED at: {what}\n"
                 f"  command: {' '.join(cmd)}\n"
                 f"  exit code: {r.returncode}\n\n"
                 f"Fix the cause above, then re-run. Completed stages are not "
                 f"repeated — their outputs are already on disk.")


def load(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: no config at {path}\n"
                 f"Copy targets/TEMPLATE.json and edit it.")
    # A hand-edited config is the likeliest thing to be malformed, and a raw
    # JSONDecodeError traceback tells you a character offset when what you need
    # is the line you broke. Show it.
    with open(path) as f:
        raw = f.read()
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        lines = raw.splitlines()
        lo, hi = max(1, e.lineno - 3), min(len(lines), e.lineno + 1)
        window = "\n".join(
            f"  {'>>' if n == e.lineno else '  '} {n:>3} | {lines[n - 1]}"
            for n in range(lo, hi + 1))
        hint = ""
        if "delimiter" in e.msg:
            hint = ("\nThis almost always means a MISSING COMMA at the end of "
                    "the line ABOVE the one marked. Every entry inside { } "
                    "needs a comma after it except the last one.\n"
                    "Editing a value often takes its trailing comma with it.")
        elif "Expecting value" in e.msg:
            hint = ("\nUsually a trailing comma before a } or ], or an unquoted "
                    "value. In JSON every string needs double quotes; there is "
                    "no None, True or False — use null, true, false.")
        elif "property name" in e.msg:
            hint = "\nUsually a trailing comma before the closing } of a block."
        sys.exit(f"ERROR: {path} is not valid JSON.\n"
                 f"  {e.msg}, line {e.lineno} column {e.colno}\n\n"
                 f"{window}\n{hint}\n\n"
                 f"Check it any time you edit a config:\n"
                 f"  python -c \"import json; json.load(open('{path}')); "
                 f"print('ok')\"")
    for k in ("name", "targets"):
        if k not in cfg:
            sys.exit(f"ERROR: config is missing required key '{k}'.")

    # Catch the field mix-up early. `assay_type` is a ChEMBL column with
    # single-letter values; `bao_format` is the assay-format code. Putting a BAO
    # code in assay_type matches nothing, so the cascade silently empties and
    # the failure surfaces three stages later as a confusing error.
    at = (cfg.get("cascade") or {}).get("assay_type")
    if at is not None and (len(str(at)) > 1 or str(at).startswith("BAO")):
        sys.exit(f"ERROR: cascade.assay_type is {at!r}.\n\n"
                 f"That field takes a single letter — B (binding), F "
                 f"(functional), A (ADME), T (toxicity) — or null.\n"
                 f"A BAO_* code belongs in `filter.bao_format` instead:\n\n"
                 f'    "cascade": {{ ..., "assay_type": "B" }},\n'
                 f'    "filter":  {{ "bao_format": "{at}" }}\n')

    bf = (cfg.get("filter") or {}).get("bao_format")
    if bf is not None and not str(bf).startswith("BAO"):
        sys.exit(f"ERROR: filter.bao_format is {bf!r}, which is not a BAO code.\n"
                 f"Run `census` and copy a value from the assay-format table.")
    return cfg


# --------------------------------------------------------------------- census
def census(cfg, path):
    db = f"{cfg['name']}_subset.db"
    if not os.path.exists(db):
        run(["fetch_target.py", "--config", path], "FETCH — ChEMBL web API")
    else:
        print(f"{db} already exists — skipping fetch. Delete it to refetch.")

    con = sqlite3.connect(db)
    try:
        release = con.execute("SELECT name FROM version").fetchone()[0]
    except Exception:
        release = "UNKNOWN"
    types = con.execute("""
        SELECT act.standard_type, COUNT(DISTINCT act.molregno)
        FROM activities act GROUP BY act.standard_type
        ORDER BY 2 DESC""").fetchall()

    want = cfg.get("endpoint", {}).get("standard_types", ["IC50"])
    casc = cfg.get("cascade", {})
    rows = con.execute("""
        SELECT a.bao_format, a.description, act.standard_type,
               a.confidence_score, a.assay_type, act.standard_relation,
               act.pchembl_value, act.molregno
        FROM activities act JOIN assays a ON act.assay_id = a.assay_id""").fetchall()
    con.close()

    md = [f"# Census — {cfg['name']}\n",
          f"- **Target(s)**: " + ", ".join(
              f"{a} ({l})" for a, l in cfg["targets"].items()),
          f"- **ChEMBL release**: {release}",
          f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
          f"- **Cascade applied**: relation '=', pchembl present, "
          f"confidence_score >= {casc.get('min_confidence_score', 8)}, "
          f"assay_type = {casc.get('assay_type', 'B')}\n",
          "## Endpoint types present\n",
          "| standard_type | compounds | |",
          "|---|---:|---|"]

    print(f"\n{'=' * 72}\nENDPOINT TYPES PRESENT\n{'=' * 72}")
    print(f"{'standard_type':<16}{'compounds':>11}")
    for t, n in types:
        mark = "  <- selected in config" if t in want else ""
        print(f"{str(t):<16}{n:>11}{mark}")
        md.append(f"| `{t}` | {n} | "
                  f"{'**selected**' if t in want else ''} |")

    # curated counts per assay format, using the SAME cascade the build will
    min_conf = casc.get("min_confidence_score", 8)
    atype = casc.get("assay_type", "B")
    per_fmt = Counter()
    descs = {}
    for bao, desc, st, conf, at, rel, pch, mol in rows:
        if st not in want or rel != "=" or pch is None:
            continue
        if min_conf is not None and (conf or 0) < min_conf:
            continue
        if atype and at != atype:
            continue
        per_fmt[bao or "(none)"] += 1
        descs.setdefault(bao or "(none)", Counter())[(desc or "")[:70]] += 1

    print(f"\n{'=' * 72}\nASSAY FORMATS — curated compound counts\n{'=' * 72}")
    if not per_fmt:
        sys.exit("\nNo records survive the cascade. Loosen `cascade` in the "
                 "config (min_confidence_score and assay_type are the usual "
                 "culprits) and re-run.\n"
                 "For ORGANISM targets, min_confidence_score MUST be null — "
                 "organism assays score low by design.")

    total = sum(per_fmt.values())
    md += ["\n## Assay formats — curated compound counts\n",
           "One row per `bao_format`, after the same cascade the model uses. "
           "**Train on one of these.** The same compound measured in purified "
           "enzyme versus whole blood differs three- to ten-fold, so a pooled "
           "set is several quantities stacked under one column name.\n",
           "| bao_format | compounds | share | most common assay description |",
           "|---|---:|---:|---|"]
    print(f"{'bao_format':<18}{'compounds':>11}{'share':>9}   most common assay description")
    for bao, n in per_fmt.most_common():
        top = descs[bao].most_common(1)[0][0]
        print(f"{bao:<18}{n:>11}{100*n/total:>8.0f}%   {top}")
        md.append(f"| `{bao}` | {n} | {100*n/total:.0f}% | {top} |")

    md.append("\n### Other descriptions inside each format\n")
    for bao, _ in per_fmt.most_common():
        md.append(f"\n**`{bao}`**\n")
        for d, k in descs[bao].most_common(6):
            md.append(f"- {k} — {d}")

    chosen = (cfg.get("filter") or {}).get("bao_format")
    print(f"\n{'=' * 72}")
    if chosen:
        if chosen in per_fmt:
            print(f"Config already selects {chosen} ({per_fmt[chosen]} compounds).")
            print(f"Run:  python pipeline.py build --config {path}")
            md.append(f"\n## Selection\n\n**`{chosen}`** — "
                      f"{per_fmt[chosen]} compounds, "
                      f"{100*per_fmt[chosen]/total:.0f}% of the curated set.\n")
        else:
            print(f"⚠️  Config selects '{chosen}', which is NOT in the list "
                  f"above. Fix it before building.")
            md.append(f"\n## Selection\n\n⚠️ Config selects `{chosen}`, "
                      f"which is not present in this target's data.\n")
    else:
        best = per_fmt.most_common(1)[0]
        print("CHOOSE ONE FORMAT, then re-run with `build`.")
        print(f'\nPut it in {path}:\n')
        print(f'    "filter": {{ "bao_format": "{best[0]}" }}\n')
        print(f"{best[0]} is the largest ({best[1]} compounds), which is the "
              f"usual choice —\nbut read the descriptions above first. The "
              f"biggest format is not always\nthe one you want: a purified-enzyme "
              f"assay and a whole-blood assay measure\ndifferent things, and only "
              f"you know which matches your question.")
        print(f"\nBelow ~500 compounds, expect the model to struggle to beat "
              f"its null baselines.")
        md.append(f"\n## Selection\n\n**Not yet chosen.** Largest is "
                  f"`{best[0]}` ({best[1]} compounds). Read the descriptions "
                  f"above before committing — the biggest format is not always "
                  f"the right one, and only you know which measurement matches "
                  f"the question. Below ~500 compounds the model will struggle "
                  f"to beat its nulls.\n")

    out = f"census_{cfg['name']}.md"
    open(os.path.join(HERE, out), "w").write("\n".join(md))
    print(f"\nwrote {out}")


# ---------------------------------------------------------------------- build
def build(cfg, path, extra=()):
    name = cfg["name"]
    db = f"{name}_subset.db"
    if not os.path.exists(db):
        sys.exit(f"ERROR: no {db}. Run `census` first.")
    chosen = (cfg.get("filter") or {}).get("bao_format")
    if not chosen:
        sys.exit("ERROR: no assay format selected.\n\n"
                 "This pipeline requires ONE declared format and does not pool. "
                 "Pooling stacks several different measurements under one column "
                 "name; the confound that makes that unsafe was already measured "
                 "(AUC 0.887 against a 0.489 shuffled null).\n\n"
                 "Run `census` first, then add to the config:\n"
                 '    "filter": { "bao_format": "BAO_..." }')

    # Stereo runs FIRST: it reads only the subset database, and extract joins
    # its output into the curated rows. The other order leaves stereo_status
    # blank in every curated CSV.
    run(["stereo_status.py", "--db", db],
        "STEREO — chirality status, structure-checked where ChEMBL is blank")
    run(["extract.py", "--config", path, "--db", db],
        "EXTRACT — cascade, standardisation, deduplication")

    stems = []
    for kind in ("regression", "classification"):
        for label in cfg["targets"].values():
            stem = f"{label.replace('-', '').lower()}_{kind}"
            csv_path = os.path.join("curated", f"{stem}.csv")
            if os.path.exists(os.path.join(HERE, csv_path)):
                fold = ((cfg.get("features") or {}).get("pharmacophore")
                        or {}).get("folded_bits")
                run(["featurise.py", "--csv", csv_path]
                    + (["--fold", str(fold)] if fold else []),
                    f"FEATURISE — {stem}\n"
                    f"2D pharmacophore signature (Gobbi, topological "
                    f"distances) + RDKit descriptors. No conformers, no "
                    f"stereochemistry needed.")
                if kind == "regression":
                    stems.append(stem)

    if not stems:
        sys.exit("ERROR: extract produced no regression CSV. Check curated/.")

    for stem in stems:
        run(["explore_space.py", "--stem", stem, "--config", path],
            f"DATA CHARACTERISATION — {stem}\n"
            f"Describes the data. Produces no performance number and validates "
            f"nothing — read it to catch a dataset that cannot support a model.")
        run(["train_model.py", "--stem", stem, "--config", path] + list(extra),
            f"POTENCY MODEL + NEGATIVE CONTROLS — {stem}")
        base = stem.replace("_regression", "")
        if os.path.exists(os.path.join(
                HERE, "features", f"{base}_classification_fp.npy")):
            # the classifier has no split axes of its own; pass only --blocks
            clf_extra, skip = [], False
            for a in extra:
                if skip:
                    skip = False; continue
                if a.startswith("--blocks"):
                    clf_extra.append(a)
                    if "=" not in a:
                        clf_extra.append(extra[list(extra).index(a) + 1])
                elif a in ("--repeats", "--splits"):
                    skip = True
            run(["train_classifier.py", "--stem", base, "--config", path]
                + clf_extra,
                f"CLASSIFICATION — active vs not, with the measured inactives "
                f"the potency model cannot use — {base}")

    print(f"\n{'=' * 72}\nDONE\n{'=' * 72}")
    sfx = ""
    for a in extra:
        if a.startswith("--blocks"):
            v = a.split("=", 1)[1] if "=" in a else extra[list(extra).index(a) + 1]
            sfx = "" if v == "both" else f"_{v}"
    for f in ("census_" + name + ".md", "curation_log.md",
              "featurisation_report.md", f"table2{sfx}.md",
              f"table3_classification{sfx}.md", f"negative_controls{sfx}.md"):
        if os.path.exists(os.path.join(HERE, f)):
            print(f"  {f}")
    print("  figures/, curated/, features/, results_table2.csv")
    print("\nRead in this order:")
    print(f"  negative_controls{sfx}.md{'':<{max(0, 6 - len(sfx))}}  "
          f"decides whether to believe anything else")
    print(f"  curation_log.md           what the data became — Table 1")
    print(f"  table2{sfx}.md            potency model, margin over null")
    print(f"  table3_classification{sfx}.md  active/not, and whether the extra "
          f"model earns its place")
    print("\nThen, to run your own compound list through it:")
    print(f"  python pipeline.py screen --config {path} --input mylist.csv")


# --------------------------------------------------------------------- screen
def screen(cfg, path, extra):
    """Run an external compound list against the built models."""
    name = cfg["name"]
    base = next(iter(cfg["targets"].values())).replace("-", "").lower()
    need = os.path.join(HERE, "features", f"{base}_regression_fp.npy")
    if not os.path.exists(need):
        sys.exit(f"ERROR: no trained features at {need}\n"
                 f"Run `python pipeline.py build --config {path}` first.")
    if not any(a == "--input" for a in extra):
        sys.exit("ERROR: screen needs --input yourlist.csv\n\n"
                 "  python pipeline.py screen --config " + path +
                 " --input cmfoa.csv \\\n"
                 "      --smiles-col smiles --id-col compound_id\n\n"
                 "The CSV needs one column of SMILES. Everything else is "
                 "optional.")
    run(["screen.py", "--config", path] + extra,
        f"SCREEN — external compound list against {name}")
    print(f"\n{'=' * 72}\nDONE\n{'=' * 72}")
    print("  screen_report.md     read this")
    print("  screen_results.csv   every compound, with a decision and a reason")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["census", "build", "screen"])
    ap.add_argument("--config", required=True)
    args, extra = ap.parse_known_args()
    cfg = load(args.config)
    if args.command == "screen":
        screen(cfg, args.config, extra)
    elif args.command == "census":
        census(cfg, args.config)
    else:
        allowed = {"--blocks", "--repeats", "--splits"}
        bad = [a for a in extra
               if a.startswith("--") and a.split("=")[0] not in allowed]
        if bad:
            sys.exit(f"ERROR: unrecognised argument(s) for `build`: "
                     f"{' '.join(bad)}\n"
                     f"`build` accepts --blocks "
                     f"{{both,pharmacophore,descriptors}}, --repeats N and "
                     f"--splits scaffold,temporal. Everything else lives in "
                     f"the config.")
        build(cfg, args.config, extra)


if __name__ == "__main__":
    main()
