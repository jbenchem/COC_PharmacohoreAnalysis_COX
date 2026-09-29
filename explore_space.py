#!/usr/bin/env python3
"""
Four diagnostics on the curated set, before any model is fitted.

    python explore_space.py --stem cox2_regression

WHAT THIS IS FOR, AND WHAT IT IS NOT
------------------------------------
It characterises the DATA. It does not validate a model and produces no
performance number. Read it to understand what the model is being asked to
learn, and to catch a dataset that cannot support a model before spending an
hour finding out the hard way.

The assay-format confound test and the paired cross-format deltas that used to
live here are GONE. That question is settled: structure predicts assay format at
AUC 0.887 against a 0.489 shuffled null, and two of three format transitions
carry divergence beyond measurement noise that no constant correction removes.
This pipeline trains on one declared format and does not re-argue it. The
evidence is in RESULTS-2026-09-29.md.

THE FOUR PANELS
---------------
A  Chemical space      t-SNE of the pharmacophore signature, coloured by
                       potency. Orientation only — t-SNE distances are not
                       metric and neighbourhoods at different scales are not
                       comparable. Look for whether potency has any spatial
                       structure at all, not for clusters to name.
B  Potency             the distribution being modelled, with the declared active
                       cutoff. A narrow distribution means a low null and little
                       room for any model to beat it.
C  Scaffold concentration
                       how few scaffolds hold how much of the data. The fewer
                       needed to cover half the rows, the harsher a scaffold
                       split is, and the more a random split would have
                       flattered the model.
D  Pharmacophore bits  how much signature each molecule actually sets. This
                       representation is only as good as its coverage, and a
                       molecule setting zero bits is running on descriptors
                       alone. If the zero count is large, the representation is
                       wrong for this dataset.

Light mode only: this is a figure for a printed document, not a web page.
"""

import argparse
import csv as _csv
import textwrap
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    sys.exit("ERROR: matplotlib not installed. pip install matplotlib")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- validated palette (see references/palette.md; checked with
# --- scripts/validate_palette.js against surface #fcfcfb)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"      # categorical slot 1. Panels B, C, D are each a single
                        # series, so they share one accent rather than implying
                        # three categories.
WARN = "#eb6834"        # categorical slot 2, for the reference lines only

# Sequential ramp for potency, steps 250->700 of the blue ramp. NOT started at
# step 100: low potency is meaningful data rather than "near zero", so the light
# end must stay visible against the surface. Step 250 clears 2:1 (2.06:1) and
# the ramp validates as monotone single-hue.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
POTENCY_CMAP = LinearSegmentedColormap.from_list("potency", RAMP)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def titled(ax, title, subtitle, width=58):
    """Wrap explicitly. matplotlib's wrap=True does not respect axes-fraction
    coordinates, so long subtitles run over the neighbouring panel."""
    lines = textwrap.wrap(" ".join(subtitle.split()), width=width)
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold",
                 loc="left", pad=13 + 11 * len(lines))
    for i, line in enumerate(lines):
        ax.text(0, 1.012 + 0.030 * (len(lines) - 1 - i), line,
                transform=ax.transAxes, color=INK_2, fontsize=8,
                va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="cox2_regression")
    ap.add_argument("--featdir", default="features")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--config", default=None,
                    help="optional, only to read the active cutoff for panel B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    try:
        fp = np.load(f"{args.featdir}/{args.stem}_fp.npy")
        cols = json.load(open(f"{args.featdir}/{args.stem}_columns.json"))
        with open(f"{args.featdir}/{args.stem}_meta.csv", newline="") as f:
            meta = list(_csv.DictReader(f))
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nRun featurise.py first.")

    cutoff = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cutoff = (json.load(f).get("label") or {}).get("active_cutoff")

    y = np.array([float(r["pchembl_value"]) for r in meta])
    scaf = np.array([r["murcko_scaffold"] or "NONE" for r in meta])
    keys = np.array([r["inchikey"] for r in meta])
    bits = fp.sum(axis=1)

    print("=" * 72)
    print(f"DATA CHARACTERISATION — {args.stem}")
    print("=" * 72)
    print(f"{len(meta)} rows, {len(set(keys))} unique compounds, "
          f"{len(set(scaf))} scaffolds")
    print(f"pharmacophore: {bits.mean():.1f} bits/molecule set of "
          f"{fp.shape[1]}, {int((bits == 0).sum())} molecule(s) set none")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.46, wspace=0.28, top=0.855)
    (axA, axB), (axC, axD) = axes

    # ---------------------------------------------- A: chemical space
    print(f"\nembedding {fp.shape[1]} pharmacophore bits -> PCA(50) -> t-SNE(2)...")
    Xp = PCA(n_components=min(50, fp.shape[1], len(fp) - 1),
             random_state=args.seed).fit_transform(fp.astype(np.float32))
    emb = TSNE(n_components=2, init="pca", random_state=args.seed,
               perplexity=min(30, max(5, len(fp) // 100))).fit_transform(Xp)
    style(axA)
    sc = axA.scatter(emb[:, 0], emb[:, 1], c=y, cmap=POTENCY_CMAP, s=7,
                     linewidths=0.3, edgecolors=SURFACE, zorder=3)
    cb = fig.colorbar(sc, ax=axA, fraction=0.045, pad=0.02)
    cb.set_label("pIC50", color=INK_2, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=2)
    cb.outline.set_visible(False)
    axA.set_xticklabels([]); axA.set_yticklabels([])
    axA.tick_params(length=0)
    titled(axA, "A · Chemical space",
           "t-SNE of the pharmacophore signature, coloured by potency. "
           "Orientation only — t-SNE distances are not metric.", width=52)

    # ---------------------------------------------- B: potency distribution
    style(axB)
    axB.hist(y, bins=40, color=ACCENT, edgecolor=SURFACE, linewidth=0.6,
             zorder=3)
    if cutoff is not None:
        axB.axvline(cutoff, color=WARN, linewidth=2, zorder=4)
        axB.annotate(f"active cutoff {cutoff}", xy=(cutoff, axB.get_ylim()[1]),
                     xytext=(4, -10), textcoords="offset points",
                     color=WARN, fontsize=8, va="top", fontweight="bold")
    axB.set_xlabel("pIC50", color=INK_2, fontsize=9)
    axB.set_ylabel("compounds", color=INK_2, fontsize=9)
    titled(axB, "B · What is being modelled",
           f"SD {y.std():.2f} log units — the global-mean null scores about "
           f"this, so it is the room any model has to beat.")

    # ---------------------------------------------- C: scaffold concentration
    counts = np.sort(np.array(list(
        {s: int((scaf == s).sum()) for s in set(scaf)}.values())))[::-1]
    cum = np.cumsum(counts) / counts.sum()
    style(axC)
    axC.plot(np.arange(1, len(cum) + 1), cum * 100, color=ACCENT, linewidth=2,
             zorder=3)
    marks = []
    for frac in (0.25, 0.50, 0.80):
        n = int(np.searchsorted(cum, frac) + 1)
        marks.append((frac, n))
        axC.plot([n], [frac * 100], "o", color=ACCENT, markersize=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        axC.annotate(f"{int(frac*100)}% of rows in {n} scaffolds",
                     xy=(n, frac * 100), xytext=(8, -3),
                     textcoords="offset points", color=INK_2, fontsize=8,
                     va="center")
    axC.set_xscale("log")
    axC.set_xlabel("scaffolds, ranked by size (log)", color=INK_2, fontsize=9)
    axC.set_ylabel("cumulative % of rows", color=INK_2, fontsize=9)
    axC.set_ylim(0, 103)
    titled(axC, "C · Scaffold concentration",
           "The fewer scaffolds needed to cover half the rows, the harsher a "
           "scaffold split is.", width=52)
    print("\nscaffold concentration:")
    for frac, n in marks:
        print(f"  {int(frac*100)}% of rows sit in the top {n} of "
              f"{len(counts)} scaffolds")

    # ---------------------------------------------- D: pharmacophore coverage
    style(axD)
    axD.hist(bits, bins=40, color=ACCENT, edgecolor=SURFACE, linewidth=0.6,
             zorder=3)
    n_zero = int((bits == 0).sum())
    if n_zero:
        axD.axvline(0, color=WARN, linewidth=2, zorder=4)
        axD.annotate(f"{n_zero} molecule(s) set NO bits",
                     xy=(0, axD.get_ylim()[1]), xytext=(6, -10),
                     textcoords="offset points", color=WARN, fontsize=8,
                     va="top", fontweight="bold")
    axD.set_xlabel(f"pharmacophore bits set (of {fp.shape[1]})",
                   color=INK_2, fontsize=9)
    axD.set_ylabel("compounds", color=INK_2, fontsize=9)
    titled(axD, "D · Does the representation cover these molecules?",
           f"Median {int(np.median(bits))} bits set. A molecule setting none "
           f"runs on descriptors alone.")

    fig.suptitle(f"Data characterisation — {args.stem}", color=INK,
                 fontsize=13, fontweight="bold", x=0.065, ha="left", y=0.985)
    out = f"{args.outdir}/chemspace_{args.stem}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwrote {out}")

    if n_zero > 0.05 * len(bits):
        print(f"\n⚠️  {100*n_zero/len(bits):.0f}% of molecules set no "
              f"pharmacophore bits. For those the model sees descriptors only, "
              f"and any claim about the pharmacophore representation does not "
              f"apply to them. Report the fraction.")


if __name__ == "__main__":
    main()
