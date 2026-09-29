#!/usr/bin/env python3
"""
Which feature blocks enter the model. One place, so every script agrees.

There are two blocks:

    fingerprint   The folded Gobbi 2D PHARMACOPHORE signature. Pairs and
                  triplets of pharmacophore features (donor, acceptor, aromatic,
                  hydrophobe, acidic, basic) binned by topological distance.
                  Scaffold-agnostic by construction, which is the reason this
                  pipeline uses it instead of ECFP4.
    descriptors   ~217 RDKit 2D descriptors. Whole-molecule properties —
                  logP, TPSA, atom counts, ring counts, charge indices.

Set it in the config:

    "features": { "blocks": "both" }          <- default
    "features": { "blocks": "fingerprint" }   <- ECFP4 only
    "features": { "blocks": "descriptors" }   <- descriptors only

Or override at the command line without touching the config, which is how you
run the ablation:

    python train_model.py --stem cox2_regression --config targets/cox2.json \
        --blocks fingerprint

WHY THE OPTION EXISTS
---------------------
It is an ablation, and a defensible one to report. The two blocks overlap: a
descriptor like NumHAcceptors counts the same acceptors the pharmacophore
signature encodes positionally, so part of the descriptor block is already
implied. The question "do the 217 descriptors add anything over the pharmacophore
signature alone?" has a real answer for your dataset, and running it costs one
flag.

It matters more here than it did with ECFP4, because a molecule that sets ZERO
pharmacophore bits is carried entirely by its descriptors. Check panel D of the
data characterisation figure: if that count is large, `--blocks fingerprint`
tells you what the pharmacophore signature is worth on its own, and `--blocks
descriptors` tells you how much of the result never needed it.

Three outcomes, all worth reporting:

* **Pharmacophore-only matches `both`** — the descriptor block is redundant
  here. Drop it: a simpler model and a smaller multiple-comparisons surface.
* **`both` beats fingerprint-only** — the descriptors carry something the bits
  do not. Say what: look at which descriptors the model ranks highly.
* **Pharmacophore-only BEATS `both`** — the descriptor block was adding noise, or
  a few descriptors correlate with something assay-related rather than
  molecular. Worth chasing, not worth hiding.

Report the margin over null for each, on the SAME splits and the same seed.
Comparing an ablation across different splits compares the splits.
"""

import sys

import numpy as np

CHOICES = ("both", "fingerprint", "descriptors")


def resolve(cfg, cli=None):
    """Config value, overridden by the CLI flag when given. Validated."""
    blocks = cli or (cfg.get("features") or {}).get("blocks") or "both"
    blocks = str(blocks).strip().lower()
    aliases = {"fp": "fingerprint", "fingerprints": "fingerprint",
               "pharm": "fingerprint", "pharm2d": "fingerprint",
               "pharmacophore": "fingerprint", "ph2d": "fingerprint",
               "desc": "descriptors", "descriptor": "descriptors",
               "all": "both", "fp+desc": "both"}
    blocks = aliases.get(blocks, blocks)
    if blocks not in CHOICES:
        sys.exit(f"ERROR: features.blocks is {blocks!r}. "
                 f"Choose one of: {', '.join(CHOICES)}.")
    return blocks


def assemble(fp, de, keep_cols, blocks):
    """Stack the requested blocks into one float32 matrix.

    fp         (n, n_bits) fingerprint array
    de         (n, n_desc) descriptor array
    keep_cols  descriptor column indices surviving `drop_descriptors`, or None
               when `de` has already been trimmed
    blocks     one of CHOICES
    """
    d = de if keep_cols is None else de[:, keep_cols]
    if blocks == "fingerprint":
        X = np.asarray(fp, dtype=np.float32)
    elif blocks == "descriptors":
        X = np.asarray(d, dtype=np.float32)
    else:
        X = np.hstack([np.asarray(fp, dtype=np.float32),
                       np.asarray(d, dtype=np.float32)])
    return np.ascontiguousarray(X, dtype=np.float32)


def width(n_bits, n_desc_kept, blocks):
    return {"fingerprint": n_bits,
            "descriptors": n_desc_kept,
            "both": n_bits + n_desc_kept}[blocks]


def describe(blocks, n_bits, n_desc_kept):
    return {
        "fingerprint": f"PHARMACOPHORE ONLY — {n_bits} folded Gobbi 2D bits, "
                       f"no descriptors",
        "descriptors": f"DESCRIPTORS ONLY — {n_desc_kept} RDKit descriptors, "
                       f"no pharmacophore",
        "both": f"{n_bits} pharmacophore bits + {n_desc_kept} descriptors "
                f"= {n_bits + n_desc_kept}",
    }[blocks]


def add_arg(ap):
    """Attach --blocks to an argparse parser, consistently worded."""
    ap.add_argument("--blocks", choices=CHOICES, default=None,
                    help="which feature blocks enter the model. Overrides "
                         "features.blocks in the config. Use this to run the "
                         "ablation without editing the config.")
