-- =====================================================================
-- W1 CENSUS — ChEMBL queries
-- =====================================================================
-- Run against a LOCAL ChEMBL release (SQLite or PostgreSQL).
-- Written to be release-portable: no vendor-specific functions, no
-- hardcoded ChEMBL IDs (targets resolve via UniProt accession or name
-- pattern, both of which are stable across releases).
--
-- REPRODUCIBILITY CONTRACT
--   * Pin the release. Record it in every output (query 00 reads it).
--   * Run query 01 FIRST. It fails loudly if any column these queries
--     depend on is missing or renamed in your release.
--   * No query mutates anything. All are SELECT.
--   * Every count is "distinct molregno" unless stated — never row count.
--     Row counts double-count compounds measured more than once and will
--     silently inflate the census.
--
-- Queries are split by the "-- @query: <name>" markers for run_census.py.
-- Each is also runnable standalone in any SQL client.
-- =====================================================================


-- @query: release_version
-- Which release is this? Goes in the report header and the paper.
SELECT name, creation_date, comments
FROM version;


-- NOTE: the schema assertion is NOT a query in this file. run_census.py does it
-- in Python before anything else runs, from its own REQUIRED table/column map,
-- using PRAGMA table_info() on SQLite and information_schema on PostgreSQL. It
-- exits non-zero if any column these queries depend on is missing. Keep
-- REQUIRED in sync with this file — if you add a column here, add it there.


-- =====================================================================
-- PART A — COX benchmark: how much survives curation?
-- =====================================================================

-- @query: cox_cascade
-- The curation cascade → Table 1 of the COC.
-- Each row is a filter stage; the count is DISTINCT COMPOUNDS remaining.
-- P23219 = human COX-1 (PTGS1); P35354 = human COX-2 (PTGS2).
WITH cox_targets AS (
    SELECT td.tid, cseq.accession,
           CASE cseq.accession WHEN 'P23219' THEN 'COX-1'
                               WHEN 'P35354' THEN 'COX-2' END AS cox
    FROM target_dictionary td
    JOIN target_components tc   ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219', 'P35354')
      AND td.target_type = 'SINGLE PROTEIN'
),
base AS (
    SELECT act.molregno, act.standard_type, act.standard_relation,
           act.pchembl_value, act.data_validity_comment,
           act.potential_duplicate, a.confidence_score, a.assay_type,
           t.cox
    FROM activities act
    JOIN assays a      ON act.assay_id = a.assay_id
    JOIN cox_targets t ON a.tid = t.tid
),
isoforms AS (SELECT DISTINCT cox FROM cox_targets),
stages AS (
              SELECT 1 AS stage_no, '0. all activity records'       AS stage
    UNION ALL SELECT 2, '1. + type in (IC50,Ki,Kd)'
    UNION ALL SELECT 3, '2. + relation = ''='''
    UNION ALL SELECT 4, '3. + pchembl_value present'
    UNION ALL SELECT 5, '4. + no validity comment'
    UNION ALL SELECT 6, '5. + not potential duplicate'
    UNION ALL SELECT 7, '6. + confidence_score >= 8'
    UNION ALL SELECT 8, '7. + assay_type = ''B'''
)
-- CROSS JOIN then correlated count, so EVERY (isoform, stage) pair gets a row
-- even when the count is zero. The earlier UNION ALL + GROUP BY version simply
-- omitted empty groups, which made a COX-1 block with fewer rows than the COX-2
-- block and silently misaligned the stages in Table 1.
SELECT i.cox, s.stage_no, s.stage,
       (SELECT COUNT(DISTINCT b.molregno) FROM base b
        WHERE b.cox = i.cox
          AND (s.stage_no < 2 OR b.standard_type IN ('IC50','Ki','Kd'))
          AND (s.stage_no < 3 OR b.standard_relation = '=')
          AND (s.stage_no < 4 OR b.pchembl_value IS NOT NULL)
          AND (s.stage_no < 5 OR b.data_validity_comment IS NULL)
          AND (s.stage_no < 6 OR (b.potential_duplicate = 0
                                  OR b.potential_duplicate IS NULL))
          AND (s.stage_no < 7 OR b.confidence_score >= 8)
          AND (s.stage_no < 8 OR b.assay_type = 'B')
       ) AS n_compounds
FROM isoforms i
CROSS JOIN stages s
ORDER BY i.cox, s.stage_no;
-- Stages 8 and 9 of Table 1 — STANDARDISATION and DEDUPLICATION — cannot be
-- done in SQL; they need RDKit. Add those two rows from the Python curation
-- step and label them so the cascade in the document is complete.


-- @query: cox_by_standard_type
-- Do NOT pool IC50 / Ki / Kd. This says how much you have of each, so the
-- "keep them separate" decision is made against real numbers.
WITH cox_targets AS (
    SELECT DISTINCT td.tid,
           CASE cseq.accession WHEN 'P23219' THEN 'COX-1'
                               WHEN 'P35354' THEN 'COX-2'
                               ELSE 'UNEXPECTED ACCESSION' END AS cox
    FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219','P35354') AND td.target_type = 'SINGLE PROTEIN'
)
SELECT t.cox, act.standard_type,
       COUNT(*) AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds
FROM activities act
JOIN assays a      ON act.assay_id = a.assay_id
JOIN cox_targets t ON a.tid = t.tid
WHERE act.pchembl_value IS NOT NULL
  AND act.standard_relation = '='
  AND act.data_validity_comment IS NULL
GROUP BY t.cox, act.standard_type
ORDER BY t.cox, n_compounds DESC;


-- @query: cox_assay_format
-- The celecoxib problem, quantified → Fig 2.
-- bao_format is the structured field; description is the messy truth.
-- Pull both: if bao_format is too coarse, you classify from description.
WITH cox_targets AS (
    SELECT DISTINCT td.tid,
           CASE cseq.accession WHEN 'P23219' THEN 'COX-1'
                               WHEN 'P35354' THEN 'COX-2'
                               ELSE 'UNEXPECTED ACCESSION' END AS cox
    FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219','P35354') AND td.target_type = 'SINGLE PROTEIN'
),
-- Classification order matters and is deliberate. 'cell-free' and 'acellular'
-- are tested BEFORE the bare '%cell%' catch, because '%cell%' matches
-- "cell-free" and would file a cell-free biochemical assay under 'cell-based'
-- — exactly backwards, in the figure that motivates contribution #1.
labelled AS (
    SELECT t.cox, a.bao_format, act.molregno, act.pchembl_value,
           CASE
             WHEN LOWER(a.description) LIKE '%whole blood%'  THEN 'whole blood'
             WHEN LOWER(a.description) LIKE '%cell-free%'
               OR LOWER(a.description) LIKE '%cell free%'
               OR LOWER(a.description) LIKE '%acellular%'    THEN 'cell-free / biochemical'
             WHEN LOWER(a.description) LIKE '%recombinant%'  THEN 'recombinant enzyme'
             WHEN LOWER(a.description) LIKE '%microsom%'     THEN 'microsomal'
             WHEN LOWER(a.description) LIKE '%platelet%'     THEN 'platelet'
             WHEN LOWER(a.description) LIKE '%cell%'         THEN 'cell-based'
             ELSE 'unclassified'
           END AS format_guess
    FROM activities act
    JOIN assays a      ON act.assay_id = a.assay_id
    JOIN cox_targets t ON a.tid = t.tid
    WHERE act.pchembl_value IS NOT NULL
      AND act.standard_type = 'IC50'
      AND act.standard_relation = '='
      AND act.data_validity_comment IS NULL
),
-- Collapse to ONE value per compound per stratum before averaging. Averaging
-- raw records lets a heavily-retested compound dominate its stratum: 40 records
-- for indomethacin at pchembl 6 outvote 20 other compounds at 8, and the
-- reported stratum mean is then a statement about retesting effort, not potency.
per_compound AS (
    SELECT cox, bao_format, format_guess, molregno,
           AVG(pchembl_value) AS cmpd_pchembl,
           COUNT(*)           AS cmpd_records
    FROM labelled
    GROUP BY cox, bao_format, format_guess, molregno
)
SELECT cox, bao_format, format_guess,
       COUNT(*)              AS n_compounds,
       SUM(cmpd_records)     AS n_records,
       AVG(cmpd_pchembl)     AS mean_pchembl_per_compound,
       MIN(cmpd_pchembl)     AS min_pchembl,
       MAX(cmpd_pchembl)     AS max_pchembl
FROM per_compound
GROUP BY cox, bao_format, format_guess
ORDER BY cox, n_compounds DESC;
-- Medians and the boxplot itself are computed in Python from the per-compound
-- values (no portable SQL median). Fig 2 must be drawn per compound, not per
-- record, for the same reason.
-- Read the 'unclassified' fraction: if it dominates, the format-held-out
-- split has no usable strata and contribution #1 needs a rethink. Check
-- this in W1, not W3.


-- @query: cox_relation_censoring
-- How much COX data is CENSORED, and which way?
--
-- '=' is a point value. '>' means "no 50% inhibition up to the highest dose
-- tested" — the compound is INACTIVE and the real IC50 is somewhere above the
-- number. '<' means more potent than the lowest dose tested (rare for IC50).
--
-- Why this matters more than it looks: the '>' records ARE the inactives. The
-- IC50 path filters to '=' only, which is correct for regression and WRONG for
-- anything that needs a negative class. Filter to '=' and the training set is
-- enriched for potent compounds by construction — so EF1% is then computed on a
-- set with almost no true inactives in it, and the enrichment number means
-- nothing. Read this table before fixing the EF1% actives threshold.
WITH cox_targets AS (
    SELECT DISTINCT td.tid,
           CASE cseq.accession WHEN 'P23219' THEN 'COX-1'
                               WHEN 'P35354' THEN 'COX-2'
                               ELSE 'UNEXPECTED ACCESSION' END AS cox
    FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219','P35354') AND td.target_type = 'SINGLE PROTEIN'
)
SELECT t.cox, act.standard_type, act.standard_relation,
       CASE act.standard_relation
         WHEN '='  THEN 'point value — usable for regression'
         WHEN '>'  THEN 'right-censored — INACTIVE, true value above this'
         WHEN '>=' THEN 'right-censored'
         WHEN '<'  THEN 'left-censored — more potent than lowest dose'
         WHEN '<=' THEN 'left-censored'
         WHEN '~'  THEN 'approximate — depositor flagged as imprecise'
         ELSE 'other / unrecorded'
       END AS meaning,
       COUNT(*)                     AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds_overlapping
FROM activities act
JOIN assays a      ON act.assay_id = a.assay_id
JOIN cox_targets t ON a.tid = t.tid
WHERE act.standard_type IN ('IC50','Ki','Kd')
  AND act.standard_value IS NOT NULL
  AND act.data_validity_comment IS NULL
GROUP BY t.cox, act.standard_type, act.standard_relation
ORDER BY t.cox, act.standard_type, n_records DESC;
-- ⚠️ n_compounds_overlapping: a compound with one '=' and one '>' record is in
-- both rows. Use n_records for fractions.
--
-- Decision rule once you have the numbers:
--   censored < ~5%   -> drop for regression, note it, move on
--   censored 5-20%   -> drop for regression BUT keep as the inactive class for
--                       classification and EF1%. State both uses.
--   censored > ~20%   -> dropping biases the set badly. Censored (Tobit)
--                       regression stops being optional.


-- @query: cox_year_distribution
-- Temporal split feasibility. A temporal split needs enough mass on both
-- sides of a cut year.
WITH cox_targets AS (
    SELECT td.tid FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219','P35354') AND td.target_type = 'SINGLE PROTEIN'
)
SELECT d.year, COUNT(DISTINCT act.molregno) AS n_compounds
FROM activities act
JOIN assays a      ON act.assay_id = a.assay_id
JOIN cox_targets t ON a.tid = t.tid
JOIN docs d        ON a.doc_id = d.doc_id
WHERE act.pchembl_value IS NOT NULL AND d.year IS NOT NULL
GROUP BY d.year
ORDER BY d.year;


-- =====================================================================
-- PART B — Stereochemistry audit  (the new W1 decision)
-- =====================================================================

-- @query: cox_stereo_status
-- ChEMBL's own chirality flag:
--    0 = racemic mixture, 1 = single stereoisomer, 2 = achiral, -1 = unknown
-- This is the field that decides useChirality. If most of the set is 0 or
-- -1, asserting stereo in the fingerprint claims precision the assay never
-- had — the same error as training on ug/mL.
WITH cox_targets AS (
    SELECT DISTINCT td.tid,
           CASE cseq.accession WHEN 'P23219' THEN 'COX-1'
                               WHEN 'P35354' THEN 'COX-2'
                               ELSE 'UNEXPECTED ACCESSION' END AS cox
    FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219','P35354') AND td.target_type = 'SINGLE PROTEIN'
)
SELECT t.cox, md.chirality,
       CASE md.chirality
         WHEN 0  THEN 'racemic mixture'
         WHEN 1  THEN 'single stereoisomer'
         WHEN 2  THEN 'achiral'
         WHEN -1 THEN 'unknown'
         ELSE 'UNDOCUMENTED VALUE — investigate'
       END AS meaning,
       COUNT(DISTINCT act.molregno) AS n_compounds
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN cox_targets t          ON a.tid = t.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
WHERE act.pchembl_value IS NOT NULL
  AND act.standard_type IN ('IC50','Ki','Kd')
  AND act.standard_relation = '='
  AND act.data_validity_comment IS NULL
GROUP BY t.cox, md.chirality
ORDER BY t.cox, n_compounds DESC;


-- @query: cox_smiles_for_stereo_check
-- SMILES for the curated COX set, to cross-check ChEMBL's chirality flag
-- against what RDKit actually finds in the structure. Disagreements are a
-- curation-log finding, not a bug.
WITH cox_targets AS (
    SELECT td.tid FROM target_dictionary td
    JOIN target_components tc     ON td.tid = tc.tid
    JOIN component_sequences cseq ON tc.component_id = cseq.component_id
    WHERE cseq.accession IN ('P23219','P35354') AND td.target_type = 'SINGLE PROTEIN'
)
SELECT DISTINCT md.chembl_id, md.chirality, cstr.canonical_smiles
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN cox_targets t          ON a.tid = t.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
JOIN compound_structures cstr ON md.molregno = cstr.molregno
WHERE act.pchembl_value IS NOT NULL
  AND act.standard_type IN ('IC50','Ki','Kd')
  AND act.standard_relation = '='
  AND act.data_validity_comment IS NULL
  AND cstr.canonical_smiles IS NOT NULL;


-- =====================================================================
-- PART C — AMR census
-- =====================================================================

-- @query: all_targets_data_census
-- COUNT EVERYTHING. Every single-protein target in the release, put through the
-- same curation cascade as COX, with no name or organism filter anywhere.
--
-- This is the denominator. Without it, "target X has 340 compounds" is a number
-- with nothing to compare it to. With it, you can say where X sits in the whole
-- distribution, and where the n_min line falls across all of ChEMBL.
--
-- Two extra things are counted here on purpose:
--
--   n_approved  - compounds against this target that reached approved-drug
--                 status (max_phase = 4). This is the "already solved" signal.
--                 A target with lots of data AND an approved drug is a solved
--                 problem. A target with lots of data and NO approved drug, or
--                 one where resistance has broken the drug, is the interesting
--                 cell. Ranking by compound count alone recommends solved
--                 problems, which is the circularity trap.
--
--   pchembl spread + n_assays + n_docs - because raw count is a bad proxy for
--                 modellability. 2000 compounds from one paper, one scaffold
--                 series, half a log unit of potency is worse than 300 across
--                 twenty papers and four orders of magnitude. Scaffold counts
--                 need RDKit, so they are computed in Python from the CSV; these
--                 columns are the cheap SQL-side version of the same question.
--
-- Runs against the FULL release. On the COX subset it returns only COX.
SELECT td.chembl_id,
       td.pref_name,
       td.organism,
       COUNT(DISTINCT act.molregno)    AS n_compounds,
       COUNT(*)                        AS n_records,
       COUNT(DISTINCT act.assay_id)    AS n_assays,
       COUNT(DISTINCT a.doc_id)        AS n_docs,
       COUNT(DISTINCT a.bao_format)    AS n_assay_formats,
       COUNT(DISTINCT CASE WHEN md.max_phase = 4 THEN act.molregno END)
                                       AS n_approved,
       -- ⚠️ GRAIN: these four are per-RECORD, not per-compound. A heavily
       -- retested compound pulls the mean toward its own potency. That is
       -- acceptable here because these columns are only a coarse "is there any
       -- dynamic range at this target" signal for ranking, and computing them
       -- per-compound across all 24M activities is an expensive extra pass.
       -- Do NOT quote mean_pchembl_per_record as a target's potency, and do not
       -- put it in a figure. cox_assay_format computes the per-compound version
       -- for the one target where the number actually appears in Fig 2.
       MIN(act.pchembl_value)          AS min_pchembl_per_record,
       MAX(act.pchembl_value)          AS max_pchembl_per_record,
       AVG(act.pchembl_value)          AS mean_pchembl_per_record,
       MAX(act.pchembl_value) - MIN(act.pchembl_value) AS pchembl_range_per_record
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN target_dictionary td   ON a.tid = td.tid
JOIN molecule_dictionary md ON act.molregno = md.molregno
WHERE td.target_type = 'SINGLE PROTEIN'
  AND act.standard_type IN ('IC50','Ki','Kd')
  AND act.standard_relation = '='
  AND act.pchembl_value IS NOT NULL
  AND act.data_validity_comment IS NULL
  AND (act.potential_duplicate = 0 OR act.potential_duplicate IS NULL)
  AND a.confidence_score >= 8
  AND a.assay_type = 'B'      -- MUST match cox_cascade's final stage. Without
                              -- this the denominator is drawn from a LARGER set
                              -- than Table 1's final row, so every AMR-vs-COX
                              -- comparison would compare two different cascades.
GROUP BY td.chembl_id, td.pref_name, td.organism
HAVING COUNT(DISTINCT act.molregno) >= 25
ORDER BY n_compounds DESC;
-- Read it like this:
--   1. Where does the n_min line fall in this distribution? If n_min = 500 and
--      the median target has 60 compounds, most of ChEMBL is unmodellable and
--      that is a headline finding, not a disappointment.
--   2. Filter to bacterial organisms and look again. That is your AMR shortlist,
--      derived rather than guessed.
--   3. Sort by n_compounds DESC and n_approved = 0. High data, no approved drug.
--      That is the shortlist worth arguing about.


-- @query: amr_target_discovery
-- Discover, don't presume. Lists every single-protein target whose name
-- matches the candidate families, with compound counts, so the shortlist
-- comes from the data rather than from a guess about what exists.
SELECT td.chembl_id, td.pref_name, td.organism, td.target_type,
       COUNT(DISTINCT act.molregno) AS n_compounds_any,
       COUNT(DISTINCT CASE WHEN act.pchembl_value IS NOT NULL
                           THEN act.molregno END) AS n_compounds_pchembl
FROM target_dictionary td
JOIN assays a     ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
WHERE td.target_type = 'SINGLE PROTEIN'
  AND (
       LOWER(td.pref_name) LIKE '%dihydrofolate reductase%'
    OR LOWER(td.pref_name) LIKE '%dna gyrase%'
    OR LOWER(td.pref_name) LIKE '%topoisomerase iv%'
    OR LOWER(td.pref_name) LIKE '%udp-3-o%'          -- LpxC
    OR LOWER(td.pref_name) LIKE '%lpxc%'
    OR LOWER(td.pref_name) LIKE '%enoyl%reductase%'  -- FabI
    OR LOWER(td.pref_name) LIKE '%fabi%'
    OR LOWER(td.pref_name) LIKE '%lactamase%'
    OR LOWER(td.pref_name) LIKE '%mur%ligase%'
    OR LOWER(td.pref_name) LIKE '%udp-n-acetyl%'     -- MurA/MurB
  )
GROUP BY td.chembl_id, td.pref_name, td.organism, td.target_type
HAVING COUNT(DISTINCT act.molregno) >= 20
ORDER BY n_compounds_pchembl DESC;


-- @query: amr_biochemical_curated
-- Same filter cascade as COX, applied to every bacterial single-protein
-- target. This is the number that goes on Fig 4 — curated, not raw.
SELECT td.chembl_id, td.pref_name, td.organism,
       COUNT(DISTINCT act.molregno) AS n_curated_compounds
FROM target_dictionary td
JOIN assays a       ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
WHERE td.target_type = 'SINGLE PROTEIN'
  AND act.standard_type IN ('IC50','Ki','Kd')
  AND act.standard_relation = '='
  AND act.pchembl_value IS NOT NULL
  AND act.data_validity_comment IS NULL
  AND (act.potential_duplicate = 0 OR act.potential_duplicate IS NULL)
  AND a.confidence_score >= 8
  AND a.assay_type = 'B'      -- same reason as all_targets_data_census: this
                              -- must be the SAME cascade as cox_cascade stage 8
                              -- or Fig 4 compares unlike numbers.
  AND (
       LOWER(td.organism) LIKE '%escherichia%'   OR LOWER(td.organism) LIKE '%staphylococcus%'
    OR LOWER(td.organism) LIKE '%pseudomonas%'   OR LOWER(td.organism) LIKE '%acinetobacter%'
    OR LOWER(td.organism) LIKE '%klebsiella%'    OR LOWER(td.organism) LIKE '%mycobacterium%'
    OR LOWER(td.organism) LIKE '%enterococcus%'  OR LOWER(td.organism) LIKE '%streptococcus%'
  )
GROUP BY td.chembl_id, td.pref_name, td.organism
HAVING COUNT(DISTINCT act.molregno) >= 50
ORDER BY n_curated_compounds DESC;


-- @query: amr_mic_census
-- Phenotypic path. Target type is ORGANISM, not SINGLE PROTEIN.
-- NOTE the deliberate absence of confidence_score >= 8 — see next query.
SELECT td.organism, td.chembl_id, td.pref_name,
       COUNT(*) AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds
FROM target_dictionary td
JOIN assays a       ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
WHERE td.target_type = 'ORGANISM'
  AND act.standard_type = 'MIC'
  AND act.standard_value IS NOT NULL
GROUP BY td.organism, td.chembl_id, td.pref_name
HAVING COUNT(DISTINCT act.molregno) >= 50
ORDER BY n_compounds DESC;


-- @query: mic_confidence_score_trap
-- Proof, not assertion. If the IC50 filter (confidence_score >= 8) were
-- reused on the MIC path, this is what would survive. Expect the high-score
-- buckets to be near-empty: organism assays score low BY DESIGN.
-- Put this table in the curation log.
-- Filters match amr_mic_census exactly (including standard_value IS NOT NULL)
-- so the buckets here sum to that query's total and the fractions are real.
--
-- ⚠️ n_compounds columns OVERLAP between buckets: one compound tested in a
-- score-3 assay and a score-9 assay appears in both rows, so the column does
-- NOT sum to the distinct total. n_records DOES sum. Use n_records for any
-- "what fraction would survive" arithmetic; the last column gives the honest
-- distinct total to divide by.
SELECT a.confidence_score,
       COUNT(*)                     AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds_overlapping,
       (SELECT COUNT(DISTINCT act2.molregno)
        FROM target_dictionary td2
        JOIN assays a2       ON td2.tid = a2.tid
        JOIN activities act2 ON a2.assay_id = act2.assay_id
        WHERE td2.target_type = 'ORGANISM'
          AND act2.standard_type = 'MIC'
          AND act2.standard_value IS NOT NULL) AS n_compounds_total_distinct
FROM target_dictionary td
JOIN assays a       ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
WHERE td.target_type = 'ORGANISM'
  AND act.standard_type = 'MIC'
  AND act.standard_value IS NOT NULL
GROUP BY a.confidence_score
ORDER BY a.confidence_score;


-- @query: mic_units_distribution
-- MIC is reported as mass concentration. This sizes the ug/mL -> molar
-- conversion job and flags anything that cannot be converted.
SELECT act.standard_units, act.standard_type,
       COUNT(*) AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds
FROM target_dictionary td
JOIN assays a       ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
WHERE td.target_type = 'ORGANISM'
  AND act.standard_type IN ('MIC','MIC50','MIC90')
GROUP BY act.standard_units, act.standard_type
ORDER BY n_records DESC;


-- @query: mic_relation_censoring
-- How much MIC data is censored (> or <)? If it is a large fraction,
-- dropping it biases the set toward potent compounds and censored
-- regression stops being optional.
-- Filters match amr_mic_census so these rows sum to that query's record total.
SELECT act.standard_relation, COUNT(*) AS n_records
FROM target_dictionary td
JOIN assays a       ON td.tid = a.tid
JOIN activities act ON a.assay_id = act.assay_id
WHERE td.target_type = 'ORGANISM'
  AND act.standard_type = 'MIC'
  AND act.standard_value IS NOT NULL
GROUP BY act.standard_relation
ORDER BY n_records DESC;


-- =====================================================================
-- PART D — the half-hour query that could reshape Phase II
-- =====================================================================

-- @query: resistance_variant_coverage
-- Does ChEMBL annotate target VARIANTS (resistance mutations) well enough
-- to support a variant-aware model? If yes, "known target, broken by
-- resistance" becomes a data-rich AND unsolved problem — which is the
-- escape from the circularity that ligand-based methods are trapped in.
SELECT td.pref_name, td.organism,
       vs.mutation, vs.accession,
       COUNT(*) AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds
FROM activities act
JOIN assays a               ON act.assay_id = a.assay_id
JOIN target_dictionary td   ON a.tid = td.tid
JOIN variant_sequences vs   ON a.variant_id = vs.variant_id
WHERE act.standard_value IS NOT NULL
  -- Restricted to bacterial targets. Without this, human targets with annotated
  -- variants (kinase gatekeeper mutants, above all) dominate the table and the
  -- answer to "does ChEMBL annotate RESISTANCE mutations" is inflated by
  -- oncology data that has nothing to do with the question.
  AND (
       LOWER(td.organism) LIKE '%escherichia%'   OR LOWER(td.organism) LIKE '%staphylococcus%'
    OR LOWER(td.organism) LIKE '%pseudomonas%'   OR LOWER(td.organism) LIKE '%acinetobacter%'
    OR LOWER(td.organism) LIKE '%klebsiella%'    OR LOWER(td.organism) LIKE '%mycobacterium%'
    OR LOWER(td.organism) LIKE '%enterococcus%'  OR LOWER(td.organism) LIKE '%streptococcus%'
    OR LOWER(td.organism) LIKE '%salmonella%'    OR LOWER(td.organism) LIKE '%enterobacter%'
    OR LOWER(td.organism) LIKE '%neisseria%'     OR LOWER(td.organism) LIKE '%haemophilus%'
  )
GROUP BY td.pref_name, td.organism, vs.mutation, vs.accession
HAVING COUNT(DISTINCT act.molregno) >= 10
ORDER BY n_compounds DESC;


-- @query: variant_annotation_rate
-- Blunter version of the same question: what fraction of bacterial-target
-- activity records carry ANY variant annotation? A low number means the
-- resistance framing needs a different data source.
-- ⚠️ Read the record column, not the compound column, for the fraction.
-- A compound with one annotated and one unannotated record belongs to BOTH
-- groups, so n_compounds_overlapping does not sum to the distinct total and
-- "percent annotated" computed from it is not a fraction of anything.
-- n_records partitions cleanly. The distinct total is given for division.
SELECT CASE WHEN a.variant_id IS NULL THEN 'no variant annotation'
            ELSE 'variant annotated' END AS variant_status,
       COUNT(*)                     AS n_records,
       COUNT(DISTINCT act.molregno) AS n_compounds_overlapping,
       (SELECT COUNT(DISTINCT act2.molregno)
        FROM activities act2
        JOIN assays a2             ON act2.assay_id = a2.assay_id
        JOIN target_dictionary td2 ON a2.tid = td2.tid
        WHERE LOWER(td2.pref_name) LIKE '%lactamase%'
          AND act2.standard_value IS NOT NULL) AS n_compounds_total_distinct
FROM activities act
JOIN assays a             ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
WHERE LOWER(td.pref_name) LIKE '%lactamase%'
  AND act.standard_value IS NOT NULL
GROUP BY variant_status;
