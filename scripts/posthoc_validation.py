"""
posthoc_validation.py -- two-sided post-hoc validity audit of existing IFT tools.

Applies CAFT's constraint validator to the IDIs saved by existing fairness
testing tools, producing the RQ1 (soundness of prior evidence) and RQ2
(actionable yield) tables. The existing tools are used only as IDI *sources*;
CAFT supplies the constraint set C_tau and the two-sided validity oracle.

For each (tool, dataset, classifier, sensitive attribute) cell it:
  - loads the tool's saved disc_inputs;
  - deduplicates instances that differ only in the protected attributes. Some
    tools store both members of each IDI pair, which are the same IDI for
    validity purposes; collapsing them avoids double-counting. Raw and unique
    counts are both reported;
  - CONFIRMS each unique instance under the current trained model: the row is
    an IDI iff its predicted label is not constant as the protected attributes
    range over their full domain (the standard individual-fairness oracle;
    this drops stale rows generated against an older model snapshot);
  - ONE-SIDED validity (Kitamura-style): the instance itself satisfies C_tau;
  - TWO-SIDED validity (CAFT's standard): at least two constraint-valid
    protected-attribute variants receive different predictions. This is the
    validity guard CAFT enforces during search, applied here post-hoc.

Validation reuses the CAFT class itself (constraint extraction, partitioned
rule arrays, vectorised violation counting), so baseline outputs are judged
by machinery byte-identical to CAFT's in-search guard.

Timing: TTD (time to 1,000 raw IDIs) is interpolated from the tool's saved
cumulative_efficiency series. VTT (time to 1,000 two-sided-valid IDIs) is
estimated under a uniform-validity assumption: if a fraction v of a tool's
IDIs are valid, 1,000 valid IDIs require 1,000/v raw discoveries, and VTT is
the interpolated time at which the tool had accumulated that many. Saved row
order is not guaranteed to be discovery order for all tools, so the uniform
estimate is used for every tool. -1 means the milestone was never reached
within the tool's recorded run.

Run from the repo root:
    python scripts/posthoc_validation.py
Optional flags:
    --tools THEMIS,SBFT,ExpGA,...   --datasets census,bank,...
    --classifiers lr,rf,dnn         --min_score 0.97
    --include_intersectional        (also audit age,race / age,race,sex ... files)
    --refresh ExpGA,AFT             (recompute only these cells)
    --refresh all                   (recompute everything)
Outputs CSV + LaTeX under results/posthoc/.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Resolve the repo root and run from there so relative dataset/model paths resolve.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf

from src.utils.helpers import get_config_dict, load_disc_inputs
from src.caft.caft import CAFT

# Baseline set agreed for the paper: random (THEMIS), adaptive random (AFT),
# evolutionary (ExpGA), gradient-guided (GRFT, MAFT; MAFT is DNN-only),
# realism-aware RL (MAEFT). LOEFT is excluded (unpublished). CAFT itself is
# audited as the 100%-VDR sanity check.
TOOLS = ['THEMIS', 'AFT', 'ExpGA', 'GRFT', 'MAFT', 'MAEFT', 'CAFT']
CLASSIFIERS = ['lr', 'rf', 'dnn']
DATASETS = ['census', 'bank', 'credit', 'compas', 'meps']
VARIANT_BATCH = 200_000          # max rows*M per batched predict/validate pass
MILESTONE = 1000                 # IDI count for the TTD / VTT milestones
OUT_DIR = os.path.join('results', 'posthoc')


def load_model(dataset, clf):
    """Load a trained model; DNN without optimiser state, sklearn unpickled."""
    if clf == 'dnn':
        return tf.keras.models.load_model(
            os.path.join('models', dataset, 'dnn.keras'), compile=False)
    return joblib.load(os.path.join('models', dataset, f'{clf}.pkl'))


def dedup_idis(idis, protected_cols):
    """Collapse rows identical except in the protected columns.

    x and x' of an IDI pair differ only on protected attributes and are the
    same discrimination scenario. Returns unique rows in first-seen order.
    """
    idis = np.asarray(idis)
    if len(idis) == 0:
        return idis
    canon = idis.copy()
    canon[:, protected_cols] = 0
    _, idx = np.unique(canon, axis=0, return_index=True)
    return idis[np.sort(idx)]


def validate_rows(caft, rows):
    """Confirm + validate a batch of saved IDI rows under CAFT's oracle.

    Returns three boolean arrays aligned to `rows`:
      confirmed  -- predictions vary across the full protected domain
                    (raw IDI under the current model, no constraint check);
      valid_orig -- the reported instance itself satisfies C_tau
                    (instance-only validity, the Kitamura et al. check);
      valid_pair -- >= 2 constraint-valid variants disagree (the
                    counterfactual side has a valid disagreeing pair).

    Two-sided validity, matching CAFT's valid_disc_profiles, is the
    conjunction valid_orig & valid_pair: the reported instance is valid AND a
    valid counterfactual disagrees with it. It is formed by the caller.
    """
    rows = np.asarray(rows).astype(np.int64)
    n = len(rows)
    M = caft._M
    confirmed = np.zeros(n, dtype=bool)
    valid_orig = np.zeros(n, dtype=bool)
    valid_pair = np.zeros(n, dtype=bool)

    chunk = max(1, VARIANT_BATCH // M)
    for start in range(0, n, chunk):
        r = rows[start:start + chunk]
        k = len(r)

        # All M protected-attribute variants, one batched predict.
        batch_inputs, _, _ = caft.similar_set_(r)
        preds = caft.model.predict(batch_inputs).reshape(k, M)

        # Partitioned vectorised validation, identical to the in-search guard.
        shared_viol = caft._violations_vec(
            r, caft._shared_rule_arrs, caft._shared_domain_arrs)
        variant_viol = caft._violations_vec(
            batch_inputs, caft._variant_rule_arrs, caft._variant_domain_arrs)
        total_viol = shared_viol[np.arange(k * M) // M] + variant_viol
        valid_matrix = (total_viol == 0).reshape(k, M)

        confirmed[start:start + k] = preds.max(axis=1) != preds.min(axis=1)

        # One-sided: the saved row, at its own protected values, is valid.
        own_idx = np.array([
            caft._comb_to_idx.get(
                tuple(int(v) for v in r[i, caft.protected_attribs]), 0)
            for i in range(k)
        ])
        valid_orig[start:start + k] = valid_matrix[np.arange(k), own_idx]

        # Two-sided: >= 2 valid variants and their predictions disagree.
        masked_max = np.where(valid_matrix, preds, -np.inf).max(axis=1)
        masked_min = np.where(valid_matrix, preds, np.inf).min(axis=1)
        enough = valid_matrix.sum(axis=1) >= 2
        valid_pair[start:start + k] = enough & (masked_max != masked_min)

    return confirmed, valid_orig, valid_pair


def time_at_count(tool, ds, sname, clf, target):
    """Interpolate the wall-clock time at which a tool's cumulative disc count
    reached `target`, from its saved cumulative_efficiency series
    (columns: time, iteration, tot_inputs, disc_inputs). Returns -1 if the
    series never reaches the target, np.nan if no series was saved."""
    fpath = os.path.join('test_data', tool, ds,
                         f'{clf}_{sname}_cumulative_efficiency.parquet')
    if not os.path.exists(fpath):
        return np.nan
    ce = pd.read_parquet(fpath).to_numpy()
    if ce.shape[0] == 0 or ce.shape[1] < 2:
        return np.nan
    times, counts = ce[:, 0], ce[:, -1]
    if counts.max() < target:
        return -1.0
    return float(np.interp(target, counts, times))


def sens_configs(config, include_intersectional):
    """Yield (sname, sensitive_params) pairs matching the saved file naming:
    names sorted alphabetically and joined with ','."""
    items = sorted(config.sens_name.items(), key=lambda kv: kv[1])
    singles = [(name, [sp]) for sp, name in items]
    for s in singles:
        yield s
    if include_intersectional and len(items) > 1:
        import itertools
        for r in range(2, len(items) + 1):
            for combo in itertools.combinations(items, r):
                names = sorted(n for _, n in combo)
                yield ','.join(names), [sp for sp, _ in combo]


def to_latex(df, path, caption, label, float_format='%.1f'):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(df.to_latex(index=False, escape=True, na_rep='--',
                            float_format=float_format,
                            caption=caption, label=label, position='ht'))


def main():
    ap = argparse.ArgumentParser(
        description='Two-sided post-hoc validity audit of IFT tool outputs.')
    ap.add_argument('--tools', default=','.join(TOOLS))
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--classifiers', default=','.join(CLASSIFIERS))
    ap.add_argument('--min_score', type=float, default=0.97,
                    help='constraint confidence threshold tau (CAFT default)')
    ap.add_argument('--include_intersectional', action='store_true',
                    help='also audit multi-attribute configs (age,race ...)')
    ap.add_argument('--min_rows', type=int, default=30,
                    help='skip a cell with fewer unique saved IDIs than this')
    ap.add_argument('--refresh', default='',
                    help="comma-separated tools whose recorded cells are "
                         "discarded and recomputed; 'all' recomputes everything")
    args = ap.parse_args()

    tools = [t.strip() for t in args.tools.split(',') if t.strip()]
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(',') if c.strip()]
    refresh = {t.strip() for t in args.refresh.split(',') if t.strip()}
    refresh_all = 'all' in refresh

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = get_config_dict()

    long_path = os.path.join(OUT_DIR, 'posthoc_validation.csv')
    # Resume: keep previously recorded cells (except tools being refreshed) and
    # checkpoint after every cell, so a long run is crash-safe and re-runs only
    # compute what is missing.
    records, done = [], set()
    if os.path.exists(long_path) and not refresh_all:
        prev = pd.read_csv(long_path)
        prev = prev[~prev.tool.isin(refresh)]
        records = prev.to_dict('records')
        done = {(r['dataset'], r['attribute'], r['classifier'], r['tool'])
                for r in records}
        print(f"resuming: {len(records)} cells kept"
              f"{f', refreshing {sorted(refresh)}' if refresh else ''}",
              flush=True)
    elif refresh_all:
        print("refresh all: recomputing every cell from scratch", flush=True)

    model_cache = {}

    def get_model(ds, clf):
        if (ds, clf) not in model_cache:
            model_cache[(ds, clf)] = load_model(ds, clf)
        return model_cache[(ds, clf)]

    for ds in datasets:
        config = cfg[ds]
        for sname, sps in sens_configs(config, args.include_intersectional):
            prot_cols = [sp - 1 for sp in sps]
            for clf in classifiers:
                mfile = 'dnn.keras' if clf == 'dnn' else f'{clf}.pkl'
                if not os.path.exists(os.path.join('models', ds, mfile)):
                    continue
                caft = None   # built lazily on the first tool with data
                for tool in tools:
                    if (ds, sname, clf, tool) in done:
                        continue
                    fpath = os.path.join('test_data', tool, ds,
                                         f'{clf}_{sname}_disc_inputs.parquet')
                    if not os.path.exists(fpath):
                        continue
                    raw = load_disc_inputs(tool, ds, sname, clf)
                    idis = dedup_idis(raw, prot_cols)
                    if len(idis) < args.min_rows:
                        print(f"  {tool:6s}: only {len(idis)} unique rows, SKIP",
                              flush=True)
                        continue
                    if caft is None:
                        # Constraint set depends on (dataset, tau); the rule
                        # partition depends on the protected attributes; the
                        # model on the classifier. Built once per cell, shared
                        # across all tools audited for that cell. The GA inside
                        # is never run; pop size is minimal to keep init cheap.
                        caft = CAFT(config, get_model(ds, clf), clf, sps,
                                    constraint_mode='none',
                                    min_score=args.min_score,
                                    population_size=10,
                                    seed_strategy='random',
                                    lambda_redundancy=0.0)
                        print(f"\n[{ds}/{sname}/{clf}] M={caft._M} variants",
                              flush=True)

                    confirmed, valid_orig, valid_pair = validate_rows(caft, idis)
                    n_conf = int(confirmed.sum())        # raw IDIs (A) confirmed under model
                    if n_conf == 0:
                        print(f"  {tool:6s}/{clf:3s}: uniq={len(idis):6d} "
                              f"0 confirmed under current model, SKIP (stale?)",
                              flush=True)
                        continue
                    # Three nested metrics, all as rates over confirmed raw
                    # IDIs (the classic count):
                    #   raw     (A): confirmed disagreement, no constraint check;
                    #   partial (B): raw IDI whose original instance is valid
                    #                (Kitamura et al. 2024 validity check);
                    #   true    (C): original valid AND a valid counterfactual
                    #                disagrees (our two-sided validity).
                    true_valid = valid_orig & valid_pair
                    n_partial = int((valid_orig & confirmed).sum())
                    n_true    = int((true_valid & confirmed).sum())
                    vdr_partial = 100.0 * n_partial / n_conf
                    vdr_true    = 100.0 * n_true / n_conf

                    ttd = time_at_count(tool, ds, sname, clf, MILESTONE)
                    # Uniform-validity VTT estimate, based on the true-valid fraction.
                    if n_true > 0 and not np.isnan(ttd):
                        needed = MILESTONE / (n_true / n_conf)
                        vtt = time_at_count(tool, ds, sname, clf, needed)
                    else:
                        vtt = -1.0 if not np.isnan(ttd) else np.nan

                    rec = {
                        'dataset': ds, 'attribute': sname, 'classifier': clf,
                        'tool': tool, 'M': caft._M,
                        'n_unique': len(idis),
                        'n_raw': n_conf,      # confirmed classic IDIs
                        'confirm_rate': round(100.0 * n_conf / len(idis), 1),
                        'n_partial': n_partial,
                        'vdr_partial': round(vdr_partial, 2),
                        'n_true': n_true,
                        'vdr_true': round(vdr_true, 2),
                        'ttd_1000': round(ttd, 2) if not np.isnan(ttd) else np.nan,
                        'vtt_1000_est': round(vtt, 2) if not np.isnan(vtt) else np.nan,
                    }
                    records.append(rec)
                    pd.DataFrame(records).to_csv(long_path, index=False)
                    print(f"  {tool:6s}/{clf:3s}: raw={n_conf:7d} "
                          f"({rec['confirm_rate']:5.1f}% conf) "
                          f"partial={vdr_partial:5.2f}% true={vdr_true:5.2f}% "
                          f"VTT~{rec['vtt_1000_est']}", flush=True)

    long = pd.DataFrame(records)
    print(f"\nwrote {long_path}  ({len(long)} cells)")
    if long.empty:
        print("no cells produced; check test_data paths")
        return

    long['config'] = long.dataset + '/' + long.attribute

    # RQ1: true (two-sided) VDR, tool x (classifier, config). The headline
    # soundness table: what fraction of each tool's classic IDIs survives our
    # two-sided validity check.
    rq1 = long.pivot_table(index=['classifier', 'config'], columns='tool',
                           values='vdr_true')
    rq1 = rq1.reset_index()
    rq1.to_csv(os.path.join(OUT_DIR, 'posthoc_rq1_vdr.csv'), index=False)
    to_latex(rq1, os.path.join(OUT_DIR, 'posthoc_rq1_vdr.tex'),
             caption='True (two-sided) Valid Discrimination Rate (\\%) per '
                     'tool, classifier, and configuration: the fraction of '
                     "each tool's confirmed classic IDIs whose original "
                     'instance is valid and for which a constraint-valid '
                     'counterfactual disagrees.',
             label='tab:posthoc_vdr')

    # Progressive tightening: raw -> partial (Kitamura, instance-only) ->
    # true (two-sided). Each step is a validity criterion; the drop from
    # partial to true is the evidence that the counterfactual side matters.
    gap = long.groupby('tool').agg(
        cells=('config', 'count'),
        mean_vdr_partial=('vdr_partial', 'mean'),
        mean_vdr_true=('vdr_true', 'mean'),
    ).round(2).reset_index()
    gap['partial_to_true_drop'] = (gap.mean_vdr_partial - gap.mean_vdr_true).round(2)
    gap.to_csv(os.path.join(OUT_DIR, 'posthoc_partial_vs_true.csv'),
               index=False)
    to_latex(gap, os.path.join(OUT_DIR, 'posthoc_partial_vs_true.tex'),
             caption='Partial (instance-only, Kitamura et al.) vs true '
                     '(two-sided) validity per tool, averaged over all audited '
                     'configurations. The final column is the additional '
                     'validity lost when the counterfactual side is also '
                     'checked.',
             label='tab:posthoc_gap', float_format='%.2f')

    # RQ2 summary: true-valid yield and estimated time-to-valid per tool.
    summary = long.groupby(['classifier', 'tool']).agg(
        configs=('config', 'nunique'),
        mean_raw=('n_raw', 'mean'),
        mean_true_valid=('n_true', 'mean'),
        mean_vdr_true=('vdr_true', 'mean'),
        median_ttd=('ttd_1000', 'median'),
        median_vtt_est=('vtt_1000_est', 'median'),
    ).round(2).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, 'posthoc_summary.csv'), index=False)
    to_latex(summary, os.path.join(OUT_DIR, 'posthoc_summary.tex'),
             caption='Per-tool audit summary: raw IDI volume, true (two-sided) '
                     'valid yield, VDR, and interpolated times (s) to 1,000 '
                     'raw (TTD) and 1,000 true-valid (VTT, uniform-validity '
                     'estimate) IDIs. $-1$: milestone not reached in the '
                     'recorded run.',
             label='tab:posthoc_summary', float_format='%.2f')
    print(f"wrote RQ1 + progression + summary tables to {OUT_DIR}")


if __name__ == '__main__':
    main()
