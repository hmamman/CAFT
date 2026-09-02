"""
RQ4c: cross-method validity audit using Kitamura et al.'s t-way criterion.

Reproduces the data-validity inference of Kitamura et al. (ASE 2024,
IFT-V_t-way, Algorithm 1) as an independent validity oracle, then re-audits
every tool's IDIs under it. Because their criterion (hard, count-based t-way
co-occurrence) is constructed differently from CAFT's confidence-thresholded
association rules, agreement between the two rebuts the concern that CAFT's
high validity is an artefact of its own constraint method.

Kitamura's t-way inference (their Section 4.1, parameters from their Table 1):
  - discretise large-domain attributes into k equal-frequency bins (k=10);
  - a t-way value interaction is a combination of t attribute values (t=2);
  - an interaction is INVALID if its frequency in the training data is <= theta
    (theta=0: it never co-occurs);
  - an instance is invalid if it contains any invalid interaction.
This module checks validity at the instance level, matching their Algorithm 1
(line 4 filters data instances), and reports, per tool, the fraction of raw
IDIs that are t-way valid alongside the fraction valid under CAFT's own
constraints, plus their agreement.

Run from the repo root:
    python scripts/run_tway_audit.py
Flags: --t 2 --k 10 --theta 0 --sample 5000
       --tools THEMIS,AFT,ExpGA,GRFT,MAFT,MAEFT,CAFT
Outputs CSV + LaTeX under results/rq4/.
"""
import argparse
import os
import sys
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, ".."))
os.chdir(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import tensorflow as tf

from src.utils.helpers import get_config_dict, load_disc_inputs
from src.caft.caft import CAFT

TOOLS   = ['THEMIS', 'AFT', 'ExpGA', 'GRFT', 'MAFT', 'MAEFT', 'CAFT']
DATASETS   = ['census', 'compas', 'credit', 'bank', 'meps']
DS_MAP     = {'census': 'Census', 'compas': 'COMPAS', 'credit': 'Credit',
              'bank': 'Bank', 'meps': 'MEPS'}
CLASSIFIERS = ['lr', 'rf', 'dnn']
OUT_DIR = os.path.join('results', 'rq4')


class TwayValidator:
    """Kitamura et al. IFT-V_t-way data-validity oracle, fit on training data.

    Continuous attributes are discretised into k equal-frequency bins; nominal
    attributes keep their integer encoding. Numerical vs. categorical is decided
    by the same cardinality cutoff CAFT uses for constraint extraction
    (``cat_threshold``), so the two validity criteria being compared make
    identical type decisions and only the criterion differs. Nominal attributes
    such as native_country are large-domain but must not be quantile-binned;
    the cutoff must therefore separate genuine continuous columns, not merely
    high-cardinality ones. The datasets are already integer-encoded, so the
    binning applies only to the columns with wide numeric ranges (e.g. balance,
    hours_per_week, age)."""

    def __init__(self, df, t=2, k=10, theta=0, cat_threshold=20):
        self.t, self.k, self.theta = t, k, theta
        self.cols = list(df.columns)
        X = df.values.astype(float)
        n = X.shape[1]

        # Bin only columns CAFT treats as numerical (cardinality > cat_threshold)
        # into k equal-frequency bins; store edges so instances bin identically.
        self.edges = {}
        Xb = np.empty_like(X, dtype=np.int64)
        for j in range(n):
            col = X[:, j]
            if len(np.unique(col)) > cat_threshold:
                qs = np.quantile(col, np.linspace(0, 1, k + 1))
                edges = np.unique(qs)[1:-1]          # interior cut points
                self.edges[j] = edges
                Xb[:, j] = np.digitize(col, edges)
            else:
                self.edges[j] = None
                Xb[:, j] = col.astype(np.int64)
        self._card = Xb.max(axis=0) + 1              # per-column bin cardinality

        # For each attribute pair, the set of value-combos that occur in
        # training (encoded as a single int). An instance's pair is invalid if
        # its combo is absent (frequency <= theta = 0). theta>0 would require
        # counts; kept general below.
        self.pairs = list(combinations(range(n), 2))
        self._occur = {}
        for (a, b) in self.pairs:
            code = Xb[:, a].astype(np.int64) * self._card[b] + Xb[:, b]
            if theta <= 0:
                self._occur[(a, b)] = set(np.unique(code).tolist())
            else:
                vals, cnts = np.unique(code, return_counts=True)
                self._occur[(a, b)] = set(vals[cnts > theta].tolist())

    def _bin(self, X):
        Xb = np.empty_like(X, dtype=np.int64)
        for j in range(X.shape[1]):
            e = self.edges[j]
            Xb[:, j] = np.digitize(X[:, j].astype(float), e) if e is not None \
                else X[:, j].astype(np.int64)
        return Xb

    def valid_mask(self, X):
        """Boolean per row: contains no invalid t-way interaction."""
        Xb = self._bin(np.asarray(X, dtype=float))
        N = len(Xb)
        ok = np.ones(N, dtype=bool)
        for (a, b) in self.pairs:
            code = Xb[:, a] * self._card[b] + Xb[:, b]
            occ = self._occur[(a, b)]
            ok &= np.array([c in occ for c in code], dtype=bool)
        return ok


def load_encoded(config):
    feats = list(config.feature_name)
    rows = []
    with open(os.path.join('datasets', config.dataset_name)) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            rows.append([float(x) for x in line.strip().split(',')[:len(feats)]])
    return pd.DataFrame(rows, columns=feats).astype(int)


def dedup(idis, prot_cols):
    idis = np.asarray(idis)
    if len(idis) == 0:
        return idis
    canon = idis.copy(); canon[:, prot_cols] = 0
    _, idx = np.unique(canon, axis=0, return_index=True)
    return idis[np.sort(idx)]


def to_latex(df, path, caption, label, float_format='%.2f'):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(df.to_latex(index=False, escape=True, na_rep='--',
                            float_format=float_format,
                            caption=caption, label=label, position='ht'))


def main():
    ap = argparse.ArgumentParser(description='Cross-method t-way validity audit.')
    ap.add_argument('--t', type=int, default=2)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--theta', type=int, default=0)
    ap.add_argument('--sample', type=int, default=5000,
                    help='IDIs validated per cell (0 = all)')
    ap.add_argument('--tools', default=','.join(TOOLS))
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--classifiers', default=','.join(CLASSIFIERS))
    ap.add_argument('--min_rows', type=int, default=30)
    args = ap.parse_args()

    tools = [t.strip() for t in args.tools.split(',') if t.strip()]
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(',') if c.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = get_config_dict()
    rng = np.random.default_rng(42)
    records = []

    for ds in datasets:
        config = cfg[ds]
        try:
            df = load_encoded(config)
        except FileNotFoundError:
            continue
        print(f"\n[{ds}] fitting t-way (t={args.t}, k={args.k}, "
              f"theta={args.theta}) on {len(df)} rows ...", flush=True)
        tway = TwayValidator(df, t=args.t, k=args.k, theta=args.theta)

        for sp, sname in config.sens_name.items():
            prot_cols = [sp - 1]
            for clf in classifiers:
                mfile = 'dnn.keras' if clf == 'dnn' else f'{clf}.pkl'
                if not os.path.exists(os.path.join('models', ds, mfile)):
                    continue
                # CAFT built once per (ds, sname, clf) as the C_tau oracle for
                # the agreement comparison; model needed only for its rule arrays.
                caft = None
                for tool in tools:
                    fp = os.path.join('test_data', tool, ds,
                                      f'{clf}_{sname}_disc_inputs.parquet')
                    if not os.path.exists(fp):
                        continue
                    idis = dedup(load_disc_inputs(tool, ds, sname, clf), prot_cols)
                    if len(idis) < args.min_rows:
                        continue
                    if args.sample and len(idis) > args.sample:
                        idis = idis[rng.choice(len(idis), args.sample, replace=False)]

                    if caft is None:
                        model = (tf.keras.models.load_model(
                                    os.path.join('models', ds, 'dnn.keras'),
                                    compile=False)
                                 if clf == 'dnn' else
                                 joblib.load(os.path.join('models', ds, f'{clf}.pkl')))
                        caft = CAFT(config, model, clf, [sp],
                                    constraint_mode='none', population_size=10,
                                    lambda_redundancy=0.0)

                    # Two-sided (both-members) validity, matching Kitamura's
                    # Definition 3 and CAFT's true-validity metric: generate all
                    # protected-attribute variants, and require the reported
                    # instance valid AND a valid counterfactual that disagrees.
                    X = idis.astype(np.int64); N = len(X); M = caft._M
                    batch, _, _ = caft.similar_set_(X)
                    preds = caft.model.predict(batch).reshape(N, M)
                    own = np.array([caft._comb_to_idx.get(
                        tuple(int(v) for v in X[i, caft.protected_attribs]), 0)
                        for i in range(N)])
                    confirmed = preds.max(axis=1) != preds.min(axis=1)   # raw IDI

                    def true_valid(vmask):
                        vmask = vmask.reshape(N, M)
                        own_ok = vmask[np.arange(N), own]
                        mx = np.where(vmask, preds, -np.inf).max(axis=1)
                        mn = np.where(vmask, preds, np.inf).min(axis=1)
                        return own_ok & (vmask.sum(axis=1) >= 2) & (mx != mn)

                    tway_true = true_valid(tway.valid_mask(batch))
                    ctau_viol = (caft._violations_vec(
                                    X, caft._shared_rule_arrs, caft._shared_domain_arrs
                                 )[np.arange(N * M) // M]
                                 + caft._violations_vec(
                                    batch, caft._variant_rule_arrs,
                                    caft._variant_domain_arrs))
                    ctau_true = true_valid(ctau_viol == 0)

                    nconf = int(confirmed.sum())
                    if nconf == 0:
                        continue
                    tw = 100 * (tway_true & confirmed).sum() / nconf
                    ou = 100 * (ctau_true & confirmed).sum() / nconf
                    rec = {
                        'dataset': DS_MAP[ds], 'attribute': sname,
                        'classifier': clf, 'tool': tool, 'n_raw': nconf,
                        'tway_valid_pct': round(tw, 2),
                        'ours_valid_pct': round(ou, 2),
                        'agreement_pct': round(
                            100 * np.mean((tway_true == ctau_true)[confirmed]), 2),
                    }
                    records.append(rec)
                    print(f"  {tool:6s}/{clf:3s}: raw={nconf:6d} "
                          f"t-way={rec['tway_valid_pct']:6.2f}% "
                          f"ours={rec['ours_valid_pct']:6.2f}% "
                          f"agree={rec['agreement_pct']:5.1f}%", flush=True)

    long = pd.DataFrame(records)
    csv = os.path.join(OUT_DIR, 'rq4c_tway_audit.csv')
    long.to_csv(csv, index=False)
    print(f"\nwrote {csv} ({len(long)} cells)")
    if long.empty:
        return

    summary = long.groupby('tool').agg(
        cells=('n_raw', 'count'),
        mean_tway_valid=('tway_valid_pct', 'mean'),
        mean_ours_valid=('ours_valid_pct', 'mean'),
        mean_agreement=('agreement_pct', 'mean'),
    ).round(2).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, 'rq4c_summary.csv'), index=False)
    to_latex(summary, os.path.join(OUT_DIR, 'rq4c_summary.tex'),
             caption='RQ4c cross-method validity audit. Per tool, the mean '
                     'fraction of raw IDIs that are valid under the independent '
                     f'$t$-way criterion of Kitamura et al.\\ ($t={args.t}$, '
                     f'$k={args.k}$, $\\theta={args.theta}$) and under \\caft\'s '
                     'confidence-based constraints, both at the instance level, '
                     'with their per-instance agreement. Close agreement shows '
                     'the near-zero baseline validity is not an artefact of '
                     'either constraint method.',
             label='tab:rq4c-summary')
    print(f"wrote RQ4c summary to {OUT_DIR}")
    print("RQ4C_OK", flush=True)


if __name__ == '__main__':
    main()
