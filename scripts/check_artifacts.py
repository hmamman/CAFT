"""
Integrity check for CAFT's saved test-data artifacts.

For every (dataset, classifier, protected attribute) that has a saved
valid_disc_inputs file, re-validates a sample of the saved raw and valid
instances against the model and constraints (independent of the run's own
tracking) and reports:
  raw%   -- fraction of the raw file that are genuine raw IDIs;
  true%  -- fraction of the valid file that are genuine two-sided-valid IDIs;
  contain-- whether the valid set's non-protected profiles are a subset of raw.

Any config where raw% or true% is below 100 indicates a corrupted or stale
artifact. Also lists configs that are missing a valid or raw file.

Run from the repo root:
    python scripts/check_artifacts.py
"""
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, ".."))
os.chdir(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.utils.helpers import get_config_dict
from src.caft.caft import CAFT

DATASETS = ['census', 'bank', 'credit', 'compas', 'meps']
CLASSIFIERS = ['lr', 'rf', 'dnn']
SAMPLE = 5000   # instances re-validated per file (0 = all)


def load_model(ds, clf):
    if clf == 'dnn':
        import tensorflow as tf
        return tf.keras.models.load_model(f'models/{ds}/dnn.keras', compile=False)
    return joblib.load(f'models/{ds}/{clf}.pkl')


def revalidate(c, rows, model):
    X = np.asarray(rows).astype(np.int64)
    N, M = len(X), c._M
    batch, _, _ = c.similar_set_(X)
    preds = model.predict(batch).reshape(N, M)
    sv = c._violations_vec(X, c._shared_rule_arrs, c._shared_domain_arrs)
    vv = c._violations_vec(batch, c._variant_rule_arrs, c._variant_domain_arrs)
    vmat = ((sv[np.arange(N * M) // M] + vv) == 0).reshape(N, M)
    own = np.array([c._comb_to_idx.get(
        tuple(int(v) for v in X[i, c.protected_attribs]), 0) for i in range(N)])
    raw = preds.max(1) != preds.min(1)
    mx = np.where(vmat, preds, -np.inf).max(1)
    mn = np.where(vmat, preds, np.inf).min(1)
    two = vmat[np.arange(N), own] & (vmat.sum(1) >= 2) & (mx != mn)
    return raw, two


def main():
    cfg = get_config_dict()
    rng = np.random.default_rng(0)
    rows = []
    bad = 0
    for ds in DATASETS:
        config = cfg[ds]
        for clf in CLASSIFIERS:
            if not os.path.exists(f'models/{ds}/{clf}.pkl') and clf != 'dnn':
                continue
            if clf == 'dnn' and not os.path.exists(f'models/{ds}/dnn.keras'):
                continue
            c = model = None
            for sp, sname in config.sens_name.items():
                base = f'test_data/CAFT/{ds}/{clf}_{sname}'
                dp, vp = base + '_disc_inputs.parquet', base + '_valid_disc_inputs.parquet'
                if not os.path.exists(vp) or not os.path.exists(dp):
                    if os.path.exists(dp) or os.path.exists(vp):
                        print(f'  {ds}/{clf}/{sname}: MISSING '
                              f'{"valid" if not os.path.exists(vp) else "raw"} file')
                    continue
                raw_rows = pd.read_parquet(dp).to_numpy()
                val_rows = pd.read_parquet(vp).to_numpy()
                if c is None:
                    model = load_model(ds, clf)
                    c = CAFT(config, model, clf, [sp], constraint_mode='none',
                             population_size=10, lambda_redundancy=0.0)
                else:
                    # protected attr changed → rebuild rule partition
                    c = CAFT(config, model, clf, [sp], constraint_mode='none',
                             population_size=10, lambda_redundancy=0.0)
                rs = raw_rows if not SAMPLE or len(raw_rows) <= SAMPLE else \
                    raw_rows[rng.choice(len(raw_rows), SAMPLE, replace=False)]
                vs = val_rows if not SAMPLE or len(val_rows) <= SAMPLE else \
                    val_rows[rng.choice(len(val_rows), SAMPLE, replace=False)]
                r_raw, _ = revalidate(c, rs, model)
                _, v_two = revalidate(c, vs, model)
                nsk = lambda A: {hash(r[c._non_sens_mask].tobytes())
                                 for r in np.asarray(A).astype(np.int64)}
                contained = nsk(val_rows).issubset(nsk(raw_rows))
                raw_pct, true_pct = 100 * r_raw.mean(), 100 * v_two.mean()
                ok = raw_pct >= 99.9 and true_pct >= 99.9 and contained
                bad += (not ok)
                rows.append(dict(dataset=ds, clf=clf, attr=sname,
                                 n_raw=len(raw_rows), n_valid=len(val_rows),
                                 raw_pct=round(raw_pct, 2), true_pct=round(true_pct, 2),
                                 contained=contained, ok=ok))
                flag = 'OK ' if ok else 'BAD'
                print(f'  [{flag}] {ds}/{clf}/{sname}: raw={len(raw_rows):7d} '
                      f'valid={len(val_rows):7d} | raw%={raw_pct:6.2f} '
                      f'true%={true_pct:6.2f} contained={contained}', flush=True)

    df = pd.DataFrame(rows)
    print(f'\n{len(df)} configs checked, {bad} failed.')
    if not df.empty:
        print('coverage by dataset:')
        print(df.groupby('dataset').agg(configs=('attr', 'count'),
              all_ok=('ok', 'all')).to_string())
    print('ARTIFACT_CHECK_OK' if bad == 0 else 'ARTIFACT_CHECK_FAILED', flush=True)


if __name__ == '__main__':
    main()
