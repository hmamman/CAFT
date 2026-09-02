"""
Smoke test for the new 'repair_penalty' constraint-handling mode.

'repair_penalty' is NOT part of the manuscript's reported ablation. It is a
combined strategy added to caft.py: every generation, each GA individual is
repaired first, then the fitness penalty is applied to whatever violations
the repair pass left unresolved (repair does not guarantee convergence; see
the Repair subsection of the manuscript). This differs from 'hybrid' and
'adaptive', which apply exactly one of repair OR penalty per generation --
'repair_penalty' applies both to the same candidate every generation.

This script compares 'repair_penalty' against the three modes it sits
between -- 'repair', 'penalty', and 'adaptive' -- on COMPAS, the dataset on
which CAFT's production runs show the lowest valid-IDI rate (~6-9% true VDR
in Table ph-main, versus 58-95% on the other four datasets), so any
difference between the modes should be most visible there.

Writes to its own CSV so it never touches or is mistaken for the reported
ablation (results/variants/variants_runs*.csv, read by
make_variants_tables.py). This file is not read by any table-generation
script.

Run from the repo root:
    python scripts/smoke_repair_penalty.py
    python scripts/smoke_repair_penalty.py --datasets compas,credit
"""
import argparse
import os
import sys
import time

# caft.py prints Unicode arrows (e.g. "extracted -> active"); the default
# Windows console codepage (cp1252) cannot encode them and raises
# UnicodeEncodeError mid-run. Reconfigure this process's stdout to UTF-8
# before any CAFT import runs, without touching caft.py's print calls.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
import numpy as np
import pandas as pd
import gc
import tensorflow.keras.backend as K

base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, ".."))
os.chdir(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.utils.helpers import get_config_dict
from src.caft.caft import CAFT

DATASETS    = ['compas']
CLASSIFIERS = ['lr', 'rf', 'dnn']
REPEATS     = 1
MAX_SAMPLES = 1_000_000
OUT_PATH    = 'results/variants/smoke_repair_penalty.csv'

# Identical to COMMON in run_variants.py, so results are comparable to the
# production 'repair' / 'penalty' / 'adaptive' rows on the same dataset.
COMMON = dict(
    min_score=0.97,
    population_size=300,
    lambda_redundancy=2.0,
    seed_strategy='random',
    mutation_strategy='hybrid',
    crossover_strategy='hux',
)

VARIANTS = {
    'repair':         dict(constraint_mode='repair', max_passes=5),
    'penalty':        dict(constraint_mode='penalty', lambda_penalty=0.5,
                            lambda_gradient=0.05),
    'adaptive':       dict(constraint_mode='adaptive', lambda_penalty=0.5,
                            lambda_gradient=0.05, max_passes=5, adaptive_target=0.5),
    'repair_penalty': dict(constraint_mode='repair_penalty', lambda_penalty=0.5,
                            lambda_gradient=0.05, max_passes=5),
}


def load_model(dataset, clf):
    if clf == 'dnn':
        import tensorflow as tf
        return tf.keras.models.load_model(f'models/{dataset}/dnn.keras',
                                          compile=False)
    return joblib.load(f'models/{dataset}/{clf}.pkl')


def one_run(config, model, clf, sp, variant, seed, time_budget):
    """Drive the evolve loop directly (no report() call): nothing is written
    to the production results/ or test_data/ trees."""
    np.random.seed(seed)
    kwargs = dict(COMMON)
    kwargs.update(VARIANTS[variant])
    runner = CAFT(config=config, model=model, classifier_name=clf,
                  sensitive_params=[sp], **kwargs)

    t0 = runner.start_time            # includes constraint-extraction time
    while True:
        runner.GA.evolve()
        elapsed = time.time() - t0
        if elapsed >= time_budget or runner.total_generated >= MAX_SAMPLES:
            break

    n_tot     = max(len(runner.tot_inputs), 1)
    n_raw     = len(runner.raw_disc_inputs)      # raw IDI, classic (A)
    n_partial = len(runner.partial_disc_inputs)  # partial validity, Kitamura (B')
    n_valid   = len(runner.valid_disc_profiles)  # true two-sided validity (C)
    return dict(
        n_profiles=n_tot,
        n_raw=n_raw,
        n_partial=n_partial,
        n_valid=n_valid,
        vdr_partial=round(100 * n_partial / max(n_raw, 1), 2),
        vdr_true=round(100 * n_valid / max(n_raw, 1), 2),
        disc_rate=round(100 * n_valid / n_tot, 2),
        ttd=round(runner.time_to_1000_disc, 2),
        vtt=round(runner.time_to_1000_valid_disc, 2),
        vgs=round(n_valid / elapsed, 1),
        inferences=runner.inference_count,
        generated=runner.total_generated,
        wall_s=round(elapsed, 1),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Smoke test: 'repair_penalty' vs repair/penalty/adaptive.")
    ap.add_argument('--time_budget', type=int, default=3600,
                    help='seconds per run (COMPAS terminates on the 1e6 '
                         'sample cap in ~70-150s in production data, well '
                         'inside this ceiling)')
    ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--classifiers', default=','.join(CLASSIFIERS))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(',') if c.strip()]
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cfg = get_config_dict()

    rows, done = [], set()
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        rows = prev.to_dict('records')
        done = {(r['dataset'], r['classifier'], r['attribute'],
                 r['variant'], r['rep']) for r in rows}
        print(f'resuming: {len(done)} runs already recorded in {OUT_PATH}',
              flush=True)

    all_runs = []
    for ds in datasets:
        config = cfg[ds]
        for clf in classifiers:
            mfile = 'dnn.keras' if clf == 'dnn' else f'{clf}.pkl'
            if not os.path.exists(os.path.join('models', ds, mfile)):
                continue
            for sp, sname in config.sens_name.items():
                for variant in variants:
                    for rep in range(args.repeats):
                        all_runs.append((ds, clf, sp, sname, variant, rep))

    pending = [r for r in all_runs if (r[0], r[1], r[3], r[4], r[5]) not in done]
    print(f'{len(all_runs)} total runs, {len(pending)} pending', flush=True)

    model_cache = {}
    for ds, clf, sp, sname, variant, rep in pending:
        config = cfg[ds]
        seed = 1000 + rep
        if (ds, clf) not in model_cache:
            model_cache[(ds, clf)] = load_model(ds, clf)
        t0 = time.time()
        try:
            m = one_run(config, model_cache[(ds, clf)], clf, sp,
                        variant, seed, args.time_budget)
        except Exception as e:
            print(f'  ERROR {ds}/{clf}/{sname}/{variant}/r{rep}: {e}',
                  flush=True)
            continue
        m.update(dataset=ds, classifier=clf, attribute=sname,
                 variant=variant, rep=rep, seed=seed, budget=args.time_budget)
        rows.append(m)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)   # checkpoint
        print(f'{ds}/{clf}/{sname}  {variant:15s} r{rep}  '
              f'raw={m["n_raw"]:7d} partial={m["n_partial"]:7d} '
              f'true={m["n_valid"]:7d}  vdr={m["vdr_true"]:6.2f}%  '
              f'vgs={m["vgs"]:8.1f}  ({time.time() - t0:.0f}s)', flush=True)
        K.clear_session()
        gc.collect()

    print(f'\nDone. {len(rows)} rows -> {OUT_PATH}', flush=True)
    print('SMOKE_REPAIR_PENALTY_OK', flush=True)

    df = pd.DataFrame(rows)
    if len(df):
        print('\n=== Summary (mean over configs, this dataset set) ===')
        print(df.groupby('variant')[
            ['n_raw', 'n_valid', 'vdr_true', 'vgs', 'wall_s']
        ].mean().round(2).to_string())


if __name__ == '__main__':
    main()
