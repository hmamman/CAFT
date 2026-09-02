"""RQ3 constraint-handling ablation: seeded, repeated runs of the five CAFT variants.

Run from the repo root:
    python scripts/run_variants.py
    python scripts/run_variants.py --shard 1/4
"""
import argparse
import os
import sys
import time

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
from src.caft.caft_moo import CAFT_MOO

DATASETS    = ['census', 'compas', 'credit', 'bank', 'meps']
CLASSIFIERS = ['lr', 'rf', 'dnn']
REPEATS     = 10
MAX_SAMPLES = 1_000_000
OUT_PATH    = 'results/variants/variants_runs.csv'

# Shared production settings — identical across variants (controlled comparison).

COMMON = dict(
    min_score=0.97,
    population_size=300,
    lambda_redundancy=2.0,
    seed_strategy='random',
    mutation_strategy='hybrid',
    crossover_strategy='hux',
)

VARIANTS = {
    'none':    dict(constraint_mode='none'),
    'penalty': dict(constraint_mode='penalty', lambda_penalty=0.5,
                    lambda_gradient=0.05),
    'repair':  dict(constraint_mode='repair', max_passes=5),
    'hybrid':  dict(constraint_mode='hybrid', lambda_penalty=0.5,
                    lambda_gradient=0.05, max_passes=5,
                    repair_interval=50, repair_bursts=2),
    'adaptive': dict(constraint_mode='adaptive', lambda_penalty=0.5,
                     lambda_gradient=0.05, max_passes=5, adaptive_target=0.5),
    'moo':     dict(selection='rank_roulette', diversity_window=1000,
                    elite_frac=0.05),
}


def load_model(dataset, clf):
    if clf == 'dnn':
        import tensorflow as tf
        return tf.keras.models.load_model(f'models/{dataset}/dnn.keras',
                                          compile=False)
    return joblib.load(f'models/{dataset}/{clf}.pkl')


def one_run(config, model, clf, sp, variant, seed, time_budget):
    """Build the variant runner, drive the evolve loop under the time budget,
    and return the run metrics. No report() call: nothing is written to the
    production results/ or test_data/ trees."""
    np.random.seed(seed)
    kwargs = dict(COMMON)
    kwargs.update(VARIANTS[variant])
    cls = CAFT_MOO if variant == 'moo' else CAFT
    runner = cls(config=config, model=model, classifier_name=clf,
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
    n_valid   = len(runner.valid_disc_profiles)  # true two-sided validity (C) — headline VDI
    return dict(
        n_profiles=n_tot,
        n_raw=n_raw,
        n_partial=n_partial,
        n_valid=n_valid,
        vdr_partial=round(100 * n_partial / max(n_raw, 1), 2),  # Kitamura VDR = B'/A
        vdr_true=round(100 * n_valid / max(n_raw, 1), 2),       # our VDR = C/A
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
        description='Seeded design-space runs of the five CAFT variants.')
    ap.add_argument('--time_budget', type=int, default=3600,
                    help='seconds per run (non-MEPS datasets)')
    ap.add_argument('--meps_budget', type=int, default=3600,
                    help='seconds per run on MEPS (sparse 40-feature space)')
    ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--classifiers', default=','.join(CLASSIFIERS))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--shard', default='1/1',
                    help="'k/n': run only pending runs where index %% n == k-1")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(',') if c.strip()]
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    shard_k, shard_n = (int(x) for x in args.shard.split('/'))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cfg = get_config_dict()

    # Each shard writes its own file so concurrent shards never clobber each
    # other's checkpoints; the analysis script concatenates variants_runs*.csv.
    # The done-set is the union across ALL shard files, so any shard resumes
    # past work recorded by any other.
    import glob
    out_path = OUT_PATH if shard_n == 1 else \
        OUT_PATH.replace('.csv', f'_s{shard_k}of{shard_n}.csv')
    rows, done = [], set()
    for f in glob.glob(OUT_PATH.replace('.csv', '*.csv')):
        prev = pd.read_csv(f)
        done |= {(r['dataset'], r['classifier'], r['attribute'],
                  r['variant'], r['rep']) for r in prev.to_dict('records')}
        if os.path.abspath(f) == os.path.abspath(out_path):
            rows = prev.to_dict('records')
    if done:
        print(f'resuming: {len(done)} runs recorded across shard files, '
              f'{len(rows)} in this shard\'s file', flush=True)

    # Enumerate the full deterministic run list, then filter to this shard.
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

    pending = [(i, r) for i, r in enumerate(all_runs)
               if (r[0], r[1], r[3], r[4], r[5]) not in done]
    mine = [r for i, r in pending if i % shard_n == shard_k - 1]
    print(f'{len(all_runs)} total runs, {len(pending)} pending, '
          f'{len(mine)} in shard {args.shard}', flush=True)

    model_cache = {}
    for ds, clf, sp, sname, variant, rep in mine:
        config = cfg[ds]
        budget = args.meps_budget if ds == 'meps' else args.time_budget
        seed = 1000 + rep
        if (ds, clf) not in model_cache:
            model_cache[(ds, clf)] = load_model(ds, clf)
        t0 = time.time()
        try:
            m = one_run(config, model_cache[(ds, clf)], clf, sp,
                        variant, seed, budget)
        except Exception as e:
            print(f'  ERROR {ds}/{clf}/{sname}/{variant}/r{rep}: {e}',
                  flush=True)
            continue
        m.update(dataset=ds, classifier=clf, attribute=sname,
                 variant=variant, rep=rep, seed=seed, budget=budget)
        rows.append(m)
        pd.DataFrame(rows).to_csv(out_path, index=False)   # checkpoint
        print(f'{ds}/{clf}/{sname}  {variant:8s} r{rep}  '
              f'raw={m["n_raw"]:7d} partial={m["n_partial"]:7d} '
              f'true={m["n_valid"]:7d}  vdr={m["vdr_true"]:6.2f}%  '
              f'vtt={m["vtt"]:7.2f}  ({time.time() - t0:.0f}s)', flush=True)
        # Clear the session to free up memory
        K.clear_session()
        gc.collect()

    print(f'\nDone. {len(rows)} rows in this shard -> {out_path}', flush=True)
    print('VARIANTS_RUNS_OK', flush=True)


if __name__ == '__main__':
    main()
