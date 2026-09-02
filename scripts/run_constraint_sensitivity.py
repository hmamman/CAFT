"""
Constraint sensitivity ablation for CAFT.

Tests three constraint-enforcement modes (none, penalty, repair) across four
values of min_score / τ (0.90, 0.95, 0.97, 0.99) to assess how constraint
enforcement and confidence thresholds affect valid IDI yield.

All other CAFT hyperparameters are held at the production values used in
the main experiments.

Run from the repo root:
    python scripts/run_constraint_sensitivity.py
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, ".."))
sys.path.insert(0, root_path)

from src.caft.caft import CAFT
from src.utils.helpers import get_config_dict

DATASETS    = ['census', 'bank', 'credit', 'meps']
CLASSIFIERS = ['lr', 'rf', 'dnn']
REPEATS     = 3
TIME_BUDGET = 3600
MAX_SAMPLES = 1_000_000
OUTPUT_DIR  = 'results/ablation/CAFT/constraint_sensitivity'
OUT_PATH    = f'{OUTPUT_DIR}/constraint_sensitivity.csv'

# Production config — fixed across all ablation variants
PROD = dict(
    population_size=300,
    crossover_strategy='hux',
    mutation_strategy='hybrid',
    sigma=0.05,
    seed_strategy='random',
    lambda_penalty=1.0,
    lambda_gradient=0.05,
    lambda_redundancy=2.0,
    max_passes=5,
    n_bins=5,
    categorical_threshold=20,
    ga_categorical_threshold=5,
    repair_interval=50,
    repair_bursts=2,
)

# 12 configurations: 3 modes × 4 τ values
ABLATIONS = {
    f'{mode}_tau{tau}': dict(constraint_mode=mode, min_score=tau, **PROD)
    for mode in ['none', 'penalty', 'repair']
    for tau in [0.90, 0.95, 0.97, 0.99]
}


def load_model(dataset, clf):
    if clf == 'dnn':
        import tensorflow as tf
        return tf.keras.models.load_model(f'models/{dataset}/dnn.keras')
    return joblib.load(f'models/{dataset}/{clf}.pkl')


def one_run(config, clf, sp, approach_name, **kwargs):
    r = CAFT(
        config=config,
        model=load_model(config.dataset_name, clf),
        classifier_name=clf,
        sensitive_params=[sp],
        **kwargs,
    )
    r.approach_name = approach_name
    r.run(max_allowed_time=TIME_BUDGET, max_samples=MAX_SAMPLES)

    n_profiles = max(len(r._seen_profiles), 1)
    n_disc     = len(r.disc_inputs)
    n_valid    = len(r.valid_disc_profiles)
    return dict(
        n_disc=n_disc,
        n_valid_disc=n_valid,
        tot_inputs=len(r.tot_inputs),
        total_generated=r.total_generated,
        disc_rate=round(n_disc / n_profiles * 100, 2),
        valid_disc_rate=round(n_valid / n_profiles * 100, 2),
        validity_ratio=round(n_valid / n_disc * 100, 2) if n_disc > 0 else 0,
        time_to_1000_disc=round(r.time_to_1000_disc, 1),
        time_to_1000_valid_disc=round(r.time_to_1000_valid_disc, 1),
        inference_count=r.inference_count,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = get_config_dict()
    rows = []

    for ds in DATASETS:
        config = cfg[ds]
        for clf in CLASSIFIERS:
            for sp, sname in config.sens_name.items():
                for abl_name, abl_kwargs in ABLATIONS.items():
                    runs = []
                    for rep in range(REPEATS):
                        t0 = time.time()
                        try:
                            m = one_run(
                                config, clf, sp,
                                approach_name=f'CAFT_{abl_name}',
                                **abl_kwargs,
                            )
                            m.update(
                                dataset=ds, classifier=clf,
                                attribute=sname, ablation=abl_name,
                                constraint_mode=abl_kwargs['constraint_mode'],
                                min_score=abl_kwargs['min_score'],
                                repeat=rep,
                                wall_s=round(time.time() - t0, 1),
                            )
                            runs.append(m)
                            rows.append(m)
                            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
                        except Exception as e:
                            print(
                                f"  ERROR {ds}/{clf}/{sname}/{abl_name}/r{rep}: {e}",
                                flush=True,
                            )
                    if runs:
                        df = pd.DataFrame(runs)
                        print(
                            f"{ds}/{clf}/{sname}  {abl_name:20s}  "
                            f"disc={df.disc_rate.mean():.1f}±{df.disc_rate.std():.1f}%  "
                            f"valid={df.valid_disc_rate.mean():.1f}±{df.valid_disc_rate.std():.1f}%  "
                            f"ratio={df.validity_ratio.mean():.1f}%  "
                            f"nIDI={df.n_disc.mean():.0f}  "
                            f"nValid={df.n_valid_disc.mean():.0f}",
                            flush=True,
                        )

    print(f"\nDone. {len(rows)} rows -> {OUT_PATH}", flush=True)
    print("CONSTRAINT_SENSITIVITY_OK", flush=True)


if __name__ == '__main__':
    main()
