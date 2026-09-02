"""
Naturalness / authenticity of generated IDIs vs the training data.

Complements the hard rule-validity results (RQ1-RQ4) with the soft
distributional-fidelity metrics used by MAEFT (Dong et al.), so \caft's
constraint-valid IDIs can be positioned against prior tools on their own
criterion. Higher is better for CDE/PCC/P_alpha; lower is better for MD.

Metrics (per dataset, per tool, IDIs vs training data):
  CDE  -- Column-wise Density Estimation: mean per-column marginal similarity,
          1 - KS statistic for numeric columns, 1 - total-variation distance for
          categorical columns (%).
  PCC  -- Pair-wise Column Correlation: mean per-pair association agreement,
          Pearson r for numeric-numeric pairs and Cramer's V for pairs involving
          a categorical column; reported as 1 - mean|assoc_real - assoc_syn| (%).
  Palpha- alpha-Precision (Alaa et al. 2022): standardise on the training data,
          centre at its mean, and integrate over alpha the fraction of generated
          points inside the training alpha-support hypersphere (radius = the
          alpha-quantile of training-point distances to the centre) (%).
          Reimplemented here (no sdmetrics/synthcity available); parameterisation
          stated in the code.
  MD   -- mean Mahalanobis distance of generated points to the training
          distribution (regularised covariance).

Every tool, \caft included, is measured on its raw IDIs (disc_inputs), so the
comparison uses one output set per tool and no tool is measured on a filtered
subset. Tools' IDIs are pooled across a dataset's configurations and subsampled.

Run from the repo root:
    python scripts/run_naturalness.py
Outputs CSV + LaTeX under results/naturalness/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, ".."))
os.chdir(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.utils.helpers import get_config_dict, get_data, load_disc_inputs

DATASETS = ['census', 'bank', 'credit', 'compas', 'meps']
DS_MAP = {'census': 'Census', 'bank': 'Bank', 'credit': 'Credit',
          'compas': 'COMPAS', 'meps': 'MEPS'}
BASELINES = ['THEMIS', 'AFT', 'ExpGA', 'GRFT', 'MAFT', 'MAEFT']
TOOLS = BASELINES + ['CAFT']
CLASSIFIERS = ['lr', 'rf', 'dnn']
CAT_THRESHOLD = 20     # nunique <= this => categorical (matches CAFT)
SAMPLE = 5000          # generated points sampled per (dataset, tool)


def cat_mask(real):
    return np.array([len(np.unique(real[:, j])) <= CAT_THRESHOLD
                     for j in range(real.shape[1])])


def cde(real, syn, is_cat):
    sims = []
    for j in range(real.shape[1]):
        if is_cat[j]:
            vals = np.union1d(real[:, j], syn[:, j])
            pr = np.array([(real[:, j] == v).mean() for v in vals])
            ps = np.array([(syn[:, j] == v).mean() for v in vals])
            sims.append(1.0 - 0.5 * np.abs(pr - ps).sum())      # 1 - TVD
        else:
            sims.append(1.0 - ks_2samp(real[:, j], syn[:, j]).statistic)
    return 100.0 * float(np.mean(sims))


def _cramers_v(a, b):
    tab = pd.crosstab(a, b).values.astype(float)
    n = tab.sum()
    if n == 0:
        return 0.0
    row = tab.sum(1, keepdims=True); col = tab.sum(0, keepdims=True)
    exp = row @ col / n
    chi2 = np.nansum((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp))
    r, k = tab.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def _assoc(x, y, cx, cy):
    if not cx and not cy:
        s = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(s) else abs(s)
    return _cramers_v(x, y)


def pcc(real, syn, is_cat):
    d = real.shape[1]
    diffs = []
    for i in range(d):
        for j in range(i + 1, d):
            ar = _assoc(real[:, i], real[:, j], is_cat[i], is_cat[j])
            as_ = _assoc(syn[:, i], syn[:, j], is_cat[i], is_cat[j])
            diffs.append(abs(ar - as_))
    return 100.0 * (1.0 - float(np.mean(diffs))) if diffs else np.nan


def p_alpha(real, syn, n_alpha=19):
    mu = real.mean(0); sd = real.std(0); sd[sd == 0] = 1.0
    Rr = (real - mu) / sd; Rs = (syn - mu) / sd
    c = Rr.mean(0)
    dr = np.linalg.norm(Rr - c, axis=1)
    ds = np.linalg.norm(Rs - c, axis=1)
    alphas = np.linspace(0.05, 0.95, n_alpha)
    prec = [(ds <= np.quantile(dr, a)).mean() for a in alphas]
    return 100.0 * float(np.mean(prec))


def md(real, syn):
    mu = real.mean(0)
    cov = np.cov(real, rowvar=False) + 1e-6 * np.eye(real.shape[1])
    inv = np.linalg.pinv(cov)
    diff = syn - mu
    d2 = np.einsum('ij,jk,ik->i', diff, inv, diff)
    return float(np.mean(np.sqrt(np.clip(d2, 0, None))))


def load_tool_idis(tool, ds, config, rng):
    """Pool a tool's IDIs across classifiers and attributes for a dataset."""
    # Raw IDIs for every tool, CAFT included: measuring CAFT on its valid
    # subset while baselines are measured on their full output would bias the
    # distributional metrics in CAFT's favour.
    suffix = 'disc_inputs'
    pools = []
    for clf in CLASSIFIERS:
        for sp, sname in config.sens_name.items():
            fp = os.path.join('test_data', tool, ds,
                              f'{clf}_{sname}_{suffix}.parquet')
            if not os.path.exists(fp):
                continue
            try:
                pools.append(load_disc_inputs(tool, ds, sname, clf, suffix=suffix))
            except Exception:
                pass
    if not pools:
        return None
    allrows = np.vstack(pools)
    if len(allrows) > SAMPLE:
        allrows = allrows[rng.choice(len(allrows), SAMPLE, replace=False)]
    return allrows.astype(float)


def to_latex(df, path, caption, label):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(df.to_latex(index=False, escape=True, na_rep='--',
                            float_format='%.2f', caption=caption, label=label,
                            position='ht'))


def main():
    ap = argparse.ArgumentParser(description='Naturalness of generated IDIs.')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--tools', default=','.join(TOOLS))
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    tools = [t.strip() for t in args.tools.split(',') if t.strip()]

    out_dir = os.path.join('results',
                           'naturalness')
    os.makedirs(out_dir, exist_ok=True)
    cfg = get_config_dict()
    rng = np.random.default_rng(0)
    rows = []

    for ds in datasets:
        config = cfg[ds]
        real = np.asarray(get_data(config.dataset_name)()[0]).astype(float)
        is_cat = cat_mask(real)
        if len(real) > 20000:
            real = real[rng.choice(len(real), 20000, replace=False)]
        for tool in tools:
            syn = load_tool_idis(tool, ds, config, rng)
            if syn is None or len(syn) < 50:
                continue
            rec = {'dataset': DS_MAP[ds], 'tool': tool, 'n': len(syn),
                   'CDE': round(cde(real, syn, is_cat), 2),
                   'PCC': round(pcc(real, syn, is_cat), 2),
                   'Palpha': round(p_alpha(real, syn), 2),
                   'MD': round(md(real, syn), 2)}
            rows.append(rec)
            print(f"  {ds:7s}/{tool:6s}: n={len(syn):5d} CDE={rec['CDE']:6.2f} "
                  f"PCC={rec['PCC']:6.2f} Palpha={rec['Palpha']:6.2f} "
                  f"MD={rec['MD']:7.2f}", flush=True)

    df = pd.DataFrame(rows)
    csv = os.path.join(out_dir, 'naturalness.csv')
    df.to_csv(csv, index=False)
    print(f'\nwrote {csv} ({len(df)} rows)')
    if df.empty:
        return
    # Per-dataset x per-tool naturalness, with a Mean block aggregating each
    # tool over datasets. Best per column bolded within each block (max for the
    # similarity metrics, min for MD). \caft measured on its valid IDIs.
    METR = ['CDE', 'PCC', 'Palpha', 'MD']
    FEAS = ['Census', 'Bank', 'Credit', 'COMPAS', 'MEPS']
    tool_order = [t for t in TOOLS if t in df.tool.unique()]

    def block(sub, ds_label, first_multirow):
        best = {m: (sub[m].min() if m == 'MD' else sub[m].max()) for m in METR}
        out = []
        present = [t for t in tool_order if t in sub.tool.values]
        for i, t in enumerate(present):
            r = sub[sub.tool == t].iloc[0]
            name = '\\caft' if t == 'CAFT' else f'\\iftool{{{t}}}'
            cells = []
            for m in METR:
                s = f'{r[m]:.1f}'
                if r[m] == best[m]:
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
            lead = (f'\\multirow{{{len(present)}}}{{*}}{{{ds_label}}}'
                    if i == 0 and first_multirow else
                    (ds_label if i == 0 else ''))
            out.append(f'{lead} & {name} & ' + ' & '.join(cells) + ' \\\\')
        return out

    body = []
    for ds in FEAS:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        body += block(sub, ds, True)
        body.append('\\midrule')
    # Mean over datasets per tool.
    mean_df = (df.groupby('tool')[METR].mean().reset_index())
    body += block(mean_df, '\\textbf{Mean}', False)
    L = ['\\begin{table}[!htb]', '\\centering',
         '\\caption{Distributional naturalness of generated IDIs against the '
         'training data, by dataset and tool. CDE (column-wise density), PCC '
         '(pairwise column correlation), and $P_\\alpha$ ($\\alpha$-precision) '
         'are similarities in \\% (higher is better); MD is the mean '
         'Mahalanobis distance to the training distribution (lower is better). '
         'Best in each column within each dataset in bold; the \\textbf{Mean} block '
         'averages each tool over the five datasets. Every tool, \\caft '
         'included, is measured on its raw IDIs, so no tool is evaluated on a '
         'filtered subset. Naturalness is a soft distributional criterion, '
         'distinct from the hard rule-validity of '
         'Section~\\ref{sec:rq1-results}.}',
         '\\label{tab:naturalness}', '\\footnotesize',
         '\\setlength{\\tabcolsep}{5pt}',
         '\\begin{tabular}{@{}ll|rrrr@{}}', '\\toprule',
         '\\multicolumn{2}{c|}{Configuration} & '
         '\\multicolumn{4}{c}{Naturalness metric} \\\\',
         '\\cmidrule(lr){1-2}\\cmidrule(lr){3-6}',
         'Dataset & Tool & CDE$\\uparrow$ & PCC$\\uparrow$ & '
         '$P_\\alpha\\uparrow$ & MD$\\downarrow$ \\\\', '\\midrule'] + body + \
        ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    with open(os.path.join(out_dir, 'naturalness_summary.tex'), 'w',
              encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))
    print('NATURALNESS_OK', flush=True)


if __name__ == '__main__':
    main()
