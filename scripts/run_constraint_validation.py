"""
RQ4: constraint-quality validation -- do the mined constraints generalise, or
are they an artefact of the training sample that CAFT games?

This addresses the central threat to the validity study: CAFT searches toward a
constraint set C_tau and is then validated against the same set, so a high VDR
could be circular. Two independent checks:

RQ4a -- Rule generalisation (held-out precision). Mine C_train on a training
  split, then re-evaluate every dependency-exclusion rule on a disjoint holdout
  split. A rule "holds" on holdout when its confidence there is still >= tau.
  High held-out precision means the rules are real structural properties of the
  data distribution, not sampling noise the search exploits.

RQ4b -- Cross-split independence. Mine a second constraint set C_holdout on the
  holdout split. (i) Rule-set stability: Jaccard overlap of the two rule sets.
  (ii) Validity-label agreement: for a common pool of instances (real holdout
  rows and random inputs), does validity under C_train agree with validity
  under C_holdout? High agreement (Cohen's kappa) means an instance's validity
  is a property of the data, not of which sample the constraints came from, so
  CAFT's validity guarantee is not an artefact of one specific C_tau.

Constraint extraction mirrors CAFT exactly (n_bins=5, categorical_threshold=20,
min_score=tau). No model is loaded; this operates purely on the data and the
extracted rules.

Run from the repo root:
    python scripts/run_constraint_validation.py
Outputs CSV + LaTeX under results/rq4/.
"""
import os
import sys

import numpy as np
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(base_path, ".."))
os.chdir(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.utils.helpers import get_config_dict
from src.caft.constraint_extractor import DatasetConstraintExtractor
from src.caft.caft import filter_constraints

DATASETS   = ['census', 'compas', 'credit', 'bank', 'meps']
DS_MAP     = {'census': 'Census', 'compas': 'COMPAS', 'credit': 'Credit',
              'bank': 'Bank', 'meps': 'MEPS'}
TAU        = 0.97          # min_score, matching CAFT's default
N_BINS     = 5
CAT_THRESH = 20
HOLDOUT_FRAC = 0.30
MIN_SUPPORT  = 20         # a rule needs this many antecedent matches on holdout to be testable
SAMPLE_N     = 4000       # instances per pool for the validity-agreement check
SEED         = 42
OUT_DIR = os.path.join('results', 'rq4')


def load_encoded(config):
    feature_names = list(config.feature_name)
    rows = []
    with open(os.path.join('datasets', config.dataset_name)) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(',')
            rows.append([float(x) for x in parts[:len(feature_names)]])
    return pd.DataFrame(rows, columns=feature_names).astype(int)


def mine(df, categorical_cols):
    """Mine and confidence-filter constraints on df, mirroring CAFT.__init__."""
    df_str = df.copy()
    for col in categorical_cols:
        df_str[col] = df_str[col].astype(str)
    ex = DatasetConstraintExtractor(df_str, n_bins=N_BINS,
                                    categorical_threshold=CAT_THRESH)
    ex.extract_attribute_domain_constraints()
    ex.extract_inter_attribute_constraints()
    ex.extract_structural_constraints()
    return filter_constraints(ex.constraints, TAU)


def dep_rules(constraints):
    return [r for r in constraints["inter_attribute"]
            if r.get("type") == "dependency_exclusion"]


def rule_subtype(rule):
    ant = rule["antecedent"]
    if "range" in ant:
        return "range"
    if "sentinel_value" in ant:
        return "sentinel"
    return "value"


def rule_key(rule):
    """Canonical identity for cross-split matching: antecedent attr+condition
    and consequent attr+excluded set."""
    ant = rule["antecedent"]
    if "range" in ant:
        a = (ant["attribute"], "range", round(ant["range"]["min"], 4),
             round(ant["range"]["max"], 4))
    elif "sentinel_value" in ant:
        a = (ant["attribute"], "sentinel", float(ant["sentinel_value"]))
    else:
        a = (ant["attribute"], "value", str(ant["value"]))
    cons = rule["consequent_exclusion"]
    c = (cons["attribute"], tuple(sorted(str(v) for v in cons["values"])))
    return (a, c)


def antecedent_mask(rule, X, fidx):
    ant = rule["antecedent"]
    col = X[:, fidx[ant["attribute"]]]
    if "range" in ant:
        return (col > ant["range"]["min"]) & (col <= ant["range"]["max"])
    if "sentinel_value" in ant:
        return col == float(ant["sentinel_value"])
    return col == float(ant["value"])


def consequent_violated(rule, X, fidx):
    cons = rule["consequent_exclusion"]
    forbidden = [int(v) for v in cons["values"]]
    return np.isin(X[:, fidx[cons["attribute"]]], forbidden)


def holdout_confidence(rule, X, fidx):
    """Confidence of a dependency-exclusion rule on data X: 1 - P(excluded | antecedent)."""
    am = antecedent_mask(rule, X, fidx)
    support = int(am.sum())
    if support < MIN_SUPPORT:
        return None, support
    viol = int((am & consequent_violated(rule, X, fidx)).sum())
    return 1.0 - viol / support, support


def valid_mask(constraints, X, fidx):
    """Boolean validity per row against a constraint set (inter-attribute
    dependency-exclusion rules plus attribute-domain bounds)."""
    n = len(X)
    viol = np.zeros(n, dtype=np.int64)
    for rule in dep_rules(constraints):
        viol += (antecedent_mask(rule, X, fidx) &
                 consequent_violated(rule, X, fidx)).astype(np.int64)
    for attr, spec in constraints["attribute_domain"].items():
        col = X[:, fidx[attr]]
        if spec["type"] == "numerical":
            viol += ((col < spec["min"]) | (col > spec["max"])).astype(np.int64)
        else:
            allowed = [int(v) for v in spec["values"]]
            viol += (~np.isin(col, allowed)).astype(np.int64)
    return viol == 0


def cohen_kappa(a, b):
    """Cohen's kappa for two boolean labelings."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n = len(a)
    po = np.mean(a == b)
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0


def to_latex(df, path, caption, label, float_format='%.2f'):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(df.to_latex(index=False, escape=True, na_rep='--',
                            float_format=float_format,
                            caption=caption, label=label, position='ht'))


# Dataset display order: the constraint-concentration ordering used by every
# other table in the manuscript (Table tab:feasibility), so the RQ4 tables can
# be read against them row by row.
FEAS_ORDER = ['Census', 'Bank', 'Credit', 'COMPAS', 'MEPS']


def _ordered(df):
    """Reindex a per-dataset frame into the manuscript's dataset order."""
    present = [d for d in FEAS_ORDER if d in set(df.dataset)]
    return pd.concat([df[df.dataset == d] for d in present], ignore_index=True)


def _n(v):
    return f'{int(round(v)):,}'


def write_generalisation_table(gen, path):
    """RQ4a: rules mined on the training split, re-scored on the holdout.

    Written by hand rather than with DataFrame.to_latex so the header carries
    typeset column names and units instead of the raw frame identifiers, and so
    each column gets the precision its claim needs: the confidence columns are
    quoted to three decimals in the prose (0.962--0.990, drops of 0.019 and
    0.002) and would be unreadable at the two decimals to_latex applied.
    """
    sub = _ordered(gen[gen.rule_type == 'ALL'])
    body = [
        f'{r.dataset} & {_n(r.n_rules)} & {_n(r.testable)} & '
        f'{r.pct_held:.1f} & {r.mean_holdout_conf:.3f} & {r.mean_conf_drop:.3f} \\\\'
        for r in sub.itertuples()
    ]
    L = ['\\begin{table}[!htb]', '\\centering',
         '\\caption{RQ4a rule generalisation. Dependency-exclusion rules are '
         'mined on a 70\\% training split and re-scored on the disjoint 30\\% '
         'holdout. \\emph{Mined} counts the rules extracted; \\emph{testable} '
         'counts those whose antecedent occurs in the holdout and can therefore '
         'be re-scored. \\emph{Retained} is the percentage of testable rules '
         f'whose holdout confidence remains at least $\\tau={TAU}$, a threshold '
         'test that is strict at the margin; \\emph{mean} and \\emph{drop} give '
         'the average holdout confidence and the average confidence lost from '
         'training to holdout, which show how far the underlying confidence '
         'actually moves. Rule counts differ from Table~\\ref{tab:feasibility} '
         'because those are mined on the full dataset and these on 70\\% of it. '
         'Datasets are ordered by the constraint-concentration proxy '
         '$\\log_{10}\\widehat{\\phi}_{\\mathrm{ind}}$.}',
         f'\\label{{{"tab:rq4-generalisation"}}}', '\\small',
         '\\begin{tabular}{@{}lrrrrr@{}}', '\\toprule',
         ' & \\multicolumn{2}{c}{Rules} & '
         '\\multicolumn{3}{c}{Holdout confidence} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}',
         'Dataset & Mined & Testable & Retained (\\%) & Mean & Drop \\\\',
         '\\midrule'] + body + \
        ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))


def write_independence_table(indep, path):
    """RQ4b: agreement between two independently mined constraint sets."""
    sub = _ordered(indep)
    body = [
        f'{r.dataset} & {_n(r.rules_train)} & {_n(r.rules_holdout)} & '
        # kappa at three decimals: Credit's 0.695 rounds to 0.69 at two, which
        # reads as a materially weaker agreement than it is and contradicts the
        # range quoted in the prose.
        f'{r.jaccard:.3f} & {r.validity_agreement:.1f} & {r.cohen_kappa:.3f} \\\\'
        for r in sub.itertuples()
    ]
    L = ['\\begin{table}[!htb]', '\\centering',
         '\\caption{RQ4b cross-sample agreement. A second constraint set is '
         'mined independently on the holdout split and compared with the set '
         'mined on the training split. \\emph{Jaccard} is the overlap of the two '
         'rule sets as sets of rules; \\emph{agreement} and \\emph{Cohen\'s '
         '$\\kappa$} measure how often an instance receives the same validity '
         'label under the two sets, over a pool of real holdout rows and random '
         'inputs. Rule-set overlap and label agreement diverge because many '
         'rules are near-duplicates that exclude the same instances: Jaccard '
         'counts rules, agreement counts decisions. Agreement measures '
         'cross-sample stability, not correspondence with domain ground truth. '
         'Datasets are ordered by the constraint-concentration proxy '
         '$\\log_{10}\\widehat{\\phi}_{\\mathrm{ind}}$.}',
         f'\\label{{{"tab:rq4-independence"}}}', '\\small',
         '\\begin{tabular}{@{}lrrrrr@{}}', '\\toprule',
         ' & \\multicolumn{2}{c}{Rules mined} & & '
         '\\multicolumn{2}{c}{Validity-label agreement} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}',
         'Dataset & Train & Holdout & Jaccard & Agreement (\\%) & '
         'Cohen\'s $\\kappa$ \\\\',
         '\\midrule'] + body + \
        ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = get_config_dict()
    rng = np.random.default_rng(SEED)
    gen_rows, indep_rows = [], []

    for ds in DATASETS:
        config = cfg[ds]
        try:
            df = load_encoded(config)
        except FileNotFoundError:
            print(f"  {ds}: dataset file missing, skip", flush=True)
            continue
        fidx = {name: i for i, name in enumerate(config.feature_name)}
        categorical_cols = {c for c in df.columns if df[c].nunique() <= CAT_THRESH}

        # Disjoint train / holdout split.
        idx = rng.permutation(len(df))
        cut = int(len(df) * (1 - HOLDOUT_FRAC))
        tr = df.iloc[idx[:cut]].reset_index(drop=True)
        ho = df.iloc[idx[cut:]].reset_index(drop=True)
        Xho = ho.values.astype(np.int64)

        c_train = mine(tr, categorical_cols)
        c_hold  = mine(ho, categorical_cols)
        r_train = dep_rules(c_train)
        print(f"\n[{ds}] train={len(tr)} holdout={len(ho)}  "
              f"train rules={len(r_train)}  holdout rules={len(dep_rules(c_hold))}",
              flush=True)

        # ---- RQ4a: held-out precision of each train rule ----
        by_type = {}
        for rule in r_train:
            st = rule_subtype(rule)
            conf, support = holdout_confidence(rule, Xho, fidx)
            d = by_type.setdefault(st, {'n': 0, 'testable': 0, 'held': 0,
                                        'conf': [], 'drop': []})
            d['n'] += 1
            if conf is not None:
                d['testable'] += 1
                d['conf'].append(conf)
                d['drop'].append(rule.get('score', 1.0) - conf)
                if conf >= TAU:
                    d['held'] += 1
        for st, d in sorted(by_type.items()):
            gen_rows.append({
                'dataset': DS_MAP[ds], 'rule_type': st,
                'n_rules': d['n'], 'testable': d['testable'],
                'pct_held': round(100 * d['held'] / d['testable'], 1) if d['testable'] else np.nan,
                'mean_holdout_conf': round(float(np.mean(d['conf'])), 4) if d['conf'] else np.nan,
                'mean_conf_drop': round(float(np.mean(d['drop'])), 4) if d['drop'] else np.nan,
            })
        # Dataset-level rollup.
        allc = [c for d in by_type.values() for c in d['conf']]
        n_test = sum(d['testable'] for d in by_type.values())
        n_held = sum(d['held'] for d in by_type.values())
        gen_rows.append({
            'dataset': DS_MAP[ds], 'rule_type': 'ALL',
            'n_rules': sum(d['n'] for d in by_type.values()),
            'testable': n_test,
            'pct_held': round(100 * n_held / n_test, 1) if n_test else np.nan,
            'mean_holdout_conf': round(float(np.mean(allc)), 4) if allc else np.nan,
            'mean_conf_drop': round(float(np.mean(
                [x for d in by_type.values() for x in d['drop']])), 4) if allc else np.nan,
        })

        # ---- RQ4b(i): rule-set stability (Jaccard) ----
        k_train = {rule_key(r) for r in r_train}
        k_hold  = {rule_key(r) for r in dep_rules(c_hold)}
        inter = len(k_train & k_hold)
        union = len(k_train | k_hold)
        jaccard = inter / union if union else np.nan

        # ---- RQ4b(ii): validity-label agreement on a shared instance pool ----
        real = Xho[rng.choice(len(Xho), min(SAMPLE_N, len(Xho)), replace=False)]
        bounds = np.array(config.input_bounds)
        rand = np.column_stack([
            rng.integers(int(bounds[j, 0]), int(bounds[j, 1]) + 1, SAMPLE_N)
            for j in range(len(config.feature_name))
        ])
        pool = np.vstack([real, rand])
        v_train = valid_mask(c_train, pool, fidx)
        v_hold  = valid_mask(c_hold, pool, fidx)
        agree = float(np.mean(v_train == v_hold))
        kappa = cohen_kappa(v_train, v_hold)

        indep_rows.append({
            'dataset': DS_MAP[ds],
            'rules_train': len(k_train), 'rules_holdout': len(k_hold),
            'jaccard': round(jaccard, 3),
            'validity_agreement': round(100 * agree, 1),
            'cohen_kappa': round(kappa, 3),
        })
        print(f"  RQ4a: {n_held}/{n_test} rules hold at tau={TAU} "
              f"({100*n_held/max(n_test,1):.1f}%), mean holdout conf "
              f"{np.mean(allc) if allc else float('nan'):.3f}", flush=True)
        print(f"  RQ4b: Jaccard={jaccard:.3f}  agreement={100*agree:.1f}%  "
              f"kappa={kappa:.3f}", flush=True)

    gen = pd.DataFrame(gen_rows)
    indep = pd.DataFrame(indep_rows)
    gen.to_csv(os.path.join(OUT_DIR, 'rq4_generalisation.csv'), index=False)
    indep.to_csv(os.path.join(OUT_DIR, 'rq4_independence.csv'), index=False)

    write_generalisation_table(gen, os.path.join(OUT_DIR,
                                                 'rq4_generalisation.tex'))
    write_independence_table(indep, os.path.join(OUT_DIR,
                                                 'rq4_independence.tex'))
    print(f"\nwrote RQ4 tables to {OUT_DIR}")
    print("RQ4_OK", flush=True)


if __name__ == '__main__':
    main()
