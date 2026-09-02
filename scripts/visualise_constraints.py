"""
visualise_constraints.py  --  publication figures for extracted constraints

Run from the caft directory:
    python src/visualise_constraints.py

Outputs (each ~4×3 inches):
    manuscript/figures/constraint_bar.pdf/.png
    manuscript/figures/constraint_graph.pdf/.png
    manuscript/figures/constraint_table.pdf/.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
sys.path.append(ROOT)

from src.caft.constraint_extractor import DatasetConstraintExtractor


def filter_constraints(constraints, min_score):
    filtered = dict(constraints)
    filtered["inter_attribute"] = [
        r for r in constraints.get("inter_attribute", [])
        if r.get("score", 0.0) >= min_score
    ]
    return filtered


def save(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  saved {stem}.pdf / .png")
    plt.close(fig)


# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join(BASE, "../datasets/adult_raw/adult.csv")
OUT_DIR         = os.path.join(BASE, "../manuscript/figures")
MIN_SCORE       = 0.97
N_BINS          = 5
CAT_THRESH      = 20
PROTECTED       = {"race", "sex", "age"}
GRAPH_MIN_SCORE = 0.99
EXCLUDE_ATTRS   = {"fnlwgt", "education-num"}   # data-collection artefact; not a meaningful feature

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & encode ─────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip()
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  {len(df)} rows, {len(df.columns)} columns")

encoders = {}
df_enc = df.copy()
for col in df.select_dtypes(include=["object", "string"]).columns:
    codes, uniques = pd.factorize(df_enc[col])
    df_enc[col] = codes
    encoders[col] = uniques


def decode(attr, val_str):
    if attr not in encoders:
        return val_str
    try:
        idx = int(float(val_str))
        labels = encoders[attr]
        return str(labels[idx]) if idx < len(labels) else val_str
    except (ValueError, IndexError):
        return val_str


# ── Extract & filter ──────────────────────────────────────────────────────────
print("Extracting constraints ...")
ext = DatasetConstraintExtractor(df_enc, n_bins=N_BINS, categorical_threshold=CAT_THRESH)
ext.extract_attribute_domain_constraints()
ext.extract_inter_attribute_constraints()
ext.extract_structural_constraints()
constraints = filter_constraints(ext.constraints, MIN_SCORE)

# Drop excluded attributes everywhere
constraints["attribute_domain"] = {
    k: v for k, v in constraints["attribute_domain"].items()
    if k not in EXCLUDE_ATTRS
}
constraints["inter_attribute"] = [
    r for r in constraints["inter_attribute"]
    if r["antecedent"].get("attribute") not in EXCLUDE_ATTRS
    and r.get("consequent_exclusion", {}).get("attribute") not in EXCLUDE_ATTRS
]
constraints["structural"] = [
    r for r in constraints["structural"]
    if r.get("attribute") not in EXCLUDE_ATTRS
]

dep_rules  = [r for r in constraints["inter_attribute"] if r["type"] == "dependency_exclusion"]
corr_rules = [r for r in constraints["inter_attribute"] if r["type"] == "correlation"]

n_dom     = len(constraints["attribute_domain"])
n_struct  = len(constraints["structural"])
n_cat_dep = sum(1 for r in dep_rules if "value"          in r["antecedent"])
n_bin_dep = sum(1 for r in dep_rules if "range"          in r["antecedent"])
n_sen_dep = sum(1 for r in dep_rules if "sentinel_value" in r["antecedent"])
n_corr    = len(corr_rules)

print(f"  Domain: {n_dom},  cat-dep: {n_cat_dep},  bin-dep: {n_bin_dep},  "
      f"sentinel: {n_sen_dep},  corr: {n_corr},  structural: {n_struct}")

# ── Save constraints as JSON ──────────────────────────────────────────────────
import json

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

json_path = os.path.join(OUT_DIR, "constraints.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(constraints, f, indent=2, cls=_NumpyEncoder)
print(f"  saved constraints.json  ({os.path.getsize(json_path) // 1024} KB)")

# Flat CSV 1: inter-attribute rules
# Each row is one rule; antecedent type is inferred from which fields are present.
inter_rows = []
for r in constraints["inter_attribute"]:
    if r["type"] == "correlation":
        inter_rows.append({
            "type":           "correlation",
            "ant_attribute":  r["attributes"][0],
            "ant_value":      "",
            "ant_range_min":  "",
            "ant_range_max":  "",
            "cons_attribute": r["attributes"][1],
            "cons_excluded":  "",
            "score":          r["score"],
        })
    else:  # dependency_exclusion
        ant  = r["antecedent"]
        cons = r["consequent_exclusion"]
        if "sentinel_value" in ant:
            ant_type, ant_val, lo, hi = "sentinel", ant["sentinel_value"], "", ""
        elif "range" in ant:
            ant_type = "binned"
            ant_val  = ""
            lo, hi   = ant["range"]["min"], ant["range"]["max"]
        else:
            ant_type, ant_val, lo, hi = "categorical", ant.get("value", ""), "", ""
        # Decode categorical value to original label
        ant_label = decode(ant["attribute"], str(ant_val)) if ant_val != "" else ""
        exc_labels = "; ".join(decode(cons["attribute"], str(v)) for v in cons["values"])
        inter_rows.append({
            "type":           ant_type,
            "ant_attribute":  ant["attribute"],
            "ant_value":      ant_label,
            "ant_range_min":  lo,
            "ant_range_max":  hi,
            "cons_attribute": cons["attribute"],
            "cons_excluded":  exc_labels,
            "score":          r["score"],
        })

inter_csv = os.path.join(OUT_DIR, "constraints_inter.csv")
pd.DataFrame(inter_rows).to_csv(inter_csv, index=False)
print(f"  saved constraints_inter.csv  ({len(inter_rows)} rows)")

# Flat CSV 2: attribute-domain constraints
domain_rows = []
for attr, spec in constraints["attribute_domain"].items():
    domain_rows.append({
        "attribute": attr,
        "dtype":     spec.get("type"),
        "min":       spec.get("min", ""),
        "max":       spec.get("max", ""),
        "values":    "; ".join(spec["values"]) if "values" in spec else "",
    })

domain_csv = os.path.join(OUT_DIR, "constraints_domain.csv")
pd.DataFrame(domain_rows).to_csv(domain_csv, index=False)
print(f"  saved constraints_domain.csv  ({len(domain_rows)} rows)")

# ── Palette ───────────────────────────────────────────────────────────────────
C_PROT = "#C0392B"
C_NORM = "#2980B9"
C_CAT  = "#E67E22"
C_BIN  = "#8E44AD"
C_SEN  = "#27AE60"

RCPARAMS = {
    "font.family":    "serif",
    "font.size":      8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — constraint counts bar chart
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(RCPARAMS)
fig1, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=True)

bar_labels = [
    "Structural",
    "Correlation",
    "Sentinel dep.",
    "Binned dep.",
    "Cat. dep.",
    "Attr. domain",
]
bar_vals = [n_struct, n_corr, n_sen_dep, n_bin_dep, n_cat_dep, n_dom]
bar_cols = ["#BDC3C7", "#95A5A6", C_SEN, C_BIN, C_CAT, C_NORM]

ax.barh(bar_labels, bar_vals, color=bar_cols, height=0.55, edgecolor="white")
for label, val in zip(bar_labels, bar_vals):
    offset = max(bar_vals) * 0.012   # small fixed gap so "0" labels don't touch axis
    ax.text(val + offset, bar_labels.index(label),
            str(val), va="center", fontsize=7.5, fontweight="bold", color="#2C3E50")

ax.set_xlim(0, max(bar_vals) * 1.32)
ax.set_xlabel("Count")
ax.set_title("(a) Extracted constraint counts", fontweight="bold", pad=5)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=7.5)

save(fig1, "constraint_bar")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — inter-attribute dependency graph
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_NAME = {
    "marital-status":  "marital-s.",
    "education-num":   "educ.-num",
    "native-country":  "native-c.",
    "hours-per-week":  "hrs/wk",
    "capital-gain":    "cap-gain",
    "capital-loss":    "cap-loss",
}

fig2, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=True)

high_conf = [r for r in dep_rules if r["score"] >= GRAPH_MIN_SCORE]
edge_meta = {}
for rule in high_conf:
    ant  = rule["antecedent"]
    cons = rule["consequent_exclusion"]
    src, dst = ant["attribute"], cons["attribute"]
    etype = ("sentinel" if "sentinel_value" in ant
             else "binned" if "range" in ant
             else "categorical")
    key = (src, dst)
    if key not in edge_meta or rule["score"] > edge_meta[key][1]:
        edge_meta[key] = (etype, rule["score"])

G = nx.DiGraph()
for (src, dst), (etype, score) in edge_meta.items():
    G.add_edge(src, dst, etype=etype, score=score)
G.remove_nodes_from(list(nx.isolates(G)))

pos = nx.spring_layout(G, seed=42, k=2.5)
pos = {n: np.array(p, dtype=float) for n, p in pos.items()}

node_colors = [C_PROT if n in PROTECTED else C_NORM for n in G.nodes()]
node_sizes  = [900 if n in PROTECTED else 750 for n in G.nodes()]
edge_colors = [
    {"categorical": C_CAT, "binned": C_BIN, "sentinel": C_SEN}[d["etype"]]
    for _, _, d in G.edges(data=True)
]
# Shorten long names so labels fit inside nodes
display_labels = {n: DISPLAY_NAME.get(n, n) for n in G.nodes()}

nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors, node_size=node_sizes, alpha=0.92)
for node, (x, y) in pos.items():
    ax.text(x, y, display_labels[node] or node,
            fontsize=5.5, fontweight="bold", color="white",
            ha="center", va="center",
            path_effects=[mpe.withStroke(linewidth=1.8, foreground="#1a1a1a")])
nx.draw_networkx_edges(G, pos, ax=ax,
                       edge_color=edge_colors, arrows=True, arrowsize=10,
                       width=1.2, alpha=0.70,
                       connectionstyle="arc3,rad=0.10",
                       node_size=600)

legend_handles = [
    mpatches.Patch(color=C_PROT, label="Protected attr."),
    mpatches.Patch(color=C_NORM, label="Non-protected attr."),
    mpatches.Patch(color=C_CAT,  label="Categorical dep."),
    mpatches.Patch(color=C_BIN,  label="Binned dep."),
    mpatches.Patch(color=C_SEN,  label="Sentinel dep."),
]
ax.legend(handles=legend_handles, loc="lower left",
          fontsize=6, framealpha=0.85, ncol=1)
ax.set_title(
    f"(b) Dependency graph  (conf $\\geq$ {GRAPH_MIN_SCORE},  $n={len(edge_meta)}$ edges)",
    fontweight="bold", pad=5,
)
ax.axis("off")

save(fig2, "constraint_graph")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — sample dependency rules table
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
ax.axis("off")
ax.set_title("(c) Sample dependency rules", fontweight="bold", pad=5)


def rule_to_row(rule):
    ant  = rule["antecedent"]
    cons = rule["consequent_exclusion"]
    attr_a, attr_c = ant["attribute"], cons["attribute"]
    if "sentinel_value" in ant:
        ant_str = f"{attr_a} = {ant['sentinel_value']:.0f}"
    elif "range" in ant:
        lo, hi = ant["range"]["min"], ant["range"]["max"]
        ant_str = f"{attr_a} $\\in$ [{lo:.0f},{hi:.0f}]"
    else:
        label   = decode(attr_a, str(ant.get("value", "?")))
        label   = label[:15] + "…" if len(label) > 15 else label
        ant_str = f"{attr_a} = {label}"
    vals         = cons["values"]
    decoded_vals = [decode(attr_c, str(v)) for v in vals]
    short_vals   = [v[:11] for v in decoded_vals[:2]]
    exc_str = "{" + ", ".join(short_vals)
    if len(vals) > 2:
        exc_str += f", +{len(vals)-2}"
    exc_str += "}"
    return [ant_str, f"{attr_c} ∉ {exc_str}", f"{rule['score']:.2f}"]


selected = []
for kind, limit in [("categorical", 2), ("binned", 2), ("sentinel", 1)]:
    count = 0
    for rule in sorted(dep_rules, key=lambda r: -r["score"]):
        if count >= limit:
            break
        ant = rule["antecedent"]
        if   kind == "sentinel"    and "sentinel_value" in ant:
            selected.append(rule); count += 1
        elif kind == "binned"      and "range" in ant and "sentinel_value" not in ant:
            selected.append(rule); count += 1
        elif kind == "categorical" and "value" in ant:
            selected.append(rule); count += 1

rows = [rule_to_row(r) for r in selected[:5]]

col_labels = ["Antecedent", "Exclusion", "Conf."]
tbl = ax.table(
    cellText=rows, colLabels=col_labels,
    cellLoc="left", loc="center",
    bbox=[0.0, 0.02, 1.0, 0.90],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)

col_widths = [0.37, 0.47, 0.16]
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_width(col_widths[ci])

for j in range(3):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, len(rows) + 1):
    bg = "#EBF5FB" if i % 2 == 0 else "white"
    for j in range(3):
        tbl[i, j].set_facecolor(bg)
        tbl[i, j].set_edgecolor("#D5D8DC")

save(fig3, "constraint_table")

print("\nDone.")
