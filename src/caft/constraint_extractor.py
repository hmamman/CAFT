import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Any
import re

class DatasetConstraintExtractor:
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_threshold: int = 20,
        correlation_threshold: float = 0.8,
        min_confidence: float = 0.95,
        n_bins: int = 5
    ):
        self.df = df.copy()
        self.categorical_threshold = categorical_threshold
        self.correlation_threshold = correlation_threshold
        self.min_confidence = min_confidence
        self.n_bins = n_bins

        self.constraints = {
            "attribute_domain": {},
            "inter_attribute": [],
            "structural": []
        }
        
    def infer_attribute_types(self) -> Dict[str, str]:
        types = {}
        for col in self.df.columns:
            is_cat = (self.df[col].dtype == 'object' or 
                     self.df[col].nunique() <= self.categorical_threshold)
            types[col] = "categorical" if is_cat else "numerical"
        return types

    def extract_attribute_domain_constraints(self) -> Dict[str, Any]:
        """Extract min/max for numerical and unique values for categorical."""
        types = self.infer_attribute_types()
        for col in self.df.columns:
            series = self.df[col].dropna()
            if series.empty:
                continue
            if types[col] == "numerical":
                self.constraints["attribute_domain"][col] = {
                    "type": "numerical",
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "dtype": str(self.df[col].dtype)
                }
            else:
                self.constraints["attribute_domain"][col] = {
                    "type": "categorical",
                    "values": sorted([str(x) for x in series.unique()])
                }
        return self.constraints["attribute_domain"]

    def extract_inter_attribute_constraints(self) -> List[Dict[str, Any]]:
        types = self.infer_attribute_types()
        num_cols = [c for c, t in types.items() if t == "numerical"]
        cat_cols = [c for c, t in types.items() if t == "categorical"]
        
        # 1. Correlations
        if len(num_cols) > 1:
            corr_matrix = self.df[num_cols].corr().abs()
            for c1, c2 in combinations(num_cols, 2):
                val = corr_matrix.loc[c1, c2]
                if val >= self.correlation_threshold:
                    self.constraints["inter_attribute"].append({
                        "type": "correlation",
                        "attributes": [c1, c2],
                        "strength": float(val),
                        "score": float(val) # Use correlation as score
                    })

        # 2. Categorical -> Categorical
        for c1, c2 in combinations(cat_cols, 2):
            self._check_dependency(c1, c2)
            self._check_dependency(c2, c1) 
            
        # 3. Sentinel value dependencies (e.g., pdays=-1 => previous=0)
        # A sentinel is a numerical value that is both frequent (>= 5% of rows)
        # and statistically distant from the column mean (>= 2 std deviations).
        # Such values encode a special state (e.g. "never contacted") and often
        # constrain other columns to specific values.
        for num_col in num_cols:
            for sentinel in self._detect_sentinels(num_col):
                self._check_sentinel_dependency(num_col, sentinel, cat_cols + num_cols)

        # 4. Numerical (Binned) -> Categorical
        for num_col in num_cols:
            try:
                # Use retbins=True to get edges if needed, but qcut intervals are enough
                binned_series, edges = pd.qcut(self.df[num_col], self.n_bins, duplicates='drop', retbins=True)
                
                # We need to process each Categorical column against this binned series
                for cat_col in cat_cols:
                    # Create a temp DF where 'source' is the INTERVAL object (not string yet)
                    temp_data = pd.DataFrame({
                        'source': binned_series,
                        'target': self.df[cat_col]
                    })
                    
                    # Custom check for intervals
                    self._check_interval_dependency(temp_data, 'source', 'target', num_col, cat_col)

            except ValueError:
                continue

        return self.constraints["inter_attribute"]
        
    def _check_interval_dependency(self, df_subset, src_col, tgt_col, src_orig_name, tgt_name):
        """Handle Interval objects in source column."""
        ct = pd.crosstab(df_subset[src_col], df_subset[tgt_col], normalize='index')
        
        exclusion_threshold = 1.0 - self.min_confidence
        
        for interval in ct.index:
            # interval is a pandas Interval object
            row_probs = ct.loc[interval]
            excluded_vals = row_probs[row_probs <= exclusion_threshold].index.tolist()
            
            if excluded_vals:
                # prob of excluded value is low (<= threshold). 
                # Score can be 1.0 - max(prob_of_excluded_vals) or just 1.0 for hard exclusion?
                # Let's use 1.0 - max(probs of excluded items) as 'confidence' of exclusion
                confidence = 1.0 - row_probs[excluded_vals].max()
                
                self.constraints["inter_attribute"].append({
                    "type": "dependency_exclusion",
                    "antecedent": {
                        "attribute": src_orig_name, 
                        "range": {"min": float(interval.left), "max": float(interval.right)},
                        "description": str(interval)
                    },
                    "consequent_exclusion": {"attribute": tgt_name, "values": excluded_vals},
                    "score": float(confidence)
                })

    def _detect_sentinels(self, col: str) -> List[float]:
        """
        Find sentinel values in a numerical column.
        A sentinel must satisfy both:
          - frequency >= 5% of non-null rows (common enough to be meaningful)
          - |value - mean| >= 2 * std  (statistically distinct from the bulk)
        """
        series = self.df[col].dropna()
        if len(series) < 10:
            return []
        mean, std = float(series.mean()), float(series.std())
        if std == 0:
            return []
        total = len(series)
        sentinels = []
        for val, count in series.value_counts().items():
            if (count / total) >= 0.05 and abs(float(val) - mean) / std >= 2.0:
                sentinels.append(float(val))
        return sentinels

    def _check_sentinel_dependency(self, sentinel_col: str, sentinel_val: float,
                                   target_cols: List[str]) -> None:
        """
        When sentinel_col == sentinel_val, check whether any other column
        is constrained to a near-certain value.  Extracts a dependency_exclusion
        rule with antecedent type 'sentinel' so the validator can use numeric
        equality instead of string matching.
        """
        mask = self.df[sentinel_col] == sentinel_val
        subset = self.df[mask]
        if len(subset) < 5:
            return

        exclusion_threshold = 1.0 - self.min_confidence

        for tgt_col in target_cols:
            if tgt_col == sentinel_col:
                continue
            tgt_series = subset[tgt_col].dropna()
            if tgt_series.empty:
                continue

            value_counts = tgt_series.value_counts(normalize=True)
            top_val = value_counts.index[0]
            top_freq = float(value_counts.iloc[0])

            if top_freq >= self.min_confidence:
                # All values other than top_val are effectively excluded
                excluded = [str(v) for v in value_counts.index[1:]
                            if float(value_counts[v]) <= exclusion_threshold]
                if excluded:
                    self.constraints["inter_attribute"].append({
                        "type": "dependency_exclusion",
                        "antecedent": {
                            "attribute": sentinel_col,
                            "sentinel_value": sentinel_val,   # numeric — compared as float
                        },
                        "consequent_exclusion": {
                            "attribute": tgt_col,
                            "values": excluded,
                        },
                        "score": float(top_freq),
                    })

    def _check_dependency(self, col1, col2):
        """Standard Categorical check."""
        ct = pd.crosstab(self.df[col1], self.df[col2], normalize='index')
        exclusion_threshold = 1.0 - self.min_confidence
        
        for val in ct.index:
            row_probs = ct.loc[val]
            excluded_vals = row_probs[row_probs <= exclusion_threshold].index.tolist()
            if excluded_vals:
                confidence = 1.0 - row_probs[excluded_vals].max()
                self.constraints["inter_attribute"].append({
                    "type": "dependency_exclusion",
                    "antecedent": {"attribute": col1, "value": str(val)},
                    "consequent_exclusion": {"attribute": col2, "values": excluded_vals},
                    "score": float(confidence)
                })

    def extract_structural_constraints(self) -> List[Dict[str, Any]]:
        # Missingness
        for col in self.df.columns:
            if not self.df[col].isna().any():
                 self.constraints["structural"].append({
                    "type": "non_null",
                    "attribute": col
                })
        
        # Immutability
        immutable_keywords = ["sex", "race", "gender", "age", "native-country"] 
        for col in self.df.columns:
            if any(k in col.lower() for k in immutable_keywords):
                self.constraints["structural"].append({
                    "type": "immutable_candidate",
                    "attribute": col
                })
        return self.constraints["structural"]

    def to_llm_specification(self) -> str:
        """Human-readable constraint summary for debugging and inspection only.
        The generation pipeline uses LLMInterface.create_generation_prompt() instead."""
        spec_lines = ["Dataset Constraints:"]
        
        # Domain
        spec_lines.append("\n1. Attribute Domains:")
        for attr, details in self.constraints["attribute_domain"].items():
            if details["type"] == "numerical":
                spec_lines.append(f"   - {attr}: Numerical, range [{details['min']}, {details['max']}]")
            else:
                vals = ", ".join(details["values"][:10]) 
                if len(details["values"]) > 10: vals += "..."
                spec_lines.append(f"   - {attr}: Categorical, allowed types: {{{vals}}}")
                
        # Inter-attribute
        if self.constraints["inter_attribute"]:
            spec_lines.append("\n2. Dependencies:")
            for rule in self.constraints["inter_attribute"]:
                if rule["type"] == "correlation":
                    spec_lines.append(f"   - High correlation between {rule['attributes'][0]} and {rule['attributes'][1]}")
                elif rule["type"] == "dependency_exclusion":
                    ant = rule['antecedent']
                    cons = rule['consequent_exclusion']
                    
                    if "range" in ant:
                        # Interval rule
                        r = ant['range']
                        spec_lines.append(f"   - IF {ant['attribute']} is between {r['min']} and {r['max']}, THEN {cons['attribute']} cannot be {cons['values']}")
                    else:
                        # Value rule
                        spec_lines.append(f"   - IF {ant['attribute']} is '{ant['value']}' THEN {cons['attribute']} cannot be {cons['values']}")
                    
        return "\n".join(spec_lines)
