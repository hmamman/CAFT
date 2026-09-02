import itertools
import math
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

# Path setup
base_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(base_path, "../.."))   # repo root
sys.path.append(path)

from src.caft.genetic_algorithm import GA
from src.caft.constraint_extractor import DatasetConstraintExtractor
from src.caft.validator import ConstraintValidator
from src.utils.helpers import get_experiment_params, generate_report
from src.utils.dnn_wrapper import dnn_model_wrapper


def filter_constraints(constraints, min_score):
    """Drop inter-attribute rules whose confidence is below min_score."""
    filtered = dict(constraints)
    filtered["inter_attribute"] = [
        r for r in constraints.get("inter_attribute", [])
        if r.get("score", 0.0) >= min_score
    ]
    return filtered


class CAFT:
    def __init__(self, config, model, classifier_name, sensitive_params,
                 constraint_mode='penalty', lambda_penalty=1.0, lambda_gradient=0.05,
                 min_score=0.97,
                 population_size=300, max_passes=5,
                 seed_strategy='random', n_clusters=10, seed_fashion='RoundRobin',
                 crossover_strategy='hux', mutation_strategy='hybrid',
                 sigma=0.05, categorical_threshold=20, ga_categorical_threshold=5,
                 lambda_redundancy=2.0, n_bins=5,
                 repair_interval=50, repair_bursts=2,
                 adaptive_target=0.5,
                 save_all_disc=False,
                 selection_strategy='tournament'):
        """Configure the constraint-aware GA; constraint_mode selects the enforcement strategy (Section 3.3)."""
        print(f"\nInitializing CAFT with constraint_mode={constraint_mode}, "
              f"lambda_penalty={lambda_penalty}, lambda_gradient={lambda_gradient}, "
              f"min_score={min_score}, "
              f"max_passes={max_passes}, seed_strategy={seed_strategy}, "
              f"n_clusters={n_clusters}, seed_fashion={seed_fashion}, "
              f"lambda_redundancy={lambda_redundancy}, repair_interval={repair_interval}, "
              f"repair_bursts={repair_bursts}")
        if constraint_mode not in ('none', 'penalty', 'repair', 'hybrid', 'adaptive', 'repair_penalty'):
            raise ValueError("constraint_mode must be 'none', 'penalty', 'repair', 'hybrid', "
                              "'adaptive', or 'repair_penalty'")

        self.start_time = time.time()
        self.config = config
        self.constraint_mode = constraint_mode
        self.lambda_penalty = lambda_penalty
        self.lambda_gradient = lambda_gradient
        self.max_passes = max_passes
        self.repair_interval = repair_interval
        self.repair_bursts = repair_bursts
        self.adaptive_target = adaptive_target      # target validity fraction for 'adaptive' mode
        self._last_valid_frac = 1.0                 # population validity from the previous generation
        self.population_size = population_size
        self.lambda_redundancy = lambda_redundancy
        self.selection_strategy = selection_strategy
        self._gen = 0  # generation counter for hybrid scheduling
        self.classifier_name = classifier_name
        self.model = dnn_model_wrapper(model=model)
        self.sensitive_params = sensitive_params
        df_encoded = self.get_dataset()

        # Feature metadata
        self.feature_names = list(config.feature_name)
        self.num_attribs = len(self.feature_names)
        self.input_bounds = np.array(config.input_bounds)
        self._feature_idx = {name: i for i, name in enumerate(self.feature_names)}

        self.protected_attribs = [s - 1 for s in sensitive_params]
        self.non_protected_attribs = [
            i for i in range(self.num_attribs) if i not in self.protected_attribs
        ]
        self.protected_domains = [
            list(range(int(self.input_bounds[i][0]), int(self.input_bounds[i][1]) + 1))
            for i in self.protected_attribs
        ]

        # Constraint extraction
        self._categorical_cols = {
            col for col in df_encoded.columns
            if df_encoded[col].nunique() <= categorical_threshold
        }
        df_str = df_encoded.copy()
        for col in self._categorical_cols:
            df_str[col] = df_str[col].astype(str)

        extractor = DatasetConstraintExtractor(
            df_str, n_bins=n_bins, categorical_threshold=categorical_threshold
        )
        print("  Extracting attribute domain constraints ...")
        extractor.extract_attribute_domain_constraints()
        print("  Extracting inter-attribute constraints ...")
        extractor.extract_inter_attribute_constraints()
        print("  Extracting structural constraints ...")
        extractor.extract_structural_constraints()
        raw = extractor.constraints
        self.constraints = filter_constraints(raw, min_score)
        self.validator = ConstraintValidator(self.constraints)

        n_raw = (len(raw["inter_attribute"]) + len(raw["attribute_domain"])
                 + len(raw["structural"]))
        n_active = (len(self.constraints["inter_attribute"])
                    + len(self.constraints["attribute_domain"])
                    + len(self.constraints["structural"]))
        print(f"  Constraints: {n_raw} extracted → {n_active} active "
              f"(min_score={min_score}, dropped {n_raw - n_active})")

        # Partition rules into shared (non-protected attributes only) vs variant-specific
        prot_names = {self.feature_names[i] for i in self.protected_attribs}

        shared_inter = []
        variant_inter = []
        for rule in self.constraints["inter_attribute"]:
            if rule.get("type") != "dependency_exclusion":
                continue
            ant_attr = rule["antecedent"]["attribute"]
            cons_attr = rule["consequent_exclusion"]["attribute"]
            if ant_attr in prot_names or cons_attr in prot_names:
                variant_inter.append(rule)
            else:
                shared_inter.append(rule)

        shared_domain = {}
        variant_domain = {}
        for attr, spec in self.constraints["attribute_domain"].items():
            if attr in prot_names:
                variant_domain[attr] = spec
            else:
                shared_domain[attr] = spec

        print(f"  Variant guard partition: {len(shared_inter)} shared inter-rules, "
              f"{len(variant_inter)} variant-specific inter-rules "
              f"(protected attrs: {sorted(prot_names)})")

        # Vectorised rule evaluation arrays
        self._shared_rule_arrs = self._build_rule_arrays(shared_inter)
        self._variant_rule_arrs = self._build_rule_arrays(variant_inter)
        self._shared_domain_arrs = self._build_domain_arrays(shared_domain)
        self._variant_domain_arrs = self._build_domain_arrays(variant_domain)

        # Repair index (repair mode only)
        self._repair_rules = [
            r for r in self.constraints["inter_attribute"]
            if r["type"] == "dependency_exclusion"
        ]
        self._value_freq = {}
        for col in self._categorical_cols:
            if col in df_str.columns:
                self._value_freq[col] = df_str[col].value_counts().to_dict()

        for rule in self._repair_rules:
            ant = rule["antecedent"]
            tgt = rule["consequent_exclusion"]["attribute"]
            forbidden_ints = {int(v) for v in rule["consequent_exclusion"]["values"]}
            freq = self._value_freq.get(tgt, {})
            sorted_valid = [
                v for v, _ in sorted(freq.items(), key=lambda x: -x[1])
                if int(v) not in forbidden_ints
            ]

            rule["_ant_col"] = self._feature_idx.get(ant["attribute"], -1)
            if "range" in ant:
                rule["_ant_type"] = 1  # 1 = range
                rule["_lo"] = float(ant["range"]["min"])
                rule["_hi"] = float(ant["range"]["max"])
                rule["_ant_v"] = 0.0
            else:
                rule["_ant_type"] = 0  # 0 = equality (value or sentinel)
                rule["_ant_v"] = float(ant.get("sentinel_value",
                                       ant.get("value", 0)))
                rule["_lo"] = rule["_hi"] = 0.0

            rule["_col_idx"] = self._feature_idx.get(tgt, -1)
            rule["_replacement_int"] = int(sorted_valid[0]) if sorted_valid else None
            rule["_forbidden"] = np.array(
                sorted(forbidden_ints), dtype=np.int64
            )

            rule["_forbidden_ints"] = forbidden_ints
            rule["_replacements"] = sorted_valid

        _prot_set = set(self.protected_attribs)
        self._non_sens_mask = np.array(
            [i not in _prot_set for i in range(self.num_attribs)]
        )

        self._all_combs = np.array(
            list(itertools.product(*self.protected_domains)), dtype=int
        )
        self._M = len(self._all_combs)
        self._comb_to_idx = {
            tuple(c): i for i, c in enumerate(self._all_combs)
        }

        self._is_categorical = [
            name in self._categorical_cols for name in self.feature_names
        ]

        self.save_all_disc = save_all_disc

        # Tracking
        self.tot_inputs = set()          # non-sensitive hashes — dedup and counting
        self.tot_inputs_list = []        # full-feature arrays for saving (gated by save_all_disc)
        self._tot_full_seen = set()      # full-feature tuples — dedup for save_all_disc=True
        # Three nested IDI counts (raw >= partial >= true; and raw >= disc >= true):
        self.raw_disc_inputs = set()     # RAW: >=2 variants disagree, NO constraint check (classic IFT metric)
        self.raw_disc_list = []          # RAW full-feature rows — CAFT's reported output (audit denominator; gated by save_all_disc)
        self._raw_full_seen = set()      # full-feature tuples — dedup for save_all_disc=True
        self.partial_disc_inputs = set() # ONE-SIDED: raw IDI and original instance valid
        self.disc_inputs = set()         # internal search target: >=2 CONSTRAINT-VALID variants disagree (anchor unchecked)
        self.valid_disc_profiles = set() # TRUE: original valid AND a valid counterfactual disagrees (our two-sided validity)
        self.valid_disc_list = []        # full-feature arrays for valid disc inputs (saving/retraining)

        self.total_generated = 0
        self.inference_count = 0
        self.cumulative_efficiency = []
        self.time_to_1000_disc = -1
        self.time_to_1000_valid_disc = -1
        self.log_interval = 300
        self.initial_log_interval = self.log_interval

        # GA
        seeded_pop = (
            self._build_seeded_population(df_encoded, n_clusters, seed_fashion)
            if seed_strategy == 'dataset' else None
        )
        self.GA = GA(
            pop=seeded_pop,
            pop_size=population_size,
            dna_size=self.num_attribs,
            bound=self.input_bounds,
            discrimination_check=self.discrimination_object_func(),
            mutation_rate=0.05,
            sigma=sigma,
            crossover_strategy=crossover_strategy,
            mutation_strategy=mutation_strategy,
            categorical_threshold=ga_categorical_threshold,
            selection_strategy=selection_strategy,
            fos=self.build_constraint_fos() if crossover_strategy == 'gom' else None
        )

        # if constraint_mode == "penalty":
        #     mode_tag = f"penalty_l{lambda_penalty}"
        # elif constraint_mode == "repair":
        #     mode_tag = f"repair_p{max_passes}"
        # elif constraint_mode == "hybrid":
        #     mode_tag = f"hybrid_ri{repair_interval}_rb{repair_bursts}"
        # else:
        #     mode_tag = "none"
        # seed_tag = f"dataset_{seed_fashion}" if seed_strategy == 'dataset' else 'random'
        self.approach_name = f"CAFT"

        print(f"  Approach: {self.approach_name}")
    # Encoding helpers
    def get_dataset(self):
        import pandas as pd

        feature_names = list(self.config.feature_name)
        rows = []
        with open(f'datasets/{self.config.dataset_name}') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(',')
                rows.append([float(x) for x in parts[:len(feature_names)]])
        df_encoded = pd.DataFrame(rows, columns=feature_names).astype(int)
        print(f'  Data: {len(df_encoded):,} rows × {len(df_encoded.columns)} features')
        return df_encoded

    def _row_to_dict(self, row):
        """Convert a GA integer array into a dict for the constraint validator."""
        return {
            name: (str(int(row[i])) if cat else int(row[i]))
            for i, (name, cat) in enumerate(zip(self.feature_names, self._is_categorical))
        }

    # Repair operator

    def _repair_batch(self, x_array):
        """Vectorised fixed-point repair across the candidate batch (Algorithm 1)."""
        N = len(x_array)
        for _ in range(self.max_passes):
            any_changed = False
            for rule in self._repair_rules:
                ant_vals = x_array[:, rule["_ant_col"]]

                if rule["_ant_type"] == 1:  # range
                    ant_met = (ant_vals > rule["_lo"]) & (ant_vals <= rule["_hi"])
                else:  # equality (value or sentinel)
                    ant_met = (ant_vals == rule["_ant_v"])

                if not ant_met.any():
                    continue

                tgt_vals = x_array[:, rule["_col_idx"]]
                cons_violated = np.isin(tgt_vals, rule["_forbidden"])
                to_fix = ant_met & cons_violated

                if to_fix.any():
                    replacement = rule["_replacement_int"]
                    if replacement is not None:
                        x_array[to_fix, rule["_col_idx"]] = replacement
                        any_changed = True

            if not any_changed:
                break

    # Dataset seeding

    def _build_seeded_population(self, df_encoded, n_clusters=10, fashion='RoundRobin'):
        """Build an initial population by K-means resampling of the encoded training data."""
        from sklearn.cluster import KMeans
        X = df_encoded.values.astype(float)
        n_c = min(n_clusters, len(X))
        km = KMeans(n_clusters=n_c, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        if fashion == 'Stratified':
            all_pv = list(itertools.product(*self.protected_domains))
            strata = {}
            for k in range(n_c):
                cluster_rows = X[labels == k]
                for pv in all_pv:
                    mask = np.ones(len(cluster_rows), dtype=bool)
                    for j, attr_idx in enumerate(self.protected_attribs):
                        mask &= cluster_rows[:, attr_idx].astype(int) == pv[j]
                    rows = cluster_rows[mask]
                    if len(rows):
                        strata[(k, pv)] = rows

            slots = list(strata.keys())
            print(f"  Seeding: {n_c} clusters × {len(all_pv)} pv combos "
                  f"→ {len(slots)} active (cluster, pv) slots [Stratified]")
            if not slots:
                print("  Warning: no stratified slots found, falling back to RoundRobin")
                clusters = [X[labels == k] for k in range(n_c)]
                pop = [clusters[i % n_c][np.random.randint(len(clusters[i % n_c]))].copy()
                       for i in range(self.population_size)]
            else:
                pop = []
                for i in range(self.population_size):
                    k, pv = slots[i % len(slots)]
                    c = strata[(k, pv)]
                    pop.append(c[np.random.randint(len(c))].copy())
        else:
            clusters = [X[labels == k] for k in range(n_c)]
            print(f"  Seeding: {n_c} clusters, fashion={fashion}, "
                  f"sizes={[len(c) for c in clusters]}")
            pop = []
            if fashion == 'RoundRobin':
                for i in range(self.population_size):
                    c = clusters[i % n_c]
                    pop.append(c[np.random.randint(len(c))].copy())
            else:  # Distribution
                X_len = len(X)
                proba = np.array([len(c) / X_len for c in clusters])
                cum = np.cumsum(proba)
                for _ in range(self.population_size):
                    ci = int(np.searchsorted(cum, np.random.rand()))
                    ci = min(ci, n_c - 1)
                    c = clusters[ci]
                    pop.append(c[np.random.randint(len(c))].copy())

        arr = np.clip(np.array(pop), self.input_bounds[:, 0], self.input_bounds[:, 1])
        return arr.astype(float)

    # Protected-combination generation

    def similar_set_(self, X):
        """Generate all protected-attribute variants for a batch X."""
        N = len(X)
        batch_inputs = np.repeat(X, self._M, axis=0)
        batch_inputs[:, self.protected_attribs] = np.tile(self._all_combs, (N, 1))
        return batch_inputs, self._all_combs, self._M

    # Hybrid scheduling

    def _effective_mode(self):
        """Active enforcement mode for the current generation ('hybrid': fixed schedule, 'adaptive': feedback on validity rate)."""
        if self.constraint_mode == 'hybrid':
            cycle = self.repair_interval + self.repair_bursts
            return 'repair' if (self._gen % cycle) < self.repair_bursts else 'penalty'
        if self.constraint_mode == 'adaptive':
            return 'repair' if self._last_valid_frac < self.adaptive_target else 'penalty'
        return self.constraint_mode

    # Vectorised rule evaluation

    def _build_rule_arrays(self, rules):
        """Pack dependency-exclusion rules into parallel numpy arrays for vectorised evaluation."""
        K = len(rules)
        ant_col = np.full(K, -1, dtype=np.int32)
        cons_col = np.full(K, -1, dtype=np.int32)
        ant_type = np.zeros(K, dtype=np.int8)   # 0=value/sentinel, 1=range
        ant_v = np.zeros(K, dtype=np.float64)
        ant_min = np.zeros(K, dtype=np.float64)
        ant_max = np.zeros(K, dtype=np.float64)
        forbidden = [np.empty(0, dtype=np.int64)] * K
        active = np.zeros(K, dtype=bool)

        for k, rule in enumerate(rules):
            ant = rule["antecedent"]
            cons = rule["consequent_exclusion"]
            ac = self._feature_idx.get(ant["attribute"], -1)
            cc = self._feature_idx.get(cons["attribute"], -1)
            if ac < 0 or cc < 0:
                continue
            if "range" in ant:
                ant_type[k] = 1
                ant_min[k] = float(ant["range"]["min"])
                ant_max[k] = float(ant["range"]["max"])
            elif "sentinel_value" in ant:
                ant_type[k] = 0
                try:
                    ant_v[k] = float(ant["sentinel_value"])
                except (ValueError, TypeError):
                    continue
            else:
                ant_type[k] = 0
                try:
                    ant_v[k] = float(ant["value"])
                except (ValueError, TypeError):
                    continue

            forbidden_ints = []
            for v in cons["values"]:
                try:
                    forbidden_ints.append(int(v))
                except (ValueError, TypeError):
                    pass
            if not forbidden_ints:
                continue

            ant_col[k] = ac
            cons_col[k] = cc
            forbidden[k] = np.array(forbidden_ints, dtype=np.int64)
            active[k] = True

        return {
            "K": K,
            "active": active,
            "ant_col": ant_col,
            "cons_col": cons_col,
            "ant_type": ant_type,
            "ant_v": ant_v,
            "ant_min": ant_min,
            "ant_max": ant_max,
            "forbidden": forbidden,
        }

    def _build_domain_arrays(self, domain_dict):
        """Pack attribute_domain entries into per-attribute numpy arrays."""
        entries = []
        for attr, spec in domain_dict.items():
            col = self._feature_idx.get(attr, -1)
            if col < 0:
                continue
            if spec["type"] == "numerical":
                entries.append((col, 'num',
                                float(spec["min"]), float(spec["max"]),
                                None))
            elif spec["type"] == "categorical":
                allowed = []
                for v in spec["values"]:
                    try:
                        allowed.append(int(v))
                    except (ValueError, TypeError):
                        pass
                entries.append((col, 'cat', 0.0, 0.0,
                                np.array(allowed, dtype=np.int64)))
        return entries

    def _violations_vec(self, batch, rule_arrs, domain_arrs):
        """Vectorised violation count: attribute-domain plus dependency-exclusion rules."""
        B = len(batch)
        total = np.zeros(B, dtype=np.int32)

        for col, kind, lo, hi, allowed in domain_arrs:
            vals = batch[:, col]
            if kind == 'num':
                total += ((vals < lo) | (vals > hi)).astype(np.int32)
            else:
                if len(allowed) == 0:
                    total += np.ones(B, dtype=np.int32)
                else:
                    total += (~np.isin(vals, allowed)).astype(np.int32)

        K = rule_arrs["K"]
        if K == 0:
            return total
        active = rule_arrs["active"]
        for k in range(K):
            if not active[k]:
                continue
            ant_vals = batch[:, rule_arrs["ant_col"][k]]
            if rule_arrs["ant_type"][k] == 1:  # range: (min, max]
                ant_met = (ant_vals > rule_arrs["ant_min"][k]) & \
                          (ant_vals <= rule_arrs["ant_max"][k])
            else:  # value or sentinel: numeric equality
                ant_met = (ant_vals == rule_arrs["ant_v"][k])
            if not ant_met.any():
                continue
            cons_vals = batch[:, rule_arrs["cons_col"][k]]
            cons_violated = np.isin(cons_vals, rule_arrs["forbidden"][k])
            total += (ant_met & cons_violated).astype(np.int32)
        return total

    # Discrimination scorer (constraint-aware)

    def _score_candidates(self, x_array):
        """Discrimination fitness scored over the constraint-valid subset of counterfactual variants (Section 3.3)."""
        N = len(x_array)
        mode = self._effective_mode()

        if mode in ('repair', 'repair_penalty'):
            self._repair_batch(x_array)

        batch_inputs, _, M = self.similar_set_(x_array)
        all_outputs = self.model.predict(batch_inputs).reshape(-1)
        self.inference_count += 1
        y_matrix = all_outputs.reshape(N, M)

        shared_viol_n = self._violations_vec(
            x_array,
            self._shared_rule_arrs, self._shared_domain_arrs,
        )
        variant_viol_nm = self._violations_vec(
            batch_inputs,
            self._variant_rule_arrs, self._variant_domain_arrs,
        )
        ind_idx_nm = np.arange(N * M) // M
        variant_viol = shared_viol_n[ind_idx_nm] + variant_viol_nm
        variant_valid = (variant_viol == 0)
        valid_matrix = variant_valid.reshape(N, M)

        # Unique prediction count over CONSTRAINT-VALID variants only.
        # An IDI is a pair, so fewer than 2 valid variants ⇒ no IDI possible.
        unique_counts = np.ones(N, dtype=float)
        for i in range(N):
            valid_idx = np.where(valid_matrix[i])[0]
            if len(valid_idx) >= 2:
                valid_preds = y_matrix[i, valid_idx]
                sp = np.sort(valid_preds)
                unique_counts[i] = float((np.diff(sp) != 0).sum() + 1)

        disc_idx = np.where(unique_counts > 1)[0]

        # Index of each candidate's own protected-attribute combination; needed
        # for the partial/true validity checks and the penalty term.
        orig_comb_idx = np.array([
            self._comb_to_idx.get(
                tuple(int(x) for x in x_array[i, self.protected_attribs]), 0
            )
            for i in range(N)
        ])
        anchor_valid = variant_valid[np.arange(N) * M + orig_comb_idx]

        # Feedback signal for 'adaptive' mode: the fraction of this generation's
        # candidates whose GA individual is constraint-valid. In repair
        # generations this reads high (candidates were repaired), which flips the
        # next generation back to penalty; in penalty generations it reads the
        # true drift, which triggers repair once it falls below the target.
        if self.constraint_mode == 'adaptive' and N > 0:
            self._last_valid_frac = float(anchor_valid.mean())

        # RAW IDI (classic, unchecked): >=2 of the M variants disagree, with NO
        # constraint filtering.  This is CAFT's reported output, saved so the
        # audit judges every tool on the same raw denominator (Option A).
        # ONE-SIDED diagnostic: a raw IDI whose original instance is valid,
        # with the counterfactual side unchecked. This is not the two-sided
        # validity definition used for valid IDIs.
        raw_disc_idx = np.where((y_matrix != y_matrix[:, [0]]).any(axis=1))[0]
        for i in raw_disc_idx:
            row = x_array[i, :self.num_attribs]
            ns_key = hash(row[self._non_sens_mask].tobytes())
            is_new_raw = ns_key not in self.raw_disc_inputs
            self.raw_disc_inputs.add(ns_key)
            if anchor_valid[i]:
                self.partial_disc_inputs.add(ns_key)
            if self.save_all_disc:
                full_key = tuple(int(v) for v in row)
                if full_key not in self._raw_full_seen:
                    self._raw_full_seen.add(full_key)
                    self.raw_disc_list.append(row.copy())
            elif is_new_raw:
                self.raw_disc_list.append(row.copy())

        if mode in ('penalty', 'repair_penalty'):
            ga_idx = np.arange(N) * M + orig_comb_idx
            viol_counts = variant_viol[ga_idx]
            # Hybrid penalty: binary threshold + small per-violation gradient.
            # For 'repair_penalty', anchor_valid/viol_counts reflect the
            # POST-repair state (repair ran above), so this penalises only
            # residual violations the repair pass failed to resolve.
            unique_counts[~anchor_valid] -= self.lambda_penalty
            unique_counts -= self.lambda_gradient * viol_counts

        if len(disc_idx):
            for i in disc_idx:
                row = x_array[i, :self.num_attribs]
                ns_key = hash(row[self._non_sens_mask].tobytes())
                self.disc_inputs.add(ns_key)   # internal manifold tracking

                # TRUE validity: this profile is already a disc (>=2 valid
                # variants disagree); requiring the anchor also valid means a
                # valid counterfactual disagrees with a valid original.
                if anchor_valid[i]:
                    is_new_valid = ns_key not in self.valid_disc_profiles
                    self.valid_disc_profiles.add(ns_key)
                    if is_new_valid:
                        self.valid_disc_list.append(row.copy())
            self.set_time_to_1000_disc()

        return unique_counts

    # Fitness function

    def discrimination_object_func(self):
        """Fitness wrapper called by the GA every generation."""
        def func(indvs):
            # Clip to input bounds so GA mutation cannot produce protected-attribute
            # values outside the precomputed _all_combs domain.
            x_array = np.clip(
                np.asarray(indvs, dtype=float),
                self.input_bounds[:, 0],
                self.input_bounds[:, 1],
            )
            N = len(x_array)
            if N == 0:
                return np.array([])
            self.total_generated += N
            _sliced = x_array[:, :self.num_attribs][:, self._non_sens_mask]
            keys = [row.tobytes() for row in _sliced]

            # Non-sensitive duplicate penalty: assigns -lambda_redundancy to any
            # individual whose non-protected profile was already seen.  Duplicates
            # are penalised without evaluation; only novel profiles are scored.
            # This drives discriminatory-manifold tracking (see docstring).
            keys = [hash(k) for k in keys]
            is_dup = np.array([k in self.tot_inputs for k in keys], dtype=bool)
            self.tot_inputs.update(keys)

            full_rows = x_array[:, :self.num_attribs].astype(int)
            if self.save_all_disc:
                for i in range(N):
                    full_key = tuple(full_rows[i])
                    if full_key not in self._tot_full_seen:
                        self._tot_full_seen.add(full_key)
                        self.tot_inputs_list.append(full_rows[i].copy())
            else:
                for i in np.where(~is_dup)[0]:
                    self.tot_inputs_list.append(full_rows[i].copy())

            scores = np.full(N, -self.lambda_redundancy, dtype=float)
            new_idx = np.where(~is_dup)[0]
            if len(new_idx) > 0:
                scores[new_idx] = self._score_candidates(x_array[new_idx].astype(np.int64))

            self._gen += 1
            return scores
        return func

    def build_constraint_fos(self):
        """Build Family Of Subsets (FOS) from extracted inter-attribute dependency constraints."""

        graph = {}

        for rule in self.constraints["inter_attribute"]:

            if rule["type"] != "dependency_exclusion":
                continue

            a_name = rule["antecedent"]["attribute"]
            b_name = rule["consequent_exclusion"]["attribute"]

            a = self._feature_idx[a_name]
            b = self._feature_idx[b_name]

            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        fos = []

        # singleton genes
        for i in range(self.num_attribs):
            fos.append([i])

        # pairwise dependencies
        for a in graph:
            for b in graph[a]:
                fos.append(sorted([a, b]))

        # connected components
        visited = set()

        for start in graph:

            if start in visited:
                continue

            stack = [start]
            component = []

            while stack:

                node = stack.pop()

                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                stack.extend(graph[node])

            if len(component) > 2:
                fos.append(sorted(component))

        # deduplicate
        fos = [
            list(x)
            for x in sorted(
                {tuple(sorted(x)) for x in fos},
                key=len
            )
        ]

        return fos

    # Tracking

    def update_cumulative_efficiency(self, iteration):
        self.cumulative_efficiency.append([
            time.time() - self.start_time,
            iteration,
            len(self.tot_inputs),
            len(self.disc_inputs),
            len(self.valid_disc_profiles),
        ])

    def set_time_to_1000_disc(self):
        if len(self.disc_inputs) >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\n  Time to 1000 IDIs: {self.time_to_1000_disc:.2f}s")
        if len(self.valid_disc_profiles) >= 1000 and self.time_to_1000_valid_disc == -1:
            self.time_to_1000_valid_disc = time.time() - self.start_time
            print(f"\n  Time to 1000 valid IDIs: {self.time_to_1000_valid_disc:.2f}s")

    # Main loop

    def run(self, max_samples=1_000_000, max_allowed_time=3600):
        max_evolution = math.ceil(max_samples / self.GA.N)
        print(f"\n[CAFT] mode={self.constraint_mode}, pop={self.GA.N}, "
              f"max_time={max_allowed_time}s")

        for i in range(max_evolution):
            self.GA.evolve()
            self.update_cumulative_efficiency(iteration=i)

            elapsed = time.time() - self.start_time
            if elapsed >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=elapsed, is_log=True)

            if elapsed >= max_allowed_time or self.total_generated >= max_samples:
                break

        self.report(elapsed_time=time.time() - self.start_time, is_log=False)

    def report(self, elapsed_time, is_log):
        save_path = 'results/CAFT/'
        n_raw = len(self.raw_disc_inputs)          # raw IDI (classic)
        n_partial = len(self.partial_disc_inputs)  # one-sided diagnostic
        n_valid = len(self.valid_disc_profiles)    # true two-sided validity (ours)
        # VDR = true-valid IDIs / raw IDIs: fraction of the classic IDI count
        # that survives two-sided constraint validation.
        valid_disc_rate = round(100 * n_valid / max(1, n_raw), 1)
        valid_egs = n_valid / elapsed_time if elapsed_time > 0 else 0

        extra_save_data = {
            'valid_disc_inputs': self.valid_disc_list,
        }

        sens_names = [self.config.sens_name[s] for s in self.sensitive_params]
        generate_report(
            approach_name=self.approach_name,
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=','.join(map(str, sorted(sens_names))),
            tot_inputs=self.tot_inputs_list,
            disc_inputs=self.raw_disc_list,   # raw IDIs = CAFT's reported output (audit denominator)
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
            save_path=save_path,
            save_data=True,
            inference_count=self.inference_count,
            constraint_mode=self.constraint_mode,
            valid_disc_inputs=n_valid,
            valid_disc_rate=valid_disc_rate,
            valid_egs=valid_egs,
            raw_disc_inputs=n_raw,
            partial_disc_inputs=n_partial,
            extra_save_data=extra_save_data,
            time_to_1000_valid_disc=self.time_to_1000_valid_disc,
        )


if __name__ == '__main__':
    import argparse
    import tensorflow as tf

    # CAFT-specific args parsed first; parse_known_args avoids conflicts with
    # the shared args that get_experiment_params() will consume.
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument('--constraint_mode', default='repair',
                    choices=['none', 'penalty', 'repair', 'hybrid', 'adaptive', 'repair_penalty'])
    _p.add_argument('--lambda_penalty', type=float, default=0.5)
    _p.add_argument('--lambda_gradient', type=float, default=0.05)
    _p.add_argument('--min_score', type=float, default=0.97)
    _p.add_argument('--max_passes', type=int, default=5)
    _p.add_argument('--seed_strategy', default='random',
                    choices=['random', 'dataset'])
    _p.add_argument('--seed_fashion', default='RoundRobin',
                    choices=['RoundRobin', 'Distribution', 'Stratified'])
    _p.add_argument('--n_clusters', type=int, default=10)
    _p.add_argument('--lambda_redundancy', type=float, default=0.0)
    _p.add_argument('--save_all_disc', action='store_true', default=False)
    _p.add_argument('--selection_strategy', default='tournament',
                    choices=['tournament', 'roulette', 'rank_roulette'])
    CAFT_args, _ = _p.parse_known_args()

    config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = \
        get_experiment_params()

    print(f'Dataset:        {config.dataset_name}')
    print(f'Classifier:     {classifier_name}')
    print(f'Sensitive:      {",".join(map(str, sorted(sensitive_names)))}')
    print(f'Mode:           {CAFT_args.constraint_mode}  '
          f'(max_passes={CAFT_args.max_passes})')
    print(f'Seed strategy:  {CAFT_args.seed_strategy}')
    print()

    model_path = os.path.join(path, 'models')

    if classifier_name == 'dnn':
        classifier_path = os.path.join(model_path, config.dataset_name, 'dnn.keras')
        model = tf.keras.models.load_model(classifier_path)
    elif classifier_name == 'ftt':
        from src.utils.tabular_transformers import load_ftt_model
        model = load_ftt_model(dataset_name=config.dataset_name)
    else:
        classifier_path = os.path.join(model_path, config.dataset_name,
                                       f'{classifier_name.lower()}.pkl')
        model = joblib.load(classifier_path)

    # Load encoded dataset for constraint extraction
    data_file = os.path.join(path, 'datasets', config.dataset_name)
    feature_names = list(config.feature_name)
    rows = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(',')
            rows.append([float(x) for x in parts[:len(feature_names)]])
    df_encoded = pd.DataFrame(rows, columns=feature_names).astype(int)
    print(f'  Data: {len(df_encoded):,} rows × {len(df_encoded.columns)} features')

    runner = CAFT(
        config=config,
        model=model,
        classifier_name=classifier_name,
        sensitive_params=sensitive_params,
        constraint_mode=CAFT_args.constraint_mode,
        lambda_penalty=CAFT_args.lambda_penalty,
        lambda_gradient=CAFT_args.lambda_gradient,
        min_score=CAFT_args.min_score,
        max_passes=CAFT_args.max_passes,
        seed_strategy=CAFT_args.seed_strategy,
        n_clusters=CAFT_args.n_clusters,
        seed_fashion=CAFT_args.seed_fashion,
        lambda_redundancy=CAFT_args.lambda_redundancy,
        save_all_disc=CAFT_args.save_all_disc,
        selection_strategy=CAFT_args.selection_strategy,
    )

    runner.run(max_allowed_time=max_allowed_time)
