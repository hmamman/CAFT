"""CAFT with multi-objective constraint handling via MOOGA (Ablation A6, Section 3.3.5).

Replaces the scalarised penalty with three Pareto objectives: discrimination
(disagreeing constraint-valid variants), validity (fraction of variants
satisfying C_tau), and diversity (distance to recent discriminatory profiles).
Selection is rank-based roulette over non-dominated fronts (moo_ga.MOOGA).
"""
import numpy as np

from src.caft.caft import CAFT
from src.caft.moo_ga import MOOGA


class CAFT_MOO(CAFT):

    def __init__(self, config, model, classifier_name, sensitive_params,
                 selection='rank_roulette',
                 kappa=0.05,
                 diversity_window=100,
                 elite_frac=0.0,
                 **kwargs):
        kwargs.setdefault('lambda_redundancy', 2.0)
        # Validity pressure comes from objective 2; the scalarised enforcement
        # modes do not apply. 'none' keeps the inherited scorer inert.
        kwargs.setdefault('constraint_mode', 'none')
        super().__init__(config, model, classifier_name, sensitive_params,
                         **kwargs)

        self.selection_name   = selection
        self.diversity_window = diversity_window
        self._disc_window     = []   # sliding window of recent disc rows

        # Replace the single-objective GA built by CAFT.__init__ with MOOGA,
        # reusing its (possibly seeded) initial population.
        seeded_pop = self.GA.population.copy()
        self.GA = MOOGA(
            pop=seeded_pop,
            pop_size=self.population_size,
            dna_size=self.num_attribs,
            bound=self.input_bounds,
            fitness_func=self._moo_object_func(),
            mutation_rate=0.05,
            sigma=kwargs.get('sigma', 0.05),
            crossover_strategy=kwargs.get('crossover_strategy', 'hux'),
            mutation_strategy=kwargs.get('mutation_strategy', 'hybrid'),
            categorical_threshold=kwargs.get('ga_categorical_threshold', 5),
            selection=selection,
            kappa=kappa,
            elite_frac=elite_frac,
        )

        self.approach_name = "CAFT_MOO"
        print(f"  MOO selection: {selection}, diversity_window={diversity_window}")

    # ── Diversity helper (objective 3) ────────────────────────────────────────

    def _compute_diversity(self, x_array):
        """
        Min mixed distance to the discriminatory-profile sliding window, over
        non-protected features only.

        Per-feature distance: Hamming (0/1) for features whose range is at
        most the GA categorical threshold; normalised absolute difference
        |a - b| / range otherwise. Feature distances are averaged to a scalar
        in [0, 1]. Returns 1.0 for all candidates when the window is empty.
        """
        N = len(x_array)
        if not self._disc_window:
            return np.ones(N, dtype=float)

        ns_idx = np.where(self._non_sens_mask)[0]
        lo     = self.input_bounds[ns_idx, 0].astype(float)
        hi     = self.input_bounds[ns_idx, 1].astype(float)
        ranges = hi - lo
        ranges[ranges == 0] = 1.0
        is_cat = ranges <= self.GA.categorical_threshold

        cands = x_array[:, ns_idx].astype(float)                       # (N, n_ns)
        wind  = np.array(self._disc_window)[:, ns_idx].astype(float)   # (K, n_ns)

        abs_diff  = np.abs(cands[:, None, :] - wind[None, :, :])       # (N, K, n_ns)
        norm_diff = np.where(is_cat, (abs_diff > 0).astype(float),
                             abs_diff / ranges)
        return norm_diff.mean(axis=2).min(axis=1)

    # ── MOO fitness wrapper ───────────────────────────────────────────────────

    def _moo_object_func(self):
        """
        Returns a closure (N, dna_size) -> (N, 3). Mirrors
        CAFT.discrimination_object_func: identical clipping, full-feature
        dedup, and tracking; only the score shape differs.
        """
        def func(indvs):
            x_array = np.clip(
                np.asarray(indvs, dtype=float),
                self.input_bounds[:, 0],
                self.input_bounds[:, 1],
            )
            N = len(x_array)
            if N == 0:
                return np.empty((0, 3))
            self.total_generated += N

            # Non-sensitive hash keys, identical to the parent class: two rows
            # differing only in protected attributes are the same profile.
            _sliced = x_array[:, :self.num_attribs][:, self._non_sens_mask]
            keys = [hash(row.tobytes()) for row in _sliced]
            is_dup = np.array([k in self.tot_inputs for k in keys], dtype=bool)
            self.tot_inputs.update(keys)

            full_rows = x_array[:, :self.num_attribs].astype(int)
            if self.save_all_disc:
                for i in range(N):
                    fk = tuple(full_rows[i])
                    if fk not in self._tot_full_seen:
                        self._tot_full_seen.add(fk)
                        self.tot_inputs_list.append(full_rows[i].copy())
            else:
                for i in np.where(~is_dup)[0]:
                    self.tot_inputs_list.append(full_rows[i].copy())

            fitness = np.zeros((N, 3), dtype=float)
            fitness[is_dup, 0] = -self.lambda_redundancy
            # O2 = O3 = 0 for duplicates — dominated on O1 alone.

            new_idx = np.where(~is_dup)[0]
            if len(new_idx) > 0:
                fitness[new_idx] = self._score_candidates_moo(
                    x_array[new_idx].astype(np.int64))

            self._gen += 1
            return fitness

        return func

    # ── Tri-objective scoring ─────────────────────────────────────────────────

    def _score_candidates_moo(self, x_array):
        """
        Score novel candidates on all three objectives. Variant generation,
        partitioned validation, and IDI recording are identical to
        CAFT._score_candidates; only the returned score differs.
        """
        N = len(x_array)

        batch_inputs, _, M = self.similar_set_(x_array)
        all_outputs = self.model.predict(batch_inputs).reshape(-1)
        self.inference_count += 1
        y_matrix = all_outputs.reshape(N, M)

        shared_viol_n = self._violations_vec(
            x_array, self._shared_rule_arrs, self._shared_domain_arrs)
        variant_viol_nm = self._violations_vec(
            batch_inputs, self._variant_rule_arrs, self._variant_domain_arrs)
        ind_idx_nm    = np.arange(N * M) // M
        variant_viol  = shared_viol_n[ind_idx_nm] + variant_viol_nm
        valid_matrix  = (variant_viol == 0).reshape(N, M)

        # Index of each candidate's own protected combination.
        orig_comb_idx = np.array([
            self._comb_to_idx.get(
                tuple(int(v) for v in x_array[i, self.protected_attribs]), 0)
            for i in range(N)
        ])

        # O1: count of valid variants disagreeing with the candidate's own
        # prediction. Richer gradient than the unique-prediction count on
        # multi-valued protected attributes.
        orig_preds   = y_matrix[np.arange(N), orig_comb_idx]
        disagree_nm  = (y_matrix != orig_preds[:, None])
        o1 = (valid_matrix & disagree_nm).sum(axis=1).astype(float)

        # O2: fraction of variants that satisfy C_tau.
        o2 = valid_matrix.sum(axis=1).astype(float) / M

        # Two-sided IDI guard, identical to CAFT._score_candidates: at least
        # two valid variants whose predictions differ among themselves.
        is_disc = np.zeros(N, dtype=bool)
        for i in range(N):
            valid_idx = np.where(valid_matrix[i])[0]
            if len(valid_idx) >= 2:
                vp = y_matrix[i, valid_idx]
                is_disc[i] = vp.max() != vp.min()

        # O3: diversity, only for discriminatory candidates so that
        # non-discriminatory ones never outrank them on the Pareto front.
        o3 = np.zeros(N, dtype=float)
        if is_disc.any():
            o3[is_disc] = self._compute_diversity(x_array[is_disc])

        anchor_valid = valid_matrix[np.arange(N), orig_comb_idx]

        # RAW IDI (classic, unchecked) and a one-sided diagnostic (raw IDI with
        # a valid original instance, counterfactual side unchecked). Raw rows are
        # saved as CAFT's reported output for the shared audit denominator.
        raw_disc_idx = np.where((y_matrix != y_matrix[:, [0]]).any(axis=1))[0]
        for i in raw_disc_idx:
            row = x_array[i, :self.num_attribs]
            ns_key = hash(row[self._non_sens_mask].tobytes())
            is_new_raw = ns_key not in self.raw_disc_inputs
            self.raw_disc_inputs.add(ns_key)
            if anchor_valid[i]:
                self.partial_disc_inputs.add(ns_key)
            if self.save_all_disc:
                fk = tuple(int(v) for v in row)
                if fk not in self._raw_full_seen:
                    self._raw_full_seen.add(fk)
                    self.raw_disc_list.append(row.copy())
            elif is_new_raw:
                self.raw_disc_list.append(row.copy())

        # Recording — same keys and same valid_disc rule as the parent class.
        disc_idx = np.where(is_disc)[0]
        if len(disc_idx):
            for i in disc_idx:
                row = x_array[i, :self.num_attribs]
                ns_key = hash(row[self._non_sens_mask].tobytes())
                self.disc_inputs.add(ns_key)   # internal manifold tracking

                if valid_matrix[i, orig_comb_idx[i]]:
                    is_new_valid = ns_key not in self.valid_disc_profiles
                    self.valid_disc_profiles.add(ns_key)
                    if is_new_valid:
                        self.valid_disc_list.append(row.copy())

                self._disc_window.append(row.copy())
                if len(self._disc_window) > self.diversity_window:
                    self._disc_window.pop(0)
            self.set_time_to_1000_disc()

        return np.column_stack([o1, o2, o3])
