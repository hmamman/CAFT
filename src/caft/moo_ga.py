"""Multi-objective GA for CAFT's Pareto-based enforcement mode (Section 3.3.4)."""
import numpy as np


class MOOGA:
    """Multi-objective GA with rank-based roulette selection over non-dominated fronts."""

    def __init__(self, pop, pop_size, dna_size, bound, fitness_func,
                 mutation_rate=0.05, sigma=0.05,
                 crossover_strategy='hux', mutation_strategy='hybrid',
                 categorical_threshold=5,
                 selection='rank_roulette',
                 elite_frac=0.10,
                 kappa=0.05):

        if pop_size % 2 != 0:
            pop_size += 1

        self.N  = pop_size
        self.n  = dna_size
        self.bound = np.array(bound)
        self.low   = self.bound[:, 0]
        self.high  = self.bound[:, 1]

        self.fitness_func         = fitness_func
        self.mr                   = mutation_rate
        self.sigma                = sigma
        self.crossover_strategy   = crossover_strategy
        self.mutation_strategy    = mutation_strategy
        self.categorical_threshold = categorical_threshold
        self.selection_strategy   = selection
        self.kappa                = kappa

        self.elite_size     = 0 if elite_frac == 0 else max(1, int(self.N * elite_frac))
        self.gaussian_delta = (self.high - self.low) * sigma

        self.visit_scale = np.ones(self.N, dtype=float)

        self.non_elite_mask = np.ones((self.N, 1), dtype=bool)
        self.non_elite_mask[:self.elite_size] = False

        self.population = (
            pop.copy() if pop is not None
            else np.random.uniform(self.low, self.high, (self.N, self.n))
        )

    # Main loop

    def evolve(self):
        fitness_matrix = self.fitness_func(self.population)   # (N, 2)

        self._rank_roulette_select(fitness_matrix)
        self._hux_crossover()
        self._hybrid_mutate()

    # Rank-based roulette selection

    def _non_dominated_sort(self, F):
        """Vectorised Pareto-front decomposition (maximisation)."""
        N = len(F)
        gte = np.all(F[:, None, :] >= F[None, :, :], axis=2)
        gt  = np.any(F[:, None, :] >  F[None, :, :], axis=2)
        dom = gte & gt
        np.fill_diagonal(dom, False)

        dom_count = dom.sum(axis=0).astype(int)

        fronts   = []
        assigned = np.zeros(N, dtype=bool)

        while not assigned.all():
            front = np.where((dom_count == 0) & ~assigned)[0]
            if len(front) == 0:             # degenerate — assign all remaining
                front = np.where(~assigned)[0]
            fronts.append(front.tolist())
            assigned[front] = True
            dom_count -= dom[front].sum(axis=0)
            dom_count  = np.maximum(dom_count, 0)

        return fronts

    def _rank_roulette_select(self, F):
        """Exponentially decaying selection probability by Pareto-front rank: score(i) = exp(-rank(i) * decay)."""
        DECAY = 1.0
        N = len(F)

        fronts = self._non_dominated_sort(F)
        rank   = np.empty(N, dtype=int)
        for r, front in enumerate(fronts):
            rank[np.array(front)] = r

        score = np.exp(-rank.astype(float) * DECAY)

        order    = np.argsort(rank)
        selected = list(order[:self.elite_size])

        n_remaining = self.N - self.elite_size
        total = score.sum()
        probs = score / total if total > 0 else np.ones(N) / N
        selected.extend(
            np.random.choice(N, size=n_remaining, replace=True, p=probs).tolist()
        )

        self.population = self.population[selected]

    # Crossover

    def _hux_crossover(self):
        """Probabilistic HUX: swap differing genes with 50% probability."""
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        p1 = self.population[idx[::2]].copy()
        p2 = self.population[idx[1::2]].copy()

        diff = p1 != p2
        swap = diff & (np.random.rand(self.N // 2, self.n) < 0.5)

        c1 = np.where(swap, p2, p1)
        c2 = np.where(swap, p1, p2)

        self.population = np.vstack((c1, c2))

    # Mutation

    def _hybrid_mutate(self):
        """Creep mutation for small-range features, Gaussian for wide-range; step scaled by visit_scale."""
        mut_mask   = np.random.rand(self.N, self.n) < self.mr
        final_mask = mut_mask & self.non_elite_mask

        ranges         = self.high - self.low
        is_categorical = (ranges <= self.categorical_threshold)

        creep    = np.random.choice([-1, 1], size=(self.N, self.n))
        gaussian = np.random.uniform(-ranges * 0.1, ranges * 0.1, size=(self.N, self.n))

        scale = self.visit_scale[:, None]
        delta = np.where(is_categorical, creep * scale, gaussian * scale)
        self.population += np.where(final_mask, delta, 0.0)
        self.population  = np.clip(self.population, self.low, self.high)
