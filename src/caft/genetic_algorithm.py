import numpy as np

class GA:
    def __str__(self):
        return "GA"

    def __init__(self, pop, pop_size, dna_size, bound, discrimination_check,
                 mutation_rate=0.05, sigma=0.1, crossover_strategy='hux',
                 mutation_strategy='uniform', 
                 selection_strategy='tournament',
                  categorical_threshold=10, fos=None):

        # Ensure population size is even for crossover pairing
        if pop_size % 2 != 0:
            pop_size += 1
            print(f"Warning: pop_size must be even. Setting to {pop_size}.")

        self.gen = 0
        self.N = pop_size
        self.n = dna_size
        self.bound = np.array(bound)
        self.fitness_func = discrimination_check  # Assumes this is vectorized
        self.mr = mutation_rate
        self.sigma = sigma
        self.fos = fos

        # Pre-calculate bounds for clipping
        self.low_bound = self.bound[:, 0]
        self.high_bound = self.bound[:, 1]

        # Pre-calculate Gaussian mutation delta (shape: (n,))
        self.gaussian_delta = (self.high_bound - self.low_bound) * self.sigma

        self.population = pop if pop is not None else self.generate_initial_population()
        self.best_fitness_history = []
        self.mutation_rate_history = [[self.gen, self.mr]]
        self.elite_size = int(self.N * 0.1)  # 10% elitism
        self.tournament_size = max(5, int(self.N * 0.05))
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.selection_strategy = selection_strategy
        self.categorical_threshold = categorical_threshold

        # Pre-create non-elite mask for mutations (shape: (N, 1))
        self.non_elite_mask = np.ones((self.N, 1), dtype=bool)
        self.non_elite_mask[:self.elite_size] = False

    def generate_initial_population(self):
        return np.random.uniform(
            low=self.low_bound,
            high=self.high_bound,
            size=(self.N, self.n)
        )

    def evaluate_fitness(self):
        return self.fitness_func(self.population)

    def tournament_selection(self, fitness):
        """
        Vectorized selection combining elitism and tournament selection.
        """
        # 1. Elitism: Get elite indices (already efficient)
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite = self.population[elite_indices]

        # 2. Vectorized tournament for the remaining population
        n_remaining = self.N - self.elite_size

        # Select all tournament participants at once
        participant_indices = np.random.randint(0, self.N,
                                                size=(n_remaining, self.tournament_size))

        # Get fitness for all participants
        participant_fitness = fitness[participant_indices]

        # Find winners for all tournaments at once
        winner_local_indices = np.argmax(participant_fitness, axis=1)

        # Get the global indices of the winners
        winner_indices = participant_indices[np.arange(n_remaining), winner_local_indices]

        # Create the selected population
        selected = self.population[winner_indices]

        # 3. Combine elite and selected individuals
        self.population = np.vstack((elite, selected))

    def roulette_selection(self, fitness, power=3):
        """Elitism + fitness-proportional roulette selection with a power transform."""
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite = self.population[elite_indices]

        non_elite_mask    = np.ones(self.N, dtype=bool)
        non_elite_mask[elite_indices] = False
        non_elite_fitness = fitness[non_elite_mask]
        non_elite_pop     = self.population[non_elite_mask]

        shifted = non_elite_fitness - non_elite_fitness.min() + 1e-6
        probs   = (shifted ** power) / (shifted ** power).sum()

        n_remaining  = self.N - self.elite_size
        selected_idx = np.random.choice(len(non_elite_pop), size=n_remaining, p=probs)
        self.population = np.vstack((elite, non_elite_pop[selected_idx]))

    def rank_roulette_selection(self, fitness):
        """Elitism + rank-based roulette selection (scale-independent)."""
        elite_indices = np.argsort(fitness)[-self.elite_size:]
        elite = self.population[elite_indices]

        non_elite_mask    = np.ones(self.N, dtype=bool)
        non_elite_mask[elite_indices] = False
        non_elite_fitness = fitness[non_elite_mask]
        non_elite_pop     = self.population[non_elite_mask]

        ranks = np.argsort(np.argsort(non_elite_fitness)) + 1
        probs = ranks / ranks.sum()

        n_remaining  = self.N - self.elite_size
        selected_idx = np.random.choice(len(non_elite_pop), size=n_remaining, p=probs)
        self.population = np.vstack((elite, non_elite_pop[selected_idx]))

    def hux_crossover(self):
        """Probabilistic Half Uniform Crossover (HUX): swap each differing gene with 50% probability."""
        indices = np.arange(self.N)
        np.random.shuffle(indices)

        parents1 = self.population[indices[::2]]
        parents2 = self.population[indices[1::2]]

        diff_mask = (np.array(parents1.copy()) != np.array(parents2.copy()))
        swap_prob_mask = np.random.rand(self.N // 2, self.n) < 0.5
        final_swap_mask = diff_mask & swap_prob_mask

        child1 = np.where(final_swap_mask, parents2, parents1)
        child2 = np.where(final_swap_mask, parents1, parents2)

        self.population = np.vstack((child1, child2))

    def uniform_crossover(self):
        """Uniform crossover at gene level with probability 0.5, applied in-place."""
        indices = np.random.permutation(self.N)
        pairs = self.population[indices].reshape(self.N // 2, 2, self.n)

        crossover_mask = np.random.rand(self.N // 2, self.n) < 0.5
        children = pairs.copy()
        children[:, 0][crossover_mask] = pairs[:, 1][crossover_mask]
        children[:, 1][crossover_mask] = pairs[:, 0][crossover_mask]

        self.population = children.reshape(self.N, self.n)

    def cg_gom_crossover(self):

        indices = np.random.permutation(self.N)

        p1 = self.population[indices[::2]]
        p2 = self.population[indices[1::2]]

        c1 = p1.copy()
        c2 = p2.copy()

        n_pairs = len(c1)

        for subset in self.fos:

            subset = np.asarray(subset)

            swap_mask = np.random.rand(n_pairs) < 0.5

            rows = np.where(swap_mask)[0]

            if len(rows) == 0:
                continue

            c1[np.ix_(rows, subset)] = p2[np.ix_(rows, subset)]
            c2[np.ix_(rows, subset)] = p1[np.ix_(rows, subset)]

        self.population = np.vstack((c1, c2))

    def hybrid_mutate(self, categorical_threshold=5):
        """Creep mutation (+/-1) for low-range features, Gaussian for wide-range ones."""
        mutation_mask = np.random.rand(self.N, self.n) < self.mr
        final_mask = mutation_mask & self.non_elite_mask

        feature_ranges = self.high_bound - self.low_bound
        is_categorical = feature_ranges <= categorical_threshold
        creep_values = np.random.choice([-1, 1], size=(self.N, self.n))

        gaussian_deltas = feature_ranges * 0.1
        gaussian_values = np.random.uniform(-gaussian_deltas, gaussian_deltas, size=(self.N, self.n))

        mutation_values = np.where(is_categorical, creep_values, gaussian_values)
        self.population += np.where(final_mask, mutation_values, 0.0)
        self.population = np.clip(self.population, self.low_bound, self.high_bound)

    def uniform_mutate(self):
        """Uniform additive perturbation within [-(range*sigma), +(range*sigma)]."""
        mutation_mask = np.random.rand(self.N, self.n) < self.mr
        final_mask    = mutation_mask & self.non_elite_mask

        feature_deltas = (self.high_bound - self.low_bound) * self.sigma
        perturbations  = np.random.uniform(-feature_deltas, feature_deltas,
                                           size=(self.N, self.n))

        self.population += np.where(final_mask, perturbations, 0.0)
        self.population  = np.clip(self.population, self.low_bound, self.high_bound)

    def gaussian_mutate(self):
        """Gaussian mutation: x'_i = x_i + N(0, sigma_i) with sigma_i = (range_i * sigma) / 3."""
        mutation_mask = np.random.rand(self.N, self.n) < self.mr
        final_mask    = mutation_mask & self.non_elite_mask

        sigma_per_feature = (self.high_bound - self.low_bound) * self.sigma / 3.0
        perturbations     = np.random.normal(0.0, sigma_per_feature,
                                             size=(self.N, self.n))

        self.population += np.where(final_mask, perturbations, 0.0)
        self.population  = np.clip(self.population, self.low_bound, self.high_bound)

    def creep_mutate(self):
        """Creep mutation: +/-1 integer steps applied to mutated genes."""
        mutation_mask = np.random.rand(self.N, self.n) < self.mr
        final_mask = mutation_mask & self.non_elite_mask
        creep_values = np.random.choice([-1, 1], size=(self.N, self.n))

        self.population += np.where(final_mask, creep_values, 0.0)
        self.population = np.clip(self.population, self.low_bound, self.high_bound)

    def adaptive_step_mutate(self):
        """Adaptive integer-step mutation: step size scales with feature range via max(1, floor(range*sigma))."""
        feature_ranges = self.high_bound - self.low_bound
        max_steps = np.maximum(1, (feature_ranges * self.sigma).astype(int))

        raw = np.random.uniform(1, max_steps + 1, size=(self.N, self.n))
        step_magnitudes = np.floor(raw).astype(int)

        directions = np.random.choice([-1, 1], size=(self.N, self.n))
        mutations  = step_magnitudes * directions

        mutation_mask = np.random.rand(self.N, self.n) < self.mr
        final_mask    = mutation_mask & self.non_elite_mask

        self.population += np.where(final_mask, mutations, 0.0)
        self.population  = np.clip(self.population, self.low_bound, self.high_bound)

    def evolve(self):
        fitness = self.evaluate_fitness()

        if self.selection_strategy == 'roulette':
            self.roulette_selection(fitness)
        elif self.selection_strategy == 'rank_roulette':
            self.rank_roulette_selection(fitness)
        else:
            self.tournament_selection(fitness)

        if self.crossover_strategy == 'hux':
            self.hux_crossover()
        elif self.crossover_strategy == 'gom':
            self.cg_gom_crossover()
        else:
            self.uniform_crossover()

        if self.mutation_strategy == 'gaussian':
            self.gaussian_mutate()
        elif self.mutation_strategy == 'uniform':
            self.uniform_mutate()
        elif self.mutation_strategy == 'creep':
            self.creep_mutate()
        elif self.mutation_strategy == 'adaptive_step':
            self.adaptive_step_mutate()
        else:
            self.hybrid_mutate(categorical_threshold=self.categorical_threshold)
        

                                            