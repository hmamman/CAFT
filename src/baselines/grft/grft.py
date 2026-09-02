"""
This python file calls functions from experiments.py to reproduce the main experiments of our paper.
"""
import os
import sys

import joblib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import random
import itertools
import copy
import argparse
from tensorflow.keras import backend as K

# Get the absolute path to the directory where fairamga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairamga.py
sys.path.append(os.path.join(base_path, "../../../"))

from src.baselines.grft.GeneticAlgorithm import Population
from src.baselines.grft import generation_utilities
from src.utils.dnn_wrapper import dnn_model_wrapper
from src.utils.helpers import get_experiment_params, get_data, generate_report


class GRFT:
    def __init__(self, config, model, classifier_name, sensitive_params):
        # instance (class) variables as requested
        self.max_allowed_time = None
        self.max_samples = None
        self.sensitive_params = sensitive_params
        self.start_time = time.time()
        self.config = config
        self.model = dnn_model_wrapper(model)
        self.classifier_name = classifier_name
        # protected attributes & bounds
        self.protected_attribs = [param-1 for param in sensitive_params]
        self.constraint = np.array(config.input_bounds)

        # dataset and shape attributes
        self.X = self.load_X_train()
        # num_attribs = number of columns/features
        self.num_attribs = len(self.X[0]) if len(self.X) and hasattr(self.X[0], "__len__") else len(self.X)
        self.non_protected_attribs = [i for i in range(self.num_attribs) if i not in self.protected_attribs]
        self.global_disc_inputs = set()
        self.local_disc_inputs = set()
        self.disc_inputs = set()
        self.tot_inputs = set()
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0
        self.iteration = 0
        self.approach_name = 'GRFT'

    # ---------- utilities using instance variables ----------
    def load_X_train(self):
        data = get_data(self.config.dataset_name)
        X, Y, input_shape, nb_classes = data()
        return X

    def similar_set(self, X):
        similar_X = []
        protected_domain = []
        for i in self.protected_attribs:
            protected_domain.append(list(range(int(self.constraint[i][0]), int(self.constraint[i][1]) + 1)))
        all_combs = np.array(list(itertools.product(*protected_domain)))
        for comb in all_combs:
            X_new = copy.deepcopy(X)
            for a, c in zip(self.protected_attribs, comb):
                X_new[:, a] = c
            similar_X.append(X_new)
        return similar_X

    def similar_set_(self, X):
        similar_X = np.empty(shape=(0, self.num_attribs))
        X = np.array(X)
        protected_domain = []
        for i in self.protected_attribs:
            protected_domain.append(list(range(int(self.constraint[i][0]), int(self.constraint[i][1]) + 1)))
        all_combs = np.array(list(itertools.product(*protected_domain)))
        for comb in all_combs:
            X_new = copy.deepcopy(X)
            for a, c in zip(self.protected_attribs, comb):
                X_new[:, a] = c
            similar_X = np.append(similar_X, X_new, axis=0)
        return similar_X

    def _rand_step(self):
        return random.randint(-5, 5)

    def create_image_indvs(self, seed, num):
        indivs = [seed]
        index = np.random.choice(self.non_protected_attribs, 2, replace=False)
        for _ in range(num - 1):
            temp = seed.copy()
            temp[index[0]] = temp[index[0]] + self._rand_step()
            temp[index[1]] = temp[index[1]] + self._rand_step()
            indivs.append(temp)
        indivs = generation_utilities.clip(indivs, self.constraint)
        return np.array(indivs)

    def untarget_object_func(self, target_ratio=0.9):
        def func(indvs):
            x_array = np.array(indvs)
            similar_x = self.similar_set_(x_array)
            # Predict probabilities to maintain continuous fitness gradient
            y_pred = self.model.predict_proba(x_array)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = y_pred[:, 1]
            else:
                y_pred = y_pred.flatten()
                
            pred_similar_x = self.model.predict_proba(similar_x)
            if len(pred_similar_x.shape) > 1 and pred_similar_x.shape[1] > 1:
                pred_similar_x = pred_similar_x[:, 1]
            else:
                pred_similar_x = pred_similar_x.flatten()

            self.inference_count += 2

            # Use probability thresholds to define actual IDI matches
            pred_similar_x_label = (pred_similar_x > 0.5).astype("int")
            # Reshape labels to (num_variants, num_individuals)
            pred_similar_x_label = pred_similar_x_label.reshape(-1, len(indvs))
            
            unique_columns = [len(np.unique(pred_similar_x_label[:, col])) > 1
                              for col in range(pred_similar_x_label.shape[1])]
            new_index_result = np.where(unique_columns)[0]
            
            # Reshape probabilities to (num_variants, num_individuals) for proper broadcasting
            pred_similar_x_reshaped = pred_similar_x.reshape(-1, len(indvs))
            # Broadcast y_pred (M,) against pred_similar_x_reshaped (num_variants, M)
            fitness = np.sum(np.abs(pred_similar_x_reshaped - y_pred), axis=0)

            self.total_generated += len(indvs)
            self.tot_inputs.update(tuple(arr.tolist()) for arr in indvs)

            return x_array[new_index_result], new_index_result, fitness

        return func

    def build_mutate_func(self):
        def func(indv):
            index = np.random.choice(self.non_protected_attribs, 4, replace=False)
            mutate_indv = indv.copy()
            mutate_indv[index[0]] = mutate_indv[index[0]] + random.randint(-3, 3)
            mutate_indv[index[1]] = mutate_indv[index[1]] + random.randint(-3, 3)
            mutate_indv[index[2]] = mutate_indv[index[2]] + random.randint(-3, 3)
            mutate_indv = generation_utilities.clip(mutate_indv, self.constraint)
            return mutate_indv

        return func

    def build_mutate_func2(self):
        def func(indv):
            index = np.random.choice(self.non_protected_attribs, 1, replace=False)
            mutate_indv = np.array(indv.copy())
            if random.random() < 0.5:
                mutate_indv[:, index] += 1
            else:
                mutate_indv[:, index] -= 1
            mutate_indv = generation_utilities.clip(mutate_indv, self.constraint)
            return mutate_indv

        return func

    # ---------- core phases using instance variables ----------
    def global_generation(self, seeds, max_iter, s_g, pop_num=100):
        print("Global generation started...")
        global_disc = np.empty(shape=(0, self.num_attribs))
        all_gen_g = np.empty(shape=(0, self.num_attribs))
        try_times = 0
        for i in range(len(seeds)):
            x1 = seeds[i].copy()
            inds = self.create_image_indvs(x1, pop_num)
            mutation_function = self.build_mutate_func()
            mutate2_function = self.build_mutate_func2()
            fitness_compute_function = self.untarget_object_func(target_ratio=0)
            pop = Population(
                individuals=inds,
                mutation_function=mutation_function,
                fitness_compute_function=fitness_compute_function,
                mutate_func2=mutate2_function,
                num_attribs=self.num_attribs,
                first_attack=1,
                subtotal=10,
                max_time=1000000000,
                seed=x1,
                max_iteration=max_iter
            )
            try_times += pop.round

            for di in pop.discriminatory:
                global_disc = np.append(global_disc, [di], axis=0)
                self.global_disc_inputs.update(tuple(arr.tolist()) for arr in pop.discriminatory)
                self.set_time_to_1000_disc()
                self.update_cumulative_efficiency(self.iteration)

            elapsed_time = time.time() - self.start_time
            if self.total_generated >= self.max_samples or elapsed_time >= self.max_allowed_time:
                break

            self.iteration += 1

        global_disc = np.array(list(set([tuple(id) for id in global_disc])))
        return global_disc, all_gen_g, try_times

    def local_generation(self, max_local, global_disc, local_step_size, epsilon=1e-6):
        print("Local generation started...")
        direction = [-1, 1]
        local_disc = np.empty((0, self.num_attribs))
        all_gen_l = np.empty((0, self.num_attribs))
        gi_id = np.array(global_disc, copy=True)

        for _ in range(max_local):
            elapsed_time = time.time() - self.start_time

            if self.total_generated >= self.max_samples or elapsed_time >= self.max_allowed_time:
                break

            if gi_id.size == 0:
                print("Warning: global_disc is empty, skipping iteration")
                break

            if gi_id.ndim == 1:
                gi_id = gi_id.reshape(1, -1)

            # Store previous state for rollback
            g0_id = gi_id.copy()

            a = np.random.choice(self.non_protected_attribs)
            s = random.choice([0, 1])
            gi_id[:, a] += direction[s] * local_step_size
            gi_id = generation_utilities.clip(gi_id, self.constraint)

            self.total_generated += len(gi_id)
            self.tot_inputs.update(tuple(arr.tolist()) for arr in gi_id)

            # Build "similar" variants (protected attributes flipped)
            similar_list = self.similar_set(gi_id)

            # Base predictions
            y_pred_label = self.model.predict(gi_id)
            self.inference_count += 1

            # Predictions for all variants
            if len(similar_list) == 0:
                index_different = np.array([], dtype=int)
            else:
                variant_labels = [self.model.predict(sim_x) for sim_x in similar_list]
                self.inference_count += len(variant_labels)

                mat = np.vstack(variant_labels)
                diff_mask = np.any(mat != y_pred_label, axis=0)
                index_different = np.where(diff_mask)[0]

            # Collect locally discriminatory rows
            if index_different.size:
                idi = gi_id[index_different]
                local_disc = np.append(local_disc, idi, axis=0)

                self.local_disc_inputs.update(tuple(arr.tolist()) for arr in idi)
                self.set_time_to_1000_disc()
                self.update_cumulative_efficiency(self.iteration)

            # Rollback failed mutations
            mask = np.ones(len(gi_id), dtype=bool)
            if index_different.size:
                mask[index_different] = False
            gi_id[mask] = g0_id[mask]

            self.iteration += 1

        # Deduplicate results
        if local_disc.size:
            local_disc = np.array(list({tuple(row) for row in local_disc}))
        else:
            local_disc = np.empty((0, self.num_attribs))

        return local_disc, all_gen_l, 0

    def run(self, max_local=1000, max_iter=10, s_g=1.0, local_step_size=1.0, epsilon=1e-6, max_samples=1000 * 1000,
            max_allowed_time=3600):
        self.start_time = time.time()
        self.max_samples = max_samples
        self.max_allowed_time = max_allowed_time

        while self.total_generated < self.max_samples and time.time() - self.start_time < self.max_allowed_time:
            seeds = self.generate_seeds(num_seeds=100)

            global_disc, gen_g, g_gen_num = self.global_generation(seeds=seeds, max_iter=max_iter, s_g=s_g)

            local_disc, gen_l, l_gen_num = self.local_generation(max_local, global_disc, local_step_size, epsilon)

        elapsed_time = time.time() - self.start_time
        self.report(elapsed_time, is_log=False)

    def generate_seeds(self, c_num=4, num_seeds=100, fashion='Distribution'):
        clustered_data = generation_utilities.clustering(self.X, c_num)
        id_seeds = np.empty(shape=(0, self.num_attribs))
        for i in range(100000000):
            x_seed = generation_utilities.get_seed(clustered_data, len(self.X), c_num, i % c_num, fashion=fashion)
            id_seeds = np.append(id_seeds, [x_seed], axis=0)
            if len(id_seeds) >= num_seeds:
                break
        return id_seeds

    def update_cumulative_efficiency(self, iteration):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.local_disc_inputs) + len(self.global_disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.global_disc_inputs) + len(self.local_disc_inputs)
        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def report(self, elapsed_time, is_log: bool):
        disc_inputs =  self.global_disc_inputs | self.local_disc_inputs
        additional_data = {'inference_count': self.inference_count}
        sens_names = [self.config.sens_name[sens_param] for sens_param in self.sensitive_params]

        generate_report(
            approach_name=self.approach_name,
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=','.join(map(str, sorted(sens_names))),
            tot_inputs=self.tot_inputs,
            disc_inputs=disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
            **additional_data,
        )


if __name__ == '__main__':
    approach_name, config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {",".join(map(str, sorted(sensitive_names)))}')
    print('')

    if classifier_name == 'dnn':
        import tensorflow as tf

        classifier_path = f'models/{config.dataset_name}/dnn.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:

        classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
        model = joblib.load(classifier_path)

    grft = GRFT(config=config, model=model, sensitive_params=sensitive_params, classifier_name=classifier_name)

    grft.run()
