import itertools
import math
import os
import sys
import tensorflow as tf

# Get the absolute path to the directory where expga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../../../"))

from src.utils.dnn_wrapper import dnn_model_wrapper
from src.utils.helpers import get_data, generate_report, get_experiment_params
import numpy as np
import random
import time
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from src.baselines.expga.genetic_algorithm import GA


class ExpGA:
    def __init__(self, config, model, classifier_name, sensitive_params, threshold_l=10, threshold=0):
        self.approach_name = 'ExpGA'
        self.start_time = time.time()
        self.config = config
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.tot_inputs = set()
        self.location = np.zeros(40)
        self.threshold_l = threshold_l
        self.threshold = threshold
        self.input_bounds = np.array(self.config.input_bounds)
        self.sensitive_params = sensitive_params
        self.model = dnn_model_wrapper(model=model)
        self.classifier_name = classifier_name

        self.protected_attribs = [sens_param - 1 for sens_param in sensitive_params]

        self.num_attribs = len(config.input_bounds)
        self.non_protected_attribs = [i for i in range(len(config.input_bounds)) if i not in self.protected_attribs]
        self.protected_domains = [list(range(int(self.input_bounds[i][0]), int(self.input_bounds[i][1]) + 1)) for i in
                                  self.protected_attribs]

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0

        self.log_interval = 300
        self.initial_log_interval = self.log_interval

    def set_threshold_l(self):
        dataset_thresholds = {
            "census": 7,
            "bank": 10,
            "credit": 14,
            "meps": 10,
            "compas": 10
        }

        dataset_name = self.config.dataset_name
        if dataset_name in dataset_thresholds:
            self.threshold_l = dataset_thresholds[dataset_name]
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    def construct_explainer(self, train_vectors, feature_names, class_names):
        explainer = LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                         class_names=class_names, discretize_continuous=False)
        return explainer

    def shap_value(self, test_vectors):
        background = shap.kmeans(test_vectors, 10)
        explainer = shap.KernelExplainer(self.model.predict_proba, background)
        shap_values = explainer.shap_values(test_vectors)
        return shap_values

    def search_seed(self, feature_names, protected_attribs, explainer, train_vectors, num, X_ori):
        seed = []
        for x in train_vectors:
            self.tot_inputs.add(tuple(x))

            self.total_generated += 1

            exp = explainer.explain_instance(x, self.model.predict_proba, num_features=num)
            explain_labels = exp.available_labels()
            exp_result = exp.as_list(label=explain_labels[0])
            rank = [item[0] for item in exp_result]

            # Check if any sensitive attribute appears in top threshold_l positions
            should_add_to_seed = False
            for attrib in protected_attribs:
                sens_name = feature_names[attrib]  # Get feature name by index
                if sens_name in rank:
                    loc = rank.index(sens_name)
                    self.location[loc] += 1
                    if loc < self.threshold_l:
                        should_add_to_seed = True
                        break  # If any sensitive attribute meets criteria, add to seed

            if should_add_to_seed:
                seed.append(x)

            if len(seed) >= 200:
                return seed
        return seed

    def search_seed_shap(self, feature_names, sens_name, shap_values, train_vectors):
        seed = []
        for i in range(len(shap_values[0])):
            sample = shap_values[0][i]
            sorted_shap_value = sorted(
                [[feature_names[j], sample[j]] for j in range(len(sample))],
                key=lambda x: abs(x[1]), reverse=True
            )
            rank = [item[0] for item in sorted_shap_value]
            loc = rank.index(sens_name)
            if loc < 10:
                seed.append(train_vectors[i])
            if len(seed) > 10:
                return seed
        return seed

    class GlobalDiscovery:
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, iteration, params, input_bounds, protected_attribs):
            samples = []
            while len(samples) < iteration:
                x = np.zeros(params)
                for i in range(params):
                    random.seed(time.time())
                    x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

                for attrib in protected_attribs:
                    x[attrib - 1] = 0

                samples.append(x)
            return samples

    def discrimination_object_func(self):

        def func(indv):
            """
            Sequential discrimination evaluation for a single input.
            Supports multiple protected attributes and domains.
            Behavior matches evaluate_local:
              - Stop at first discriminatory difference
              - Return 2 * abs(out1 - out0) + 1
            """

            x = np.asarray(indv, dtype=int).copy()

            # --- Clip to input bounds ---
            for j in range(self.num_attribs):
                low, high = self.input_bounds[j]
                x[j] = np.clip(x[j], int(low), int(high))

            # --- Bookkeeping ---
            self.total_generated += 1
            self.tot_inputs.add(tuple(int(v) for v in x))

            # --- All protected combinations ---
            all_combs = list(itertools.product(*self.protected_domains))

            # --- Base prediction ---
            out0 = self.model.predict(x.reshape(1, -1))[0]

            # --- Sequentially evaluate variants ---
            final_diff = 0

            for comb in all_combs:

                # Skip original combination
                is_original = True
                for j, attr_idx in enumerate(self.protected_attribs):
                    if x[attr_idx] != comb[j]:
                        is_original = False
                        break
                if is_original:
                    continue

                # Build variant
                x_var = x.copy()
                for j, attr_idx in enumerate(self.protected_attribs):
                    x_var[attr_idx] = comb[j]

                # Predict variant
                out1 = self.model.predict(x_var.reshape(1, -1))[0]

                # Compute difference
                diff = abs(out1 - out0)

                # If difference found → stop like evaluate_local
                if diff > 0 and \
                        (tuple(x) not in self.global_disc_inputs) and \
                        (tuple(x) not in self.local_disc_inputs):
                    self.local_disc_inputs.add(tuple(int(v) for v in x))
                    self.local_disc_inputs_list.append(x.tolist())
                    self.set_time_to_1000_disc()

                    return 2 * diff + 1

                # Track last diff (in case no violation found)
                final_diff = diff

            # No discriminatory difference found
            return 2 * final_diff + 1

        return func

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

    def run(self, max_global=1000, max_local=1000, max_allowed_time=3600):
        self.total_generated = 0
        self.set_threshold_l()

        feature_names = self.config.feature_name
        class_names = self.config.class_name
        params = self.config.params

        data = get_data(self.config.dataset_name)
        X, Y, input_shape, nb_classes = data()

        global_discovery = self.GlobalDiscovery()

        train_samples = global_discovery(max_global, params, self.input_bounds, self.protected_attribs)
        train_samples = np.array(train_samples)
        np.random.shuffle(train_samples)

        explainer = self.construct_explainer(X, feature_names, class_names)
        seed = self.search_seed(feature_names, self.protected_attribs, explainer, train_samples, params, X)
        print('Finish Searchseed')

        for inp in seed:
            inp0 = np.array([int(i) for i in inp])
            self.global_disc_inputs.add(tuple(inp0))
            self.global_disc_inputs_list.append(inp0)
            self.set_time_to_1000_disc()

        print("Finished Global Search")
        print('length of total input is:' + str(len(self.tot_inputs)))
        print('length of global discovery is:' + str(len(self.global_disc_inputs_list)))

        end = time.time()

        print('Total time:' + str(end - self.start_time))

        print("")
        print("Starting Local Search")

        nums = self.global_disc_inputs_list
        DNA_SIZE = len(self.input_bounds)
        ga = GA(nums=nums, bound=self.input_bounds, func=self.discrimination_object_func(),
                DNA_SIZE=DNA_SIZE, cross_rate=0.9, mutation=0.05)
        
        max_samples = (max_local * max_global) - self.total_generated

        max_iter = math.ceil(((max_local * max_global) - self.total_generated) / len(nums))

        for i in range(max_iter):
            ga.evolution()
            self.update_cumulative_efficiency(i)
            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=use_time, is_log=True)

            if self.log_interval >= max_allowed_time or self.total_generated >= max_samples:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        disc_inputs = self.local_disc_inputs | self.global_disc_inputs

        additional_data = {'inference_count': self.inference_count}
        sens_names = [self.config.sens_name[sens_param] for sens_param in self.sensitive_params]
        generate_report(
            approach_name=f'{self.approach_name}',
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
    config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {",".join(map(str, sorted(sensitive_names)))}')
    print('')

    if classifier_name == 'dnn' or classifier_name == 'ftt' or classifier_name == 'tabt':
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.pkl'
        model = joblib.load(classifier_path)

    expga = ExpGA(
        config=config,
        model=model,
        sensitive_params=sensitive_params,
        classifier_name=classifier_name
    )

    expga.run(max_allowed_time=max_allowed_time)
