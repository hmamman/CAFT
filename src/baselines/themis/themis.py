import itertools
import math
import os
import random
import sys
import time

import joblib
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../../../"))

from src.utils.dnn_wrapper import dnn_model_wrapper
from src.utils.helpers import generate_report, get_experiment_params


class Themis:
    def __init__(self, config, model, sensitive_params, classifier_name, samples=1000):
        self.approach_name = 'THEMIS'
        self.start_time = time.time()

        self.config = config
        self.model = dnn_model_wrapper(model)
        self.sensitive_params = sensitive_params
        self.classifier_name = classifier_name
        self.samples = samples

        self.input_bounds = config.input_bounds
        self.num_attribs = len(self.input_bounds)
        self.protected_attribs = [sp - 1 for sp in sensitive_params]
        self.protected_value_comb = self._generate_protected_combinations()

        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.total_generated = 0
        self.time_to_1000_disc = -1
        self.cumulative_efficiency = []
        self.inference_count = 0

        self.log_interval = 300
        self.initial_log_interval = self.log_interval

    def _generate_protected_combinations(self):
        res = []
        for idx in self.protected_attribs:
            bound = self.input_bounds[idx]
            res.append(list(range(bound[0], bound[1] + 1)))
        return list(itertools.product(*res))

    def check_disc(self, test):
        y = int(self.model.predict(np.array([test]))[0])
        self.inference_count += 1
        self.total_generated += 1
        self.tot_inputs.add(tuple(test))

        orig_comb = tuple(test[i] for i in self.protected_attribs)
        other_combs = [c for c in self.protected_value_comb if c != orig_comb]
        random.shuffle(other_combs)

        test2 = test[:]
        for comb in other_combs:
            for j, idx in enumerate(self.protected_attribs):
                test2[idx] = comb[j]
            y2 = int(self.model.predict(np.array([test2]))[0])
            self.inference_count += 1
            if y != y2:
                self.disc_inputs.add(tuple(test))
                self.disc_inputs_list.append(test[:])
                self.set_time_to_1000_disc()
                break

    def update_cumulative_efficiency(self, iteration):
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        if len(self.disc_inputs) >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):
        iterations = math.ceil(max_samples / self.samples)

        for i in range(iterations):
            for _ in range(self.samples):
                test = [random.randint(b[0], b[1]) for b in self.input_bounds]
                self.check_disc(test)

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
        additional_data = {'inference_count': self.inference_count}
        sens_names = [self.config.sens_name[sp] for sp in self.sensitive_params]
        generate_report(
            approach_name=self.approach_name,
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=','.join(map(str, sorted(sens_names))),
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
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

    if classifier_name in ('dnn', 'ftt', 'tabt'):
        import tensorflow as tf
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.pkl'
        model = joblib.load(classifier_path)

    themis = Themis(
        config=config,
        model=model,
        sensitive_params=sensitive_params,
        classifier_name=classifier_name,
    )

    themis.run(max_allowed_time=max_allowed_time)
