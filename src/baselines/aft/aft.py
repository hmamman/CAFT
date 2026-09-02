import os
import sys
import time
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import itertools


# Get the absolute path to the directory where vbtx.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from vbtx.py
sys.path.append(os.path.join(base_path, "../../../"))

from src.utils.dnn_wrapper import dnn_model_wrapper
from src.baselines.aft.PathSearcher import PathSearcher, IntervalPool
import logging
from src.utils.helpers import generate_report


class AFT:
    def __str__(self):
        return 'AFT'

    def __init__(self, config, model, classifier_name, sensitive_params):
        self.start_time = time.time()
        self.config = config
        self.model = dnn_model_wrapper(model)
        self.classifier_name = classifier_name
        self.sensitive_params = sensitive_params

        # Initialize tracking variables (matching VBTX)
        self.disc_inputs = set()
        self.disc_inputs_list = []
        self.tot_inputs = set()
        self.total_generated = 0
        self.elapsed_time = 0
        self.time_to_1000_disc = -1
        self.cumulative_efficiency = []
        self.inference_count = 0

        # AFT specific initialization
        self.train_data = list()
        self.no_train_data_sample = 5000

        # Convert protected attributes format
        self.protected_list_no = [sens_param - 1 for sens_param in sensitive_params]  # Convert to 0-based index
        self.protected_list = [self.config.feature_name[i] for i in self.protected_list_no]

        # Generate protected value combinations
        self.protected_value_comb = self.generate_protected_value_combination()
        self.no_prot = len(self.protected_list_no)

    def generate_protected_value_combination(self):
        """Generate all possible combinations of protected attribute values"""
        res = list()
        for index_protected in self.protected_list_no:
            bound = self.config.input_bounds[index_protected]
            res.append(list(range(bound[0], bound[1] + 1)))
        return list(itertools.product(*res))

    def create_train_data(self, num):
        """Generate initial training data"""
        self.train_data = list()
        for _ in range(num):
            temp = list()
            for i in range(len(self.config.input_bounds)):
                bound = self.config.input_bounds[i]
                temp.append(random.randint(bound[0], bound[1]))
            temp.append(int(self.model.predict(np.array([temp]))[0]))
            self.train_data.append(temp)

    def train_approximate_DT(self, max_leaf_nodes=1000):
        """Train decision tree approximation of the model"""
        X = [item[:-1] for item in self.train_data]
        Y = [item[-1] for item in self.train_data]
        clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=max_leaf_nodes)
        return clf.fit(X, Y)

    def update_cumulative_efficiency(self, iteration):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])


    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def update_tracking_metrics(self, test_data, disc_data=None):
        if disc_data is None:
            disc_data = []

        test_count = len(test_data)
        disc_count = len(disc_data)

        # Ensure both lists have the same length by padding the shorter one with empty lists
        max_count = max(test_count, disc_count)
        test_data = test_data + [[]] * (max_count - test_count)
        disc_data = disc_data + [[]] * (max_count - disc_count)

        for test_sample, disc_sample in zip(test_data, disc_data):
            # Process test sample
            input_tuple = tuple(test_sample[:-1])
            self.tot_inputs.add(input_tuple)
            self.total_generated += 1

            # Process discriminative sample
            if disc_sample:
                disc_input = tuple(disc_sample[:-1])
                self.disc_inputs.add(disc_input)
                self.disc_inputs_list.append(disc_input)

            # Update metrics
            self.set_time_to_1000_disc()


    def run(self, limit=1000, max_allowed_time=3600):
        """Main testing loop matching VBTX's interface"""
        self.start_time = time.time()
        restart_flag = True
        no_new_train_count = 0
        count = 300
        IntervalP = IntervalPool()

        loop = 0

        while True:
            # Time-based reporting and stopping conditions
            end = time.time()
            use_time = end - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time:
                break

            if self.total_generated >= limit * limit:
                break

            # Generate training data from input space
            if restart_flag:
                self.create_train_data(self.no_train_data_sample)
                restart_flag = False

            # Train approximate decision tree
            DT = self.train_approximate_DT()

            # Generate test cases through path sampling
            sampler = PathSearcher(
                DT=DT,
                CuT=self.model,
                data_range=self.config.input_bounds,
                protected_value_comb=self.protected_value_comb,
                protected_list_no=self.protected_list_no,
                IntervalP=IntervalP,
            )

            satFlag = sampler.sample(
                dt_search_mode="random+flip",
                check_type="themis",
                MaxTry=10000,
                MaxDiscPathPair=100,
                max_train_data_each_path=10,
                max_sample_each_path=100
            )

            if satFlag:
                # If at least one test case is found,
                # update test cases and discriminatory instances
                test_data = sampler.get_test_data()

                new_disc_data = sampler.get_disc_data()

                if len(new_disc_data) == 0:
                    # If no one discriminatory instance is found in this iteration, then restart
                    self.update_tracking_metrics(test_data)
                    restart_flag = True
                    continue
                else:
                    self.update_tracking_metrics(test_data, new_disc_data)

                # Update the training data
                new_train_data = sampler.get_train_data()
                self.train_data += new_train_data
                if len(new_train_data) == 0:
                    no_new_train_count += 1
                    if no_new_train_count >= 5:
                        restart_flag = True
                        no_new_train_count = 0
                else:
                    no_new_train_count = 0
            else:
                # If no one test case could be found from the decision tree, then restart the loop
                restart_flag = True

            self.update_cumulative_efficiency(loop)
            self.inference_count += sampler.inference_count
            loop += 1

        elapsed_time = time.time() - self.start_time
        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        additional_data = {'inference_count': self.inference_count}
        sens_names = [self.config.sens_name[sens_param] for sens_param in self.sensitive_params]
        generate_report(
            approach_name='AFT',
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
    from src.utils.helpers import get_experiment_params
    import joblib
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    if classifier_name == 'dnn':
        import tensorflow as tf

        classifier_path = f'models/{config.dataset_name}/dnn_slfc.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name}_standard_unfair.pkl'
        model = joblib.load(classifier_path)

    aft = AFT(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
    )

    aft.run(max_allowed_time=max_allowed_time)