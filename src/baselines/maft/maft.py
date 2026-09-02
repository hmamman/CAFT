import time
import numpy as np
import tensorflow as tf
import sys, os

# Adjust the import path as needed
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../../../"))
from src.utils.helpers import get_data, generate_report, get_experiment_params
from src.baselines.maft import generation_utilities

# allocate GPU and set dynamic memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# make outputs stable across runs for validation
# alternatively remove them when dealing with real-world issues
np.random.seed(42)
tf.random.set_seed(42)

class MAFT:
    def __init__(self, config, model, sensitive_params, classifier_name, cluster_num=4, max_global=1000, max_local=1000, max_iter=10):
        self.approach_name = 'MAFT'
        self.start_time = time.time()

        self.config = config
        self.constraint = np.array(self.config.input_bounds)
        self.sensitive_params = sensitive_params
        self.protected_attribs = np.array([s - 1 for s in sensitive_params])
        self.num_attribs = len(self.constraint)
        self.model = model
        self.classifier_name = classifier_name
        self.cluster_num = cluster_num
        self.max_global = max_global
        self.max_local = max_local
        self.max_iter = max_iter
        self.perturbation_size = 1

        # Initialize testing results storage
        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.dataset_name = self.config.dataset_name

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0
        self.gradient_calls = 0
        self.log_interval = 300
        self.initial_log_interval = self.log_interval


    def compute_grad(self, x, perturbation_size=1e-4):
        # compute the gradient of model perdictions w.r.t input attributes
        h = perturbation_size
        n = len(x)
        e = np.empty(n)
        e.fill(h)
        E = np.diag(e)
        X = np.repeat([x], n, axis=0)
        X = X + E
        X = tf.constant(X, dtype=tf.float32)
        Y = self.model(X)
        self.inference_count += 1
        self.gradient_calls += 1
        x = tf.constant([x], dtype=tf.float32)
        y_pred = self.model(x)
        self.inference_count += 1
        self.gradient_calls += 1
        gradient = (Y - y_pred) / h
        gradient = tf.reshape(gradient, [1, -1])
        return gradient[0].numpy() if self.model(x) > 0.5 else -gradient[0].numpy()

    
    def is_discriminatory(self, x, similar_x):
        # identify whether the instance is discriminatory w.r.t. the model
        y_pred = (self.model(tf.constant([x])) > 0.5)
        self.inference_count += 1
        for x_new in similar_x:
            self.inference_count += 1
            if (self.model(tf.constant([x_new])) > 0.5) != y_pred:
                return True
        return False
    
    def max_diff(self, x, similar_x):
        # select a similar instance such that the DNN outputs on them are maximally different

        y_pred_proba = self.model(tf.constant([x]))
        self.inference_count += 1
        def distance(x_new, model=self.model):
            return np.sum(np.square(y_pred_proba - model(tf.constant([x_new]))))
        max_dist = 0.0
        x_potential_pair = x.copy()
        for x_new in similar_x:
            d= distance(x_new)
            self.inference_count += 1
            if d > max_dist:
                max_dist = d
                x_potential_pair = x_new.copy()
        return x_potential_pair
    
    def find_pair(self, x, similar_x):
        # find a discriminatory pair given an individual discriminatory instance

        pairs = np.empty(shape=(0, len(x)))
        y_pred = (self.model(tf.constant([x])) > 0.5)
        self.inference_count += 1
        for x_pair in similar_x:
            self.inference_count += 1
            if (self.model(tf.constant([x_pair])) > 0.5) != y_pred:
                pairs = np.append(pairs, [x_pair], axis=0)
        selected_p = generation_utilities.random_pick([1.0 / pairs.shape[0]] * pairs.shape[0])
        return pairs[selected_p]

    
    def maft_generation(self,X, seeds, decay=0.5, update_interval=5, s_g=1.0, s_l=1.0, epsilon=1e-6, max_allowed_time=3600):
        # perform global generation and local generation successively on each single seed
        direction = [-1, 1]
        for index, instance in enumerate(seeds):
            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=use_time, is_log=True)

            if self.log_interval >= max_allowed_time or self.total_generated >= self.max_global * self.max_local:
                break

            x1 = instance.copy()

            flag = False
            grad1 = np.zeros_like(X[0]).astype(float)
            grad2 = np.zeros_like(X[0]).astype(float)
            for j in range(self.max_iter):

                use_time = time.time() - self.start_time
                if use_time >= self.log_interval:
                    self.log_interval += self.initial_log_interval
                    self.report(elapsed_time=use_time, is_log=True)

                if self.log_interval >= max_allowed_time or self.total_generated >= self.max_global*self.max_local:
                    break
                similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                if self.is_discriminatory(x1, similar_x1):
                    self.inference_count += 1
                    self.disc_inputs.add(tuple(x1))
                    self.set_time_to_1000_disc()

                    flag = True
                    break
                x2 = self.max_diff(x1, similar_x1)
                grad1 = decay * grad1 + self.compute_grad(x1)
                grad2 = decay * grad2 + self.compute_grad(x2)
                direction_g = np.zeros_like(X[0])
                sign_grad1 = np.sign(grad1)
                sign_grad2 = np.sign(grad2)
                for attrib in range(self.num_attribs):
                    if attrib not in self.protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                        direction_g[attrib] = (-1) * sign_grad1[attrib]
                x1 = x1 + s_g * direction_g
                x1 = generation_utilities.clip(x1, self.constraint)

                self.tot_inputs.add(tuple(x1))
                self.total_generated += 1

            if flag == True:
                x0 = x1.copy()
                similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                x2 = self.max_diff(x1, similar_x1)
                grad1 = self.compute_grad(x1)
                grad2 = self.compute_grad(x2)
                p = generation_utilities.normalization(grad1, grad2, self.protected_attribs, epsilon)
                p0 = p.copy()
                suc_iter = 0
                for _ in range(self.max_local):
                    use_time = time.time() - self.start_time
                    if use_time >= self.log_interval:
                        self.log_interval += self.initial_log_interval
                        self.report(elapsed_time=use_time, is_log=True)

                    if self.log_interval >= max_allowed_time or self.total_generated >= self.max_global * self.max_local:
                        break

                    if suc_iter >= update_interval:
                        similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                        x2 = self.find_pair(x1, similar_x1)
                        grad1 = self.compute_grad(x1)
                        grad2 = self.compute_grad(x2)
                        p = generation_utilities.normalization(grad1, grad2, self.protected_attribs, epsilon)
                        suc_iter = 0
                    suc_iter += 1
                    a = generation_utilities.random_pick(p)
                    s = generation_utilities.random_pick([0.5, 0.5])
                    x1[a] = x1[a] + direction[s] * s_l
                    x1 = generation_utilities.clip(x1, self.constraint)
                    
                    self.tot_inputs.add(tuple(x1))
                    self.total_generated += 1
                    
                    similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                    if self.is_discriminatory(x1, similar_x1):
                        self.disc_inputs.add(tuple(x1))
                        self.set_time_to_1000_disc()
                    else:
                        x1 = x0.copy()
                        p = p0.copy()
                        suc_iter = 0

            self.update_cumulative_efficiency(index)



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

    def run(self, max_allowed_time=3600):

        data = get_data(self.config.dataset_name)
        X, Y, input_shape, nb_classes = data()

        clustered_data = generation_utilities.clustering(X, self.cluster_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(self.max_global):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), self.cluster_num, i % self.cluster_num)
            seeds = np.append(seeds, [new_seed], axis=0)

        self.maft_generation(seeds=seeds, X=X, max_allowed_time=max_allowed_time)
        self.report(elapsed_time=time.time()-self.start_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        additional_data = {
            'inference_count': self.inference_count,
            'gradient_calls': self.gradient_calls
        }
        sens_names = [self.config.sens_name[sens_param] for sens_param in self.sensitive_params]

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
    approach_name, config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {",".join(map(str, sorted(sensitive_names)))}')
    print('')

    import tensorflow as tf

    classifier_path = f'models/{config.dataset_name}/dnn.keras'
    model = tf.keras.models.load_model(classifier_path)

    maft = MAFT(
        config=config,
        model=model,
        sensitive_params=sensitive_params,
        classifier_name=classifier_name,
    )

    maft.run(max_allowed_time=max_allowed_time)
