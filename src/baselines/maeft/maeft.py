import copy
import itertools
import math
import os
import random
import sys
import time
import warnings
from collections import deque

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.metrics import silhouette_score
from torch.distributions import Categorical

# Set base path
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../../../"))

from src.utils.dnn_wrapper import dnn_model_wrapper
from src.utils.helpers import get_data, generate_report, get_experiment_params

warnings.filterwarnings("ignore")


class DummySpace:
    def __init__(self, n=None, shape=None):
        self.n = n
        self.shape = shape


class Dueling_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.V = nn.Linear(128, 1, bias=False)
        self.A = nn.Linear(128, action_dim, bias=False)
        
    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Memory:
    def __init__(self, len_limit):
        self.tansition = deque(maxlen=len_limit)
        self.size = len_limit

    def update(self, state, action, reward, state_next):
        self.tansition.append([state, action, reward, state_next])

    def sample(self, batch_size, st_act):
        length = self.size if len(self.tansition) >= self.size else len(self.tansition)
        idx = random.sample(range(0, length), batch_size)
        st = []
        act = []
        reward = []
        st_next = []
        state_action_next = []
        for i in idx:
            temp = self.tansition[i]
            st.append(temp[0])
            act.append(temp[1])
            reward.append(temp[2])
            st_next.append(temp[3])
            state_action_next.append([math.sqrt(x+1) for x in st_act[tuple(temp[3])][0]])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int16)
        reward = np.array(reward, dtype=np.float32)
        st_next = np.array(st_next, dtype=np.float32)
        state_action_next = np.array(state_action_next, dtype=np.float32)
        return st, act, reward, st_next, state_action_next


class DoubleDuelingDQN(object):
    def __init__(self, n_st, n_act):
        super(DoubleDuelingDQN, self).__init__()
        sys.setrecursionlimit(10000)
        self.n_st = n_st
        self.n_act = n_act
        self.model = Dueling_Net(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.step = 0
        self.gamma = 0.995
        self.memory_size = 10000
        self.train_step = 5000
        self.memory = Memory(self.memory_size)
        self.batch_size = 64
        self.target_update_freq = 100
        self.loss = 0
        self.T = 0.25
        self.tau = 0.5

    def stock_experience(self, st, act, r, st_dash):
        self.memory.update(st, act, r, st_dash)
    
    def forward(self, st, act, r, st_next, st_act_next):
        s = torch.unsqueeze(torch.tensor(st, dtype=torch.float), 0)
        s_next = torch.unsqueeze(torch.tensor(st_next, dtype=torch.float), 0)
        action_batch = torch.tensor(act).unsqueeze(1).type(torch.int64)
        r = torch.unsqueeze(torch.tensor(r.reshape(-1,1), dtype=torch.float), 0)[0]
        Q = self.model(s)[0]
        with torch.no_grad():
            Q_next = self.get_batch_action(s_next, st_act_next)
            Q_next_target = self.target_model(s_next)[0]
            next_target_q_value_batch = Q_next_target.gather(dim=1, index=Q_next)
            target = r + self.gamma * next_target_q_value_batch
        Q = Q.gather(dim=1, index=action_batch)
        loss = F.mse_loss(Q, target) 
        self.loss = loss.item()
        return loss 
    
    def experience_replay(self, st_act):
        st, act, reward, st_next, state_action_next = self.memory.sample(self.batch_size, st_act)
        self.optimizer.zero_grad()
        loss = self.forward(st, act, reward, st_next, state_action_next)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state, st_act):
        if self.step <= self.train_step:
            return np.random.randint(0, self.n_act)
        else:
            with torch.no_grad():
                st = tuple(state)
                state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
                q = self.model(state)
                q = q[0].data
                if st in st_act.keys():
                    to_divide = [math.sqrt(x+1)  for x in st_act[st][0]]
                    to_divide = torch.FloatTensor(to_divide)
                    q = q / to_divide  
                q = q / self.T
                func = nn.Softmax(dim=0)
                action_probs = func(q) 
                dist = Categorical(probs=action_probs)
                action = dist.sample()
            return action.numpy()  

    def get_batch_action(self, state, st_act):
        with torch.no_grad():
            q = self.model(state)
            q = q[0].data
            q = q / torch.FloatTensor(st_act)
            q = q / self.T
            func = nn.Softmax(dim=1)
            probs = func(q) 
            dist = Categorical(probs=probs)
            action = dist.sample() 
        return action.type(torch.int64).unsqueeze(1) 

    def train(self, st_act):
        if self.step >= self.train_step:
            self.experience_replay(st_act)
            if self.step >= 20000:
                self.target_update_freq = 500
                self.tau = 0.125
            if self.step % self.target_update_freq == 0:
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)   
        self.step += 1


class MyEnv(object):
    def __init__(self, runner):
        super(MyEnv, self).__init__()
        self.runner = runner
        self.action_space = DummySpace(n=len(self.runner.action_table))
        self.current_sample = []
        self.episode_end = 500
        self.counts = 0
        self.observation_space = DummySpace(
            shape=(self.runner.num_attribs - len(self.runner.protected_attribs),)
        )
        self.biasd = 0
        self.error_set = set()
        self.total = 0
        self.total_set = set()
        self.dup_error = 0
        self.dict = {}
        self.mean = 0
        self.covariance = []
        self.threshold = 0
        self.median = 0
        
        self.reward_biasd = 1.5
        self.reward_punished = -0.015

    def step(self, action):
        reward = 0
            
        index = self.runner.action_table[action][0]
        change = self.runner.action_table[action][1]
   
        range1 = self.runner.input_bounds[index]
        
        # calculate st_act
        to_check = tuple(self.current_sample)

        if to_check in self.dict.keys():
            self.dict[to_check][0][action] += 1
        else:
            self.dict[to_check] = np.zeros([1, len(self.runner.action_table)] , dtype=np.int32)
            self.dict[to_check][0][action] = 1
        
        # Insert protected attributes at their original locations using their low bounds
        for idx, val in zip(self.runner.protected_attribs, self.runner.low_bound):
            self.current_sample.insert(idx, val)
             
        if self.current_sample[index] == range1[0] or self.current_sample[index] == range1[1]:
            if self.current_sample[index] == range1[0]:
                change = 1
                self.current_sample[index] += 1
            else:
                change = -1
                self.current_sample[index] -= 1  
        else:
            self.current_sample[index] += change
            
        # Get current test input
        to_check_discriminate_sample = copy.deepcopy(self.current_sample) 
        self.current_sample = [val for idx, val in enumerate(self.current_sample) if idx not in self.runner.protected_attribs]
        
        # calculate another st_act
        to_check_second = tuple(self.current_sample)
        if to_check_second in self.dict.keys():
            if action % 2 == 0:
                self.dict[to_check_second][0][action + 1] += 1
            else:
                self.dict[to_check_second][0][action - 1] += 1
        else:
            self.dict[to_check_second] = np.zeros([1, len(self.runner.action_table)] , dtype=np.int32)
            if action % 2 == 0:
                self.dict[to_check_second][0][action + 1] += 1
            else:
                self.dict[to_check_second][0][action - 1] += 1
                
        terminated = False
        x_ = copy.deepcopy(self.current_sample)
        if tuple(x_) in self.total_set:
            if tuple(x_) in self.error_set:
                self.dup_error += 1
        else:
            self.total_set.add(tuple(x_))
            
            # Record checked inputs
            self.runner.tot_inputs.add(tuple(to_check_discriminate_sample))
            
            is_discriminate = self.runner.is_discriminate(to_check_discriminate_sample)
            if is_discriminate:
                reward = self.reward_biasd
                self.biasd += 1
                self.error_set.add(tuple(x_))
                self.runner.disc_inputs.add(tuple(to_check_discriminate_sample))
                self.runner.set_time_to_1000_disc()
                
        self.observation_space = np.array(self.current_sample, dtype=np.float32)
        self.counts += 1
        self.total += 1
        self.runner.total_generated += 1
        
        if reward == 0:
            reward = self.reward_punished
        else:
            mahalanobis_dist = distance.mahalanobis(self.current_sample, self.mean, self.covariance)
            if mahalanobis_dist <= self.median:
                reward = self.reward_biasd 
            elif mahalanobis_dist > self.median and mahalanobis_dist <= self.threshold:
                reward = self.reward_biasd / (math.sqrt(mahalanobis_dist / self.median))
            else:
                reward = self.reward_biasd /  (mahalanobis_dist / self.median)
                 
        truncated = False
        if self.counts == self.episode_end:
            truncated = True
            
        return self.observation_space, reward, terminated, truncated, self.dict

    def reset(self, options):
        self.current_sample = self.runner.X[options["seed"]].tolist()
        self.current_sample = [val for idx, val in enumerate(self.current_sample) if idx not in self.runner.protected_attribs]
        self.mean = options["mean"]
        self.covariance = options["covariance"]
        self.threshold = options["threshold"]
        self.median = options["median"]
        self.observation_space = np.array(self.current_sample, dtype=np.float32)
        self.counts = 0
        return self.observation_space, {}


class MAEFT:
    def __init__(self, config, model, classifier_name, sensitive_params):
        self.approach_name = "MAEFT"
        self.config = config
        self.threshold = 0
        
        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.input_bounds = np.array(self.config.input_bounds)
        self.sensitive_params = sensitive_params
        self.model = dnn_model_wrapper(model=model)
        self.classifier_name = classifier_name

        self.protected_attribs = [sens_param - 1 for sens_param in sensitive_params]
        self.num_attribs = len(config.input_bounds)
        self.non_protected_attribs = [i for i in range(len(config.input_bounds)) if i not in self.protected_attribs]
        self.protected_domains = [list(range(int(self.input_bounds[i][0]), int(self.input_bounds[i][1]) + 1)) for i in
                                  self.protected_attribs]

        self.start_time = time.time()
        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0
        self.log_interval = 300
        self.initial_log_interval = self.log_interval

        # Sensitive attribute bounds
        self.low_bound = [int(self.input_bounds[p][0]) for p in self.protected_attribs]
        self.high_bound = [int(self.input_bounds[p][1] + 1) for p in self.protected_attribs]

        # Setup action table
        self.action_table = []
        for i in range(self.num_attribs):
            if i not in self.protected_attribs:
                self.action_table.append([i, 1])
                self.action_table.append([i, -1])

        # Define n_c continuous mapping matching dataset
        dataset_n_c = {
            "census": [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            "bank": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            "meps": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            "credit": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            "compas": [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        }
        dataset_name = self.config.dataset_name
        if dataset_name in dataset_n_c:
            self.n_c = dataset_n_c[dataset_name]
        else:
            self.n_c = [0] * self.num_attribs

        # Load data to instance variable X
        data_fn = get_data(dataset_name)
        self.X, self.Y, self.input_shape, self.nb_classes = data_fn()

    def similar_set_(self, X):
        X = np.array(X, dtype=int)
        N = len(X)
        all_combs = np.array(list(itertools.product(*self.protected_domains)))
        M = len(all_combs)
        batch_inputs = np.repeat(X, M, axis=0)
        for j, attr_idx in enumerate(self.protected_attribs):
            batch_inputs[:, attr_idx] = np.tile(all_combs[:, j], N)
        return batch_inputs, all_combs, M

    def is_discriminate(self, x):
        variants, _, _ = self.similar_set_([x])
        preds = self.model.predict(variants)
        self.inference_count += 1
        return len(np.unique(preds)) != 1

    def cluster(self):
        # Identify discriminatory seeds from the training set
        dis_x = []
        dis_kmeans = []
        for i in range(len(self.X)):
            temp = self.X[i].tolist()
            if self.is_discriminate(temp):
                dis_x.append(i)
                temp_removed = [val for idx, val in enumerate(temp) if idx not in self.protected_attribs]
                dis_kmeans.append(temp_removed)

        seed = []
        if len(dis_kmeans) <= 10:
            seed = dis_x
        else:
            max_clusters = 10
            best_num_clusters = 0
            best_silhouette = -1
            for i in range(2, min(max_clusters + 1, len(dis_kmeans))):
                kmeans = KMeans(n_clusters=i, init="k-means++", n_init='auto', random_state=2024).fit(dis_kmeans)
                labels = kmeans.labels_
                silhouette = silhouette_score(dis_kmeans, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_num_clusters = i
            to_split = math.ceil(12 / best_num_clusters)
            kmeans = KMeans(n_clusters=best_num_clusters, init="k-means++", n_init='auto', random_state=2024)
            kmeans.fit_predict(dis_kmeans)
            for i in range(best_num_clusters):
                X_ = []
                number_x = []
                for j in range(len(kmeans.labels_)):
                    if kmeans.labels_[j] == i:
                        X_.append(dis_kmeans[j])
                        number_x.append(dis_x[j])
                if len(X_) <= to_split:
                    for this_seed in number_x:
                        seed.append(this_seed)
                else:
                    temp = kmeans.transform(X_)[:, i]
                    ind = np.argsort(temp)
                    ind = ind.tolist()
                    for this_part in range(to_split):
                        seed.append(number_x[ind[int(this_part * len(ind) / to_split)]])
        seed = list(set(seed))

        if len(seed) == 0:
            seed = list(np.random.choice(len(self.X), size=min(12, len(self.X)), replace=False))

        # calculate mean and covariance of training data using MinCovDet
        train_data = []
        for i in self.X:
            i = i.tolist()
            temp = [val for idx, val in enumerate(i) if idx not in self.protected_attribs]
            train_data.append(temp)
        train_data = np.array(train_data)
        mcd = MinCovDet(random_state=2024, support_fraction=0.9) 
        mcd.fit(train_data)
        mean = mcd.location_
        covariance = mcd.covariance_
        num_non_protected = len(train_data[0])
        rank = np.linalg.matrix_rank(covariance)
        if rank == num_non_protected:
            this_inv = np.linalg.inv(covariance)
        else:
            this_inv = np.linalg.pinv(covariance)

        # calculate MD of between each sample in training data and training data
        distances = []
        for i in train_data:
            mahalanobis_dist = distance.mahalanobis(i, mean, this_inv)
            distances.append(mahalanobis_dist)
        distances.sort()
        median = distances[int(len(self.X)/2)]
        this_max = distances[-1]
        this_threshold = this_max
        return seed, mean, this_inv, this_threshold, median

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)
        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):
        print("Starting MAEFT Seed Clustering...")
        select_seed, mean, covariance, this_threshold, median = self.cluster()
        print(f"Clustering finished. Found {len(select_seed)} seeds.")

        env = MyEnv(runner=self)
        state_size = env.observation_space.shape[0] 
        action_size = env.action_space.n 
        state_action = {}
        
        this_dict = {
            'mean': mean,
            'covariance': covariance,
            'threshold': this_threshold,
            'median': median
        }
        
        episodes = 2002
        steps = 500
        seed_num = len(select_seed)
        
        episode_count = 0
        break_all = False
        
        for i in range(seed_num):
            if break_all:
                break
                
            this_dict['seed'] = select_seed[i]
            agent = DoubleDuelingDQN(state_size, action_size)
            
            if i != seed_num - 1:
                this_episodes = int(episodes / seed_num)
            else:
                this_episodes = episodes - (seed_num - 1) * int(episodes / seed_num)
                
            for i_episode in range(this_episodes):
                if break_all:
                    break
                    
                observation, _ = env.reset(options=this_dict)
                score = 0
                
                for t in range(steps):
                    state = observation
                    action = agent.get_action(state, state_action)
                    observation, reward, terminated, truncated, state_action = env.step(action)
                    state_next = observation
                    agent.stock_experience(state, action, reward, state_next)
                    agent.train(state_action)
                    score += reward
                    
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time >= max_allowed_time or self.total_generated >= max_samples:
                        break_all = True
                        break
                        
                episode_count += 1
                self.update_cumulative_efficiency(episode_count)
                
                use_time = time.time() - self.start_time
                if use_time >= self.log_interval:
                    self.log_interval += self.initial_log_interval
                    self.report(elapsed_time=use_time, is_log=True)
                    
        elapsed_time = time.time() - self.start_time
        self.report(elapsed_time=elapsed_time, is_log=False)

    def update_cumulative_efficiency(self, episode):
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, episode, total_inputs, total_disc])

    def report(self, elapsed_time, is_log: bool):
        save_path = 'results'
        additional_data = {'inference_count': self.inference_count}
        sens_names = [self.config.sens_name[sens_param] for sens_param in self.sensitive_params]
        generate_report(
            approach_name=f'{self.approach_name}',
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
            save_path=save_path,
            save_data=True,
            **additional_data,
        )


if __name__ == '__main__':
    config, sensitive_names, sensitive_params, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {",".join(map(str, sorted(sensitive_names)))}')
    print('')

    if classifier_name == 'dnn':
        import tensorflow as tf
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.keras'
        model = tf.keras.models.load_model(classifier_path)
    elif classifier_name == 'ftt':
        from src.utils.tabular_transformers import load_ftt_model
        model = load_ftt_model(dataset_name=config.dataset_name)
    else:
        classifier_path = f'models/{config.dataset_name}/{classifier_name.lower()}.pkl'
        model = joblib.load(classifier_path)

    maeft = MAEFT(
        config=config,
        model=model,
        sensitive_params=sensitive_params,
        classifier_name=classifier_name
    )

    maeft.run(max_allowed_time=max_allowed_time)