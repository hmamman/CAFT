import random
import numpy as np
import copy
import itertools

inf = float('inf')
MAX = 2 ** 63 - 1
MIN = - MAX - 1


class Interval:

    def __init__(self, L_num, R_num, L_closed, R_closed):
        self.L_num = L_num
        self.R_num = R_num
        self.L_closed = bool(L_closed)
        self.R_closed = bool(R_closed)
        self.have_value = True
        self.check_value()

    def __str__(self):
        res = '[' if self.L_closed else '('
        res += str(self.L_num) + ',' + str(self.R_num)
        res += ']' if self.R_closed else ')'
        return res

    def __add__(self, other):
        #        if (self.L_num > other.R_num) or (self.R_num < other.L_num):
        #            return False
        if self.L_num > other.L_num:
            L = self.L_num
            L_closed = self.L_closed
        elif self.L_num == other.L_num:
            L = self.L_num
            L_closed = self.L_closed and other.L_closed
        else:
            L = other.L_num
            L_closed = other.L_closed
        if self.R_num < other.R_num:
            R = self.R_num
            R_closed = self.R_closed
        elif self.R_num == other.R_num:
            R = self.R_num
            R_closed = self.R_closed and other.R_closed
        else:
            R = other.R_num
            R_closed = other.R_closed
        return Interval(L, R, L_closed, R_closed)

    def check_value(self):
        if (self.L_num > self.R_num) or ((self.L_num == self.R_num) and (not (self.L_closed and self.R_closed))):
            self.have_value = False

    def check_num_of_int(self, n=1):
        L = self.L_num if self.L_closed else self.L_num + 1
        R = self.R_num if self.R_closed else self.R_num - 1
        if n <= (R - L + 1):
            return True
        else:
            return False

    def have_at_least_one_value(self):
        return self.check_num_of_int(n=1)

    def uniform_sample(self):
        if not self.have_value:
            print(" %s doesn't have value, please check" % (str(self)))
            return False
        L = self.L_num if self.L_closed else self.L_num + 1
        R = self.R_num if self.R_closed else self.R_num - 1
        if self.L_num == -inf:
            L = MIN
        if self.R_num == inf:
            R = MAX
        # [L,R]
        return int(random.randint(L, R))


class IntervalPool:
    def __init__(self):
        self.map_key_interval = dict()  # key -> Interval()
        self.map_key_if_valid = dict()  # key -> True/False
        self.add_res = dict()  # key+key -> key

    def create(self, interval):
        key = str(interval)
        if key not in self.map_key_interval:
            self.map_key_interval[key] = interval
        return key

    def have_at_least_one_value(self, key):
        if key not in self.map_key_interval:
            print(f"error: found no saved interval")

        if key in self.map_key_if_valid:
            return self.map_key_if_valid[key]
        else:
            if_valid = self.map_key_interval[key].have_at_least_one_value()
            self.map_key_if_valid[key] = if_valid
            return if_valid

    def add(self, key1, key2):
        # key1 and key2 must in self.map_key_interval
        if (key1 not in self.map_key_interval) or (key2 not in self.map_key_interval):
            print(f"error: found no saved interval")

        key = key1+key2
        if key in self.add_res:
            return self.add_res[key]
        else:
            new_interval = self.map_key_interval[key1] + self.map_key_interval[key2]
            new_key = str(new_interval)
            self.map_key_interval[new_key] = new_interval
            self.add_res[key] = new_key
            return new_key

    def uniform_sample(self, key):
        return self.map_key_interval[key].uniform_sample()


class Path:

    def __init__(self, no_attr, data_range, IntervalP):
        # [L, R, l_closed, r_closed] * no_attr
        self.path_list = [IntervalP.create(Interval(data_range[attr][0], data_range[attr][1], 1, 1)) for attr in range(no_attr)]
        self.predict_value = None
        self.flip_list_for_protected_attr = list()
        self.IntervalP = IntervalP

    def __str__(self):
        return ','.join([str(item) for item in self.path_list])

    def flip_for_protected_attr(self):
        return self.flip_list_for_protected_attr

    def add_flip_for_protected_attr(self, path2):
        self.flip_list_for_protected_attr.append(path2)

    def add_node(self, node_index, threshold, direction):
        if direction:  # <=
            interval_new = self.IntervalP.create(Interval(MIN, threshold, 0, 1))
        else:  # >
            interval_new = self.IntervalP.create(Interval(threshold, MAX, 0, 0))
        self.path_list[node_index] = self.IntervalP.add(self.path_list[node_index], interval_new)

    def add_predict_value(self, res):
        self.predict_value = res


class PathSearcher:

    def __init__(self, DT, CuT, protected_list_no, data_range, protected_value_comb, IntervalP=None):
        self.disc_data = list()  # discriminatory data
        self.test_data = list()  # test data
        self.train_data = list()  # train data
        self.disc_path_pair = list()
        self.CuT = CuT
        self.protected_list_no = protected_list_no
        self.no_prot = len(protected_list_no)
        self.protected_value_comb = protected_value_comb
        self.IntervalP = IntervalP

        self.no_attr = DT.tree_.n_features
        self.paths = list()
        self.get_DT_paths(DT, data_range)
        self.inference_count = 0

    def get_DT_paths(self, DT, data_range):
        # generate self.paths   and self.intervals
        # init the Path class
        tree_ = DT.tree_
        feature = tree_.feature
        no_attr = DT.tree_.n_features

        def recurse(node, depth, path):

            if feature[node] != -2:  # find a node
                index = feature[node]
                threshold = int(tree_.threshold[node])

                # create Interval of this node's left children
                path.append([index, threshold, True])
                if tree_.children_left[node] != -1:
                    path_list_left = recurse(tree_.children_left[node], depth + 1, path)
                else:
                    path_list_left = []

                # create Interval of this node's right children
                path.append([index, threshold, False])
                if tree_.children_right[node] != -1:
                    path_list_right = recurse(tree_.children_right[node], depth + 1, path)
                else:
                    path_list_right = []

                if len(path) != 0:
                    path.pop()  # pop the path which is added by the parent recursion

                if feature[node] in self.protected_list_no:
                    for path1, path2 in itertools.product(path_list_left, path_list_right):
                        path1.add_flip_for_protected_attr(path2)

                return path_list_left + path_list_right

            else:  # find leaf node
                pre_res = np.argmax(tree_.value[node][0])
                new_path = Path(no_attr, data_range, self.IntervalP)
                for p in path:
                    new_path.add_node(node_index=p[0], threshold=p[1], direction=p[2])
                new_path.add_predict_value(pre_res)
                self.paths.append(new_path)

                if len(path) != 0:
                    path.pop()  # pop the path which is added by the parent recursion

                return [new_path]

        recurse(0, 1, [])  # Start from root (node 0) with depth 1

    def sample_from_path_pair(self, n, pair):
        space = pair["space"]
        p1 = pair["path1"]
        p2 = pair["path2"]
        res = list()
        for i in range(n):
            sample1 = list()
            sample2 = list()
            for attr in range(self.no_attr):
                if attr in self.protected_list_no:
                    value1 = self.IntervalP.uniform_sample(space[attr][0])
                    value2 = self.IntervalP.uniform_sample(space[attr][1])
                    sample1.append(value1)
                    sample2.append(value2)
                else:
                    value = self.IntervalP.uniform_sample(space[attr][0])
                    sample1.append(value)
                    sample2.append(value)
            sample1.append(p1.predict_value)
            sample2.append(p2.predict_value)
            res.append(sample1)
            res.append(sample2)

        return res

    def check_if_path_pair_includes_disc(self, p1: "Path", p2: "Path", need_check_protected_attr=True):
        space = [list() for _ in range(self.no_attr)]
        # check for label
        if p1.predict_value == p2.predict_value:
            return False, space
        else:
            # check for protected attributes
            have_p_attr_meets_condition = False
            for p_attr in self.protected_list_no:
                interval1 = p1.path_list[p_attr]
                interval2 = p2.path_list[p_attr]
                space[p_attr].append(interval1)
                space[p_attr].append(interval2)

                if need_check_protected_attr:
                    interval_new = self.IntervalP.add(interval1, interval2)
                    if not self.IntervalP.have_at_least_one_value(interval_new):
                        # protected attribute doesn't have the same value
                        have_p_attr_meets_condition = True
                    else:
                        pass
                    if not have_p_attr_meets_condition:
                        return False, space

            # check for non-protected attributes
            for attr in range(self.no_attr):
                if attr not in self.protected_list_no:
                    interval1 = p1.path_list[attr]
                    interval2 = p2.path_list[attr]
                    interval_new = self.IntervalP.add(interval1, interval2)
                    space[attr].append(interval_new)
                    if self.IntervalP.have_at_least_one_value(interval_new):
                        pass
                    else:
                        return False, space
            return True, space

    def detect_disc_from_path_pair(self, pair, max_train_data_each_path, max_sample_each_path, check_type):
        num_train_data = 0
        num_sample = 0
        num_disc = 0

        # In the paper, check_type == "themis".
        if check_type == "naive":
            while num_train_data < max_train_data_each_path and num_sample < max_sample_each_path:
                # Generate a test case from subspace (path pair) randomly.
                num_sample += 1
                samples = self.sample_from_path_pair(1, pair)
                X = [item[:-1] for item in samples]
                Y = [item[-1] for item in samples]
                real_Y = self.CuT.predict(X)
                self.inference_count += 1
                i = 0
                self.test_data.append(X[i] + [Y[i]])
                self.test_data.append(X[i + 1] + [Y[i + 1]])

                # Check if the test case is discriminatory instances in CuT,
                # and then update self.disc_data and self.train_data (failing data).
                equal1 = True if Y[i] == real_Y[i] else False
                equal2 = True if Y[i + 1] == real_Y[i + 1] else False
                if not equal1:
                    self.train_data.append(X[i] + [real_Y[i]])
                    # self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                    num_train_data += 1
                if not equal2:
                    # self.train_data.append(X[i] + [real_Y[i]])
                    self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                    num_train_data += 1
                if equal1 and equal2:
                    self.disc_data.append(X[i] + [Y[i]])
                    self.disc_data.append(X[i + 1] + [Y[i + 1]])
                    num_disc += 1

        elif check_type == "themis":
            while num_train_data < max_train_data_each_path and num_sample < max_sample_each_path:
                # Generate a test case from subspace (path pair) randomly.
                num_sample += 1
                samples = self.sample_from_path_pair(1, pair)
                X = [item[:-1] for item in samples]
                Y = [item[-1] for item in samples]
                real_Y = self.CuT.predict(X)
                self.inference_count += 1
                i = 0
                self.test_data.append(X[i] + [Y[i]])
                self.test_data.append(X[i + 1] + [Y[i + 1]])

                # Check if the test case is discriminatory instances in CuT,
                # and then update self.disc_data and self.train_data (failing data).
                equal1 = True if Y[i] == real_Y[i] else False
                equal2 = True if Y[i + 1] == real_Y[i + 1] else False
                if not equal1:
                    self.train_data.append(X[i] + [real_Y[i]])
                    # self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                    num_train_data += 1
                if not equal2:
                    # self.train_data.append(X[i] + [real_Y[i]])
                    self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                    num_train_data += 1
                found_disc = False
                if real_Y[i] != real_Y[i + 1]:
                    self.disc_data.append(X[i] + [Y[i]])
                    self.disc_data.append(X[i + 1] + [Y[i + 1]])
                    num_disc += 1
                    found_disc = True

                # Here, themis examines all possible values of the protected attributes
                # to identify potential discriminatory instances in CuT.
                if (not found_disc) and (len(self.protected_value_comb) > 2):
                    X2 = copy.deepcopy(X[0])
                    comb_to_be_removed = list()
                    comb_to_be_removed.append(tuple(X[0][i] for i in self.protected_list_no))
                    comb_to_be_removed.append(tuple(X[1][i] for i in self.protected_list_no))
                    comb_removed_same = [item for item in self.protected_value_comb if
                                         item not in comb_to_be_removed]
                    random.shuffle(comb_removed_same)
                    for combination in comb_removed_same:
                        for i in range(self.no_prot):
                            X2[self.protected_list_no[i]] = combination[i]

                        real_Y2 = self.CuT.predict([X2])[0]
                        self.inference_count += 1
                        if real_Y2 != real_Y[0]:
                            self.disc_data.append(X[0] + [real_Y[0]])
                            self.disc_data.append(X2 + [real_Y2])
                            num_disc += 1
                            break

        else:
            print(f"no check type called {check_type} when detecting discriminatory instances in CuT from path pair.")

    def sample(self, dt_search_mode="random+flip", check_type="themis",
               MaxTry=10000, MaxDiscPathPair=100, max_train_data_each_path=10,
               max_sample_each_path=100):
        # The algorithm of PathSample (Algorithm 3 in paper)
        # Search path pairs including discriminatory instances in DT.
        # In the paper, dt_search_mode == "random+flip".
        disc_path_pair = list()
        if dt_search_mode == "random+flip":
            # 1. Select a path path1 from all paths at random.
            # 2. Check all flips of path1 to check if path1 and path2 include discriminatory instances in DT.
            num_found_path_pair = 0
            for path1 in random.sample(self.paths, min(len(self.paths), MaxTry)):
                if (MaxDiscPathPair is not None) and (num_found_path_pair >= MaxDiscPathPair):
                    break
                for path2 in path1.flip_for_protected_attr():
                    include_disc, space = self.check_if_path_pair_includes_disc(path1, path2, need_check_protected_attr=True)
                    if include_disc:
                        num_found_path_pair += 1
                        disc_path_pair.append({"path1": path1, "path2": path2, "space": space})
        elif dt_search_mode == "all":
            # Search all combinations of path pairs in DT
            for path1, path2 in itertools.combinations(self.paths, 2):
                include_disc, space = self.check_if_path_pair_includes_disc(path1, path2, need_check_protected_attr=True)
                if include_disc:
                    disc_path_pair.append({"path1": path1, "path2": path2, "space": space})
            disc_path_pair = random.sample(disc_path_pair, min(MaxDiscPathPair, len(disc_path_pair)))
        else:
            print(f"no mode called {dt_search_mode} when searching path pairs including discriminatory instances in DT")
        self.disc_path_pair = disc_path_pair

        # The algorithm of RandomSearch (Algorithm 4 in paper)
        # Conduct random search over subspaces to detect discriminatory instances in CuT.
        # Note that here a path pair implies a subspace.
        for pair in disc_path_pair:
            self.detect_disc_from_path_pair(pair, max_train_data_each_path, max_sample_each_path, check_type)

        satFlag = True if len(self.test_data) != 0 else False
        return satFlag

    def get_test_data(self):
        return self.test_data

    def get_disc_data(self):
        return self.disc_data

    def get_train_data(self):
        return self.train_data

    def get_inference_count(self):
        return self.inference_count
