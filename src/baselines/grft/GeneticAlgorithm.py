import sys

sys.path.append('.')
import numpy as np
import copy
import random
import time


def create_image_indvs(img, num):
    indivs = []
    indivs.append(img)
    for i in range(num - 1):
        indivs.append(img + 1)
    return np.array(indivs)


class Population():

    def __init__(self, individuals,
                 mutation_function,
                 fitness_compute_function,
                 mutate_func2, num_attribs,
                 subtotal,
                 first_attack,
                 seed,
                 max_iteration,
                 tour_size=3, cross_rate=0.5, mutate_rate=0.5, max_trials=50, max_time=30):
        self.ground_truth = 0
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.seed = seed
        self.individuals = individuals  # a list of individuals, current is numpy
        self.tournament_size = tour_size
        self.fitness = None  # a list of fitness values
        self.pop_size = len(self.individuals)
        self.subtotal = subtotal

        self.firstattack = first_attack

        self.muation_func2 = mutate_func2
        self.mutation_func = mutation_function
        self.fitness_fuc = fitness_compute_function
        self.order = []
        self.best_fitness = -1000
        self.success = 0
        self.discriminatory = np.empty(shape=(0, num_attribs))
        self.round = 0

        # self.total_bugs = 0
        # self.total_diversity = set()

        self.first_time_used = max_time
        self.first_iteration_used = max_iteration
        self.first_ids = 100
        # for i in range(max_trials):
        start_time = time.time()
        i = 0
        while True:

            if i >= max_iteration:
                self.round = i
                break
            if time.time() - start_time > max_time:
                self.round = i
                break
            i += 1
            results = self.evolvePopulation()
            if results is None:
                xx = ""
                # print("   Total generation: %d, best fitness:%.9f"%(i, self.best_fitness))
            else:
                self.discriminatory = np.append(self.discriminatory, results[-2], axis=0)
                # print(self.discriminatory)
                self.round = i
                # self.save_function(results,i)
                self.success = 1
                self.first_ids = i
                # results1[new_index_result], results2[new_index_result], array[new_index_result], r_indexes

                if self.first_time_used == max_time:
                    self.first_time_used = time.time() - start_time
                    self.first_iteration_used = i
                if self.firstattack == 1:
                    break
                else:
                    self.individuals = np.array(self.individuals)
                    self.individuals = self.muation_func2(self.individuals)

    def crossover(self, ind1, ind2):
        shape = ind1.shape
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        new_ind = np.copy(ind1)

        for i in range(len(ind1)):
            if random.uniform(0, 1) < self.cross_rate:
                new_ind[i] = ind1[i]
            else:
                new_ind[i] = ind2[i]
        return np.reshape(new_ind, shape)

    def evolvePopulation(self):

        results = self.fitness_fuc(self.individuals)

        objects = results[-2]  # new_index_result
        self.fitness = results[-1]  # fitness

        if len(objects) > 0:
            return results[:-1]

        """
            sorted_fitness_indexes: the ordered indexes based on fitness value
            sorted_fitness_indexes[0] is the index of individual with the best fitness
        """
        # print(self.fitness)
        sorted_fitness_indexes = sorted(range(0, len(self.fitness)), key=lambda k: self.fitness[k], reverse=True)

        # sorted_fitness_indexes1 = sorted(range(0,self.subtotal), key=lambda k: self.fitness[k], reverse=True)
        # sorted_fitness_indexes2 = sorted(range(self.subtotal, self.subtotal*2), key=lambda k: self.fitness[k], reverse=True)
        # sorted_fitness_indexes3 = sorted(range(self.subtotal*2, len(self.fitness)), key=lambda k: self.fitness[k], reverse=True)
        """
            tournaments: randomly select a tournament from the individuals and get the indv with best fittness
            Instead of select from individuals , we select from the sorted indexes (i.e., sorted_fitness_indexes) randomly.
            sorted_fitness_indexes[order_seq1[0]] is the index of indivitual with best fitness in the selected tournament.
        """

        new_indvs = []
        sorted_fitnesses = [sorted_fitness_indexes]
        # ranges = [(0,self.pop_size)]
        ranges = [(0, self.pop_size)]
        tour_ranges = [(0, self.subtotal)]

        for j in range(len(sorted_fitnesses)):
            sorted_fitness_indexes = sorted_fitnesses[j]
            best_index = sorted_fitness_indexes[0]
            (start, end) = ranges[j]
            (tour_start, tour_end) = tour_ranges[j]
            for i in range(start, end):
                item = self.individuals[i]
                if i == best_index:  # keep best
                    new_indvs.append(item)
                else:
                    # print(tour_start,tour_end,'-------')
                    order_seq1 = np.sort(
                        np.random.choice(np.arange(tour_start, tour_end), self.tournament_size, replace=False))
                    order_seq2 = np.sort(
                        np.random.choice(np.arange(tour_start, tour_end), self.tournament_size, replace=False))
                    first_individual = self.individuals[sorted_fitness_indexes[order_seq1[0]]]
                    second_individual = self.individuals[
                        sorted_fitness_indexes[order_seq2[0] if order_seq2[0] != order_seq1[0] else order_seq2[1]]]
                    # Cross over
                    ind = self.crossover(first_individual, second_individual)
                    if random.uniform(0, 1) < self.mutate_rate:
                        ind = self.mutation_func(ind)
                    new_indvs.append(ind)

        self.individuals = new_indvs
        self.best_fitness = self.fitness[sorted_fitness_indexes[0]]
        return None
