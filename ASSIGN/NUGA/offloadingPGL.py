import math
import random

import numpy as np
import matplotlib.pyplot as plt

from env import Adapter, Tasks, Vehicle

K = 3
LOCATION_DOMAIN = (1e-10, 3)
VELOCITY_DOMAIN = (-0.25, 0.25)
# VELOCITY_DOMAIN = (-11, 11)
WEIGHT_DOMAIN = (0.1, 0.9)
C1 = 2
C2 = 2
T = 250
SWARM_SIZE = 200



class Particle(object):
    def __init__(self, adapter, decision):
        self.adapter = adapter
        self.decision = decision
        self.p_loc = np.array(np.random.uniform(LOCATION_DOMAIN[0], LOCATION_DOMAIN[1], self.adapter.N * self.adapter.M)
                              .reshape((self.adapter.N, self.adapter.M)))
        self.p_vel = np.array(np.random.uniform(VELOCITY_DOMAIN[0], VELOCITY_DOMAIN[1], self.adapter.N * self.adapter.M)
                              .reshape((self.adapter.N, self.adapter.M)))
        self.p_best = np.array(self.p_loc)
        self.p_best_fit = fitness(self.adapter, self.decision, self.p_loc)
        self.N = 20
        self.M = 20
        self.T = 100   #初始温度
        self.T_MIN = 1  # 最小温度
        self.DELTA = 0.9  # 温度下降速率
        self.DISTURBANCE = (-0.001, 0.001)  # 扰动
        self.k = 0.95
        self.p_best_new = np.zeros((self.N, self.M))

    def update_p_best(self):
        temp = fitness(self.adapter, self.decision, self.p_loc)
        if temp > self.p_best_fit:
            self.p_best_fit = temp
            self.p_best = np.array(self.p_loc)
        else:
            self.anneal(self.p_best)

    def anneal(self, p_best):
        t = self.T

        cnt = 0
        while t > self.T_MIN:
            y = fitness(self.adapter, self.decision, self.p_best)
            for i in range(self.N):
                for j in range(self.M):
                    self.p_best_new[i][j] = p_best[i][j] + random.uniform(self.DISTURBANCE[0], self.DISTURBANCE[1]) * t
                    if self.p_best_new[i][j] > LOCATION_DOMAIN[1]:
                       self.p_best_new[i][j] = LOCATION_DOMAIN[1]
                    elif self.p_best_new[i][j] < LOCATION_DOMAIN[0]:
                       self.p_best_new[i][j] = LOCATION_DOMAIN[0]
            y_new = fitness(self.adapter, self.decision, self.p_best_new)
            # print('xxxxx', y)
            # print(y_new)
            if y_new < y:
                X = self.p_best_new
                return X
            else:
                p = math.exp((y_new - y) / t)
                # print('p等于', p)
                if random.random() > p:
                    X = self.p_best_new

            t = self.DELTA * t
            cnt += 1


class Swarm(object):
    def __init__(self, adapter, decision):
        self.adapter = adapter
        self.swarm = [Particle(self.adapter, decision) for _ in range(SWARM_SIZE)]
        self.g_best = np.array(self.swarm[0].p_best)
        self.g_best_fit = -1e10
        self.update_g_best()
        self.decision = decision


    def update_g_best(self):
        best_idx = 0
        for i in range(SWARM_SIZE):
            if self.swarm[i].p_best_fit > self.g_best_fit:
                self.g_best_fit = self.swarm[i].p_best_fit
                best_idx = i

        self.g_best = np.array(self.swarm[best_idx].p_best)

    def PSO(self):
        w = WEIGHT_DOMAIN[1]
        x, y = [], []
        for t in range(T):
            for particle in self.swarm:
                R1, R2 = np.random.random(), np.random.random()
                particle.p_vel = np.clip(w * particle.p_vel + \
                                         C1 * R1 * (particle.p_best - particle.p_loc) + \
                                         C2 * R2 * (self.g_best - particle.p_loc), VELOCITY_DOMAIN[0],
                                         VELOCITY_DOMAIN[1])
                particle.p_loc = particle.p_loc + particle.p_vel
                particle.p_loc = np.clip(particle.p_loc, LOCATION_DOMAIN[0], adapter.F)
                particle.update_p_best()
            w = WEIGHT_DOMAIN[1] - (WEIGHT_DOMAIN[1] - WEIGHT_DOMAIN[0]) * (t / T)
            self.update_g_best()
            x.append(t)
            y.append(-self.g_best_fit)
        # print(self.decision)

        # print(self.g_best)
        print(self.adapter.N, self.adapter.M)
        print(self.g_best_fit, obj_func2(adapter, self.decision, self.g_best), np.mean([p.p_best_fit for p in self.swarm]))
        plt.plot(x, y)
        plt.show()
        return self.g_best, self.g_best_fit

class Individual(object):
    def __init__(self, adapter):
        self.individual = np.zeros(shape=(adapter.N, adapter.M))
        self.individual[range(adapter.N), np.random.randint(0, adapter.M, size=adapter.N)] = 1

        # print('初始decision', self.individual)

    def get_fitness(self):
        return fitness(adapter, self.individual, get_allocation(adapter, self.individual))

class Population(object):
    def __init__(self, adapter, p_size=100):
        self.adapter = adapter
        self.p_size = p_size
        self.pop = [Individual(adapter) for i in range(p_size)]
        self.best_individual = np.zeros(shape=(adapter.N, adapter.M))

    def get_info(self):
        sum, max, avg = 0, -1e10, 0
        # for i in new_pop():

        for individual in self.pop:
            val = individual.get_fitness()
            sum += val
            if max < val:
               max = val
        return max, sum / self.p_size

    def get_best_individual(self, pop):
        max = -1e10
        for i, individual in enumerate(self.pop):
            val = individual.get_fitness()
            if max < val:
               max = val
               self.best_individual = self.pop[i].individual
        # print(self.best_individual)

    def select(self):
        new_pop = Population(self.adapter, self.p_size)
        for i in range(self.p_size):
            idx1, idx2 = np.random.randint(0, self.p_size), np.random.randint(0, self.p_size)
            new_pop.pop[i].individual = np.array(self.pop[idx1].individual) \
                if fitness(self.adapter, self.pop[idx1].individual,
                           get_allocation(self.adapter, self.pop[idx1].individual)) > \
                   fitness(self.adapter, self.pop[idx2].individual,
                           get_allocation(self.adapter, self.pop[idx2].individual)) \
                else np.array(self.pop[idx2].individual)
        self.pop = new_pop.pop

    def crossover(self, p=1):
        for i in range(0, self.p_size, 2):
            for n in range(self.adapter.N):
                if np.random.random() < p:
                    self.pop[i].individual[n], self.pop[i + 1].individual[n] = \
                        self.pop[i + 1].individual[n], self.pop[i].individual[n]

    def mutate(self, p=0.1):
        for i in range(self.p_size):
            if np.random.random() < p:
                for n in range(self.adapter.N):
                    if np.random.random() < p:
                        self.pop[i].individual[n] = np.zeros(self.adapter.M)
                        self.pop[i].individual[n, np.random.randint(0, self.adapter.M)] = 1

    def GA(self, T=200):
        x, y1, y2 = [], [], []
        best = -1e10
        for t in range(T):
            self.select()
            x.append(t)
            yy1, yy2 = self.get_info()
            if best < yy1:
                best = yy1
            # y1.append(yy1)
            # y2.append(yy2)
            self.crossover(p=0.5)
            self.mutate(p=0.1)
        self.get_best_individual(self.pop)

        # print(y2[T-1], best)

        return self.best_individual
        # plt.plot(x, y1)
        # plt.plot(x, y2)
        # plt.show()

# def obj_func(adapter, decision, allocation):
#     return (decision * (adapter.C / allocation * 1000 + adapter.Z)).sum()

def obj_func1(adapter, decision, allocation):
    return (decision * (adapter.C / allocation * 1000 + adapter.Z)).sum()

def obj_func2(adapter, decision, allocation):
    return (decision * (adapter.lambda1 * (adapter.C / allocation * 1000 + adapter.Z) + adapter.lambda2 * (allocation ** 2 * adapter.C + adapter.P_u * adapter.Z))).sum()

def obj_func3(adapter, decision, allocation):
    return (decision * (adapter.lambda1 * (adapter.C / allocation * 1000) + adapter.lambda2 * (allocation ** 2 * adapter.C))).sum()


def fitness(adapter, decision, allocation):
    return -obj_func2(adapter, decision, allocation)


def get_allocation(adapter, decision):
    lamb = adapter.ori_F / np.clip((decision * np.sqrt(adapter.C)).sum(axis=0), 1e-10, 1e10)
    f = np.clip(np.sqrt(decision * adapter.C) * lamb, 1e-10, 1e10)
    return f


def greedy(adapter):
    sorted_C = sorted(enumerate(adapter.ori_C), key=lambda x: x[1], reverse=True)
    ori_cost_F = adapter.ori_C.mean() / adapter.ori_F * 1000
    cost_F, cost_Z = np.array(ori_cost_F), adapter.Z
    decision = np.zeros((adapter.N, adapter.M))
    for task in sorted_C:
        cost = cost_F + cost_Z
        min_cost = min(enumerate(cost[task[0]]), key=lambda x: x[1])
        decision[task[0], min_cost[0]] = 1
        cost_F[min_cost[0]] = ori_cost_F[min_cost[0]] * decision[:, min_cost[0]].sum()
        # cost_F[min_cost[0]] *= 2
        # print(decision[:, min_cost[0]].sum())
    return decision

    # print(sorted_C)
    # print(cost_F)
    # print(cost_Z)



if __name__ == '__main__':

    tasks = Tasks().tasks
    adapter = Adapter([Vehicle() for i in range(20)], tasks)
    # decision = Population(adapter, p_size=100).GA(T=2000)
    decision = greedy(adapter)
    print(adapter.N, adapter.M)
    print(decision)
    swarm = Swarm(adapter, decision)
    swarm.PSO()
    # adapter = Adapter(ground_station.choose_satellites(taskss[1]), taskss[1])
    # decision = greedy(adapter)
    # swarm = Swarm(adapter, decision)
    # swarm.PSO()
