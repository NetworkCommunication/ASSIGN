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
# Vehicle1 = [(1, 9), (2, 9), (3.0, 9), (4.0, 9), (5.0, 9)]
# Vehicle2 = [1, 2, 3, 4, 5]


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

    def update_p_best(self):
        temp = fitness(self.adapter, self.decision, self.p_loc)
        if temp > self.p_best_fit:
            self.p_best_fit = temp
            self.p_best = np.array(self.p_loc)


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
        print(self.g_best_fit, obj_func(adapter, self.decision, self.g_best), np.mean([p.p_best_fit for p in self.swarm]))
        plt.plot(x, y)
        plt.show()
        return self.g_best, self.g_best_fit


def obj_func(adapter, decision, allocation):
    return (decision * (adapter.C / allocation * 1000 + adapter.Z)).sum()


def fitness(adapter, decision, allocation):
    fit = obj_func(adapter, decision, allocation) \
           + (np.clip((decision * allocation).sum(axis=0) - adapter.ori_F, 0, 11) ** 2).sum() * 10000
    # print(np.clip((decision * allocation).sum(axis=0) - adapter.ori_F, 1e-10, 11))
    return -fit


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
    decision = greedy(adapter)
    # print(adapter.N, adapter.M)
    print(decision)
    swarm = Swarm(adapter, decision)
    swarm.PSO()
    # adapter = Adapter(ground_station.choose_satellites(taskss[1]), taskss[1])
    # decision = greedy(adapter)
    # swarm = Swarm(adapter, decision)
    # swarm.PSO()
