import numpy as np
import math
import matplotlib.pyplot as plt
from env import Vehicle, Adapter, Tasks
def obj_func2(adapter, decision, allocation):
    return (decision * (adapter.lambda1 * (adapter.C / allocation * 1000 + adapter.Z) + adapter.lambda2 * (allocation ** 2 * adapter.C + adapter.P_u * adapter.Z))).sum()

def fitness(adapter, decision, allocation):
    return -obj_func2(adapter, decision, allocation)

def get_allocation(adapter, decision):
    allocations = np.zeros(shape=(adapter.N, adapter.M))
    for n in range(adapter.N):
        total_capacity = np.clip((decision[n] * np.sqrt(adapter.C[n])).sum(), 1e-10, 1e10)
        for m in range(adapter.M):
            allocations[n, m] = np.clip(np.sqrt(decision[n, m] * adapter.C[n, m]) / total_capacity, 1e-10, 1e10)
    return allocations


def update_pheromone(trails, ants, best_solution, decay_rate, evaporation_rate):
    trails *= decay_rate
    for ant in ants:
        for i in range(len(ant.solution)):
            trails[ant.solution[i], i] += 1.0 / obj_func2(adapter, ant.solution, get_allocation(adapter, ant.solution))
    trails /= evaporation_rate * best_solution

class Ant(object):
    def __init__(self, adapter):
        self.adapter = adapter
        self.solution = np.zeros(shape=(adapter.N, adapter.M), dtype=int)

    def update_solution(self, decisions):
        allocation = get_allocation(self.adapter, decisions)
        prob = 1 / obj_func2(self.adapter, decisions, allocation)
        for i in range(self.adapter.N):
            for j in range(self.adapter.M):
                if np.random.random() < prob:
                    self.solution[i, j] = 1
                else:
                    self.solution[i, j] = 0

class ACO(object):
    def __init__(self, adapter, num_ants=10, num_iterations=100, decay_rate=0.5, evaporation_rate=0.1):
        self.adapter = adapter
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.evaporation_rate = evaporation_rate
        self.best_solution = np.zeros(shape=(adapter.N, adapter.M), dtype=int)

    def run(self):
        trails = np.ones(shape=(self.adapter.N, self.adapter.M), dtype=float)
        best_fitness = -np.inf
        for iteration in range(self.num_iterations):
            ants = [Ant(self.adapter) for _ in range(self.num_ants)]
            for ant in ants:
                ant.update_solution(trails)
                fitness = obj_func2(self.adapter, ant.solution, get_allocation(self.adapter, ant.solution))
                print(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    self.best_solution = ant.solution
            update_pheromone(trails, ants, best_fitness, self.decay_rate, self.evaporation_rate)
        return self.best_solution, best_fitness

# 在主函数中调用
np.random.seed(10)
tasks = Tasks().tasks
adapter = Adapter([Vehicle() for i in range(20)], tasks)
aco = ACO(adapter)
best_solution, best_fitness = aco.run()
print(best_fitness)
print(best_solution)
