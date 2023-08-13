import numpy as np
import math
import matplotlib.pyplot as plt

from env import Vehicle, Adapter, Tasks
# from Larange import larange



def obj_func1(adapter, decision, allocation):
    return (decision * (adapter.C / allocation * 1000 + adapter.Z)).sum()

def obj_func2(adapter, decision, allocation):
    return (decision * (adapter.lambda1 * (adapter.C / allocation * 1000 + adapter.Z) + adapter.lambda2 * (allocation ** 2 * adapter.C + adapter.P_u * adapter.Z))).sum()

def obj_func3(adapter, decision, allocation):
    return (decision * (adapter.lambda1 * (adapter.C / allocation * 1000) + adapter.lambda2 * (allocation ** 2 * adapter.C))).sum()

def obj_fun4(adapter, decision, allocation):
    return (decision * (adapter.lambda1 * (adapter.C / allocation * 1000 + adapter.Z) + adapter.lambda2 * (allocation ** 2 * adapter.C + adapter.P_u * adapter.Z + adapter.P_r * adapter.H))).sum()

def fitness(adapter, decision, allocation):
    return -obj_func2(adapter, decision, allocation)


def get_allocation(adapter, decision):
    lamb = adapter.ori_F / np.clip((decision * np.sqrt(adapter.C)).sum(axis=0), 1e-10, 1e10)
    f = np.clip(np.sqrt(decision * adapter.C) * lamb, 1e-10, 1e10)
    return f


class Individual (object):
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
    #
    # def get_best_individual(self, pop):
    #     max = -1e10
    #     for individual in self.pop:
    #         val = individual.get_fitness()
    #         if max < val:
    #            max = val
    #            self.best_individual = individual
    #     print(self.best_individual)
    #     # return self.best_individual


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



    def crossover(self, p):
        for i in range(0, self.p_size, 2):
            for n in range(self.adapter.N):
                if np.random.random() < p:
                    self.pop[i].individual[n], self.pop[i + 1].individual[n] = \
                        self.pop[i + 1].individual[n], self.pop[i].individual[n]

    def mutate(self, p):
        for i in range(self.p_size):
            if np.random.random() < p:
                for n in range(self.adapter.N):
                    if np.random.random() < p:
                        self.pop[i].individual[n] = np.zeros(self.adapter.M)
                        self.pop[i].individual[n, np.random.randint(0, self.adapter.M)] = 1
            # print(self.pop[i].individual)


        # # 示例参数
        # t = 50
        # t_max = 100
        # p_m0 = 0.1
        # lamda = 0.05
        #
        # # 计算突变概率
        # p_m = mutation_probability(t, t_max, p_m0, lamda)
        #
        # print("在时间 t = {t} 时的突变概率为 " + p_m)

    def GA(self,T=2000):
        def mutation_probability(t, t_max, p_m0, lamda):

            exponent = -lamda * (t / t_max)
            if t > t_max:
                t = t_max

            p_m = p_m0 * math.exp(exponent)
            print(p_m)
            return p_m
        x, y1, y2 = [], [], []
        best = -1e10
        for t in range(T):
            self.select()
            x.append(t)
            yy1, yy2 = self.get_info()
            if best < yy1:
                best = yy1
            y1.append(yy1)
            y2.append(yy2)
            self.crossover(p=0.2)
            self.mutate(p=mutation_probability(t, 100, 0.1, 0.05))

        self.get_best_individual(self.pop)

        print(y2[T-1], best)
        print(self.pop)
        plt.plot(x, y1)
        # plt.plot(x, y2)
        plt.show()

if __name__ == '__main__':
    np.random.seed(10)
    tasks = Tasks().tasks
    adapter = Adapter([Vehicle() for i in range(20)], tasks)
    print(adapter.M, adapter.N, math.pow(adapter.M, adapter.N))

    Population(adapter, p_size=50).GA(T=1000)






