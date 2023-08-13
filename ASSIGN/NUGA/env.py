import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(12580)
class Task(object):
    def __init__(self):
        """
        随机生成一个任务
        """
        self.data_size = np.random.uniform(0.1, 0.2)  # M
        self.computation = np.random.uniform(self.data_size-0.01, self.data_size+0.01)  # Gcycle
        self.delay = np.random.uniform(80, 150)  # ms
        self.R = 10
        self.H = 15
        self.fitness = self.delay - (self.data_size / self.R) * 1000 - self.computation / 10 * 1000


class Tasks(object):
    def __init__(self, task_num=20, k=3):
        """
        产生一组任务
        :param task_num: 任务数量
        :param k: 分类数量
        """
        self.k = k
        self.task_num = task_num
        self.tasks = [Task() for i in range(task_num)]

    def classify(self):
        """
        将这组任务聚簇
        :return: [[最着急的任务s], [。。。], [。。。]]
        """
        fitness = np.array([task.fitness for task in self.tasks]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(fitness)
        classification = [[] for i in range(self.k)]
        temp, idx = -1, -1
        for item in sorted(zip(kmeans.labels_, self.tasks), key=lambda x: x[1].fitness):
            if item[0] != temp:
                temp = item[0]
                idx += 1
            classification[idx].append(item[1])

        return classification

class Vehicle(object):
    def __init__(self):

        self.computation_capacity = np.random.uniform(9, 11)  # Gcycle/s
        # self.computation_capacity = 10


class RSU(object):
    def __init__(self):

        self.computation_capacity = np.random.uniform(9, 11)  # Gcycle/s


class Adapter(object):
    def __init__(self, vehicles, tasks):
        """
        获取后续算法所需要数据
        :param vehicles: 车辆s
        :param RSU: RSU
        :param tasks: 任务s
        """
        self.N = len(tasks)
        self.M = len(vehicles)
        self.C = np.repeat([task.computation for task in tasks], self.M, 0).reshape(self.N, self.M)
        self.ori_C = np.array([task.computation for task in tasks])

        self.F = np.array([vel.computation_capacity for vel in vehicles] * self.N).reshape(self.N, self.M)
        self.ori_F = np.array([vel.computation_capacity for vel in vehicles])
        self.Z = np.zeros((self.N, self.M))
        for n in range(self.N):
            for m in range(self.M):
                self.Z[n, m] = (tasks[n].data_size / tasks[n].R * 1000)

        self.H = np.zeros((self.N, self.M))
        for n in range(self.N):
            for m in range(self.M):
                self.Z[n, m] = (tasks[n].data_size / tasks[n].H * 1000)



        self.lambda1 = 0.1
        self.lambda2 = 0.9
        self.Z2 = np.zeros((self.N, self.M))
        self.P_u = 0.5
        self.P_d = 0.5
        self.P_r = 0.3
        for n in range(self.N):
            for m in range(self.M):
                self.Z2[n, m] = (tasks[n].data_size / tasks[n].R * 1000 )