import random
import simpy


# 定义边缘计算服务器
class EdgeServer:
    def __init__(self, env, id, cpu_capacity, gpu_capacity):
        self.env = env
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.gpu_capacity = gpu_capacity
        self.cpu_available = cpu_capacity
        self.gpu_available = gpu_capacity

    # 分配计算资源
    def allocate(self, task):
        if task.is_gpu and self.gpu_available >= task.requirement:
            self.gpu_available -= task.requirement
            return True
        elif not task.is_gpu and self.cpu_available >= task.requirement:
            self.cpu_available -= task.requirement
            return True
        else:
            return False

    # 释放计算资源
    def release(self, task):
        if task.is_gpu:
            self.gpu_available += task.requirement
        else:
            self.cpu_available += task.requirement


# 定义任务
class Task:
    def __init__(self, env, id, requirement, is_gpu):
        self.env = env
        self.id = id
        self.requirement = requirement
        self.is_gpu = is_gpu
        self.start_time = None
        self.end_time = None

    # 执行任务
    def execute(self, edge_server):
        self.start_time = self.env.now
        yield self.env.timeout(random.uniform(0, 1))  # 模拟处理时延.
        self.end_time = self.env.now
        execution_time = self.end_time - self.start_time
        print("Task {} executed on edge server {} in {:.2f} seconds (start time: {:.2f}, end time: {:.2f})".format(
            self.id, edge_server.id, execution_time, self.start_time, self.end_time))
        edge_server.release(self)


# 定义仿真环境
class Simulator:
    def __init__(self, num_servers, cpu_capacity, gpu_capacity, num_tasks, task_requirement):
        self.env = simpy.Environment()
        self.servers = [EdgeServer(self.env, i, cpu_capacity, gpu_capacity) for i in range(num_servers)]
        self.tasks = [Task(self.env, i, task_requirement, random.choice([True, False])) for i in range(num_tasks)]
        self.env.process(self.run())

    # 任务调度
    def schedule(self):
        for task in self.tasks:
            allocated = False
            for server in self.servers:
                if server.allocate(task):
                    allocated = True
                    self.env.process(task.execute(server))
                    break
            if not allocated:
                print("Task {} cannot be executed due to insufficient resources".format(task.id))

    # 运行仿真
    def run(self):
        while True:
            yield self.env.timeout(1)
            self.schedule()


# 运行仿真
simulator = Simulator(3, 10, 5, 10, 2)
simulator.env.run(until=10)