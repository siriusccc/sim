import simpy
import random


class Server:
    def __init__(self, env, server_id):
        self.env = env
        self.serverd = simpy.Resource(env, capacity=80)
        self.server_id = server_id

    def process_request(self, request):
        processing_time = random.randint(1, 5)  # 模拟任务处理时间，随机取1到5的整数
        yield self.env.timeout(processing_time)


def task(env, server):
    with server.serverd.request() as req:
        yield req
        print(f"Task on Server {server.server_id} is processing at time {env.now}")
        yield env.process(server.process_request(server))
        print(f"Task on Server {server.server_id} is completed at time {env.now}")


def data_center(env, num_servers):
    servers = [Server(env, i) for i in range(num_servers)]
    for server in servers:
        env.process(task(env, server=server))
        yield env.timeout(1)


if __name__ == "__main__":
    env = simpy.Environment()
    num_spine_switches = 2
    num_leaf_switches = 4
    servers_per_leaf = 2

    total_servers = num_leaf_switches * servers_per_leaf
    data_centers = data_center(env, total_servers)
    env.process(data_centers)
    env.run(until=100)  # 运行仿真，模拟100个时间单位
