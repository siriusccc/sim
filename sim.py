import simpy


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


class DataCenter:
    def __init__(self, env, num_servers, server_capacity):
        self.env = env
        self.server = simpy.Resource(env, capacity=num_servers)
        self.server_capacity = server_capacity

    def process_request(self, request):
        yield self.env.timeout(request["processing_time"])


def server_user(env, data_center):
    for i in range(10):
        request = {"id": i, "processing_time": 3}  # 模拟请求，处理时间为3个时间单位
        with data_center.server.request() as req:
            yield req
            print(f"Request {request['id']} is processing at time {env.now}")
            yield env.process(data_center.process_request(request))
            print(f"Request {request['id']} is completed at time {env.now}")
            print("-"*20)


if __name__ == "__main__":
    env = simpy.Environment()
    data_center = DataCenter(env, num_servers=5, server_capacity=10)  # 5台服务器，每台服务器容量为10
    env.process(server_user(env, data_center))
    env.run()
