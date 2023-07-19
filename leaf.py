import simpy


class DataCenter:
    def __init__(self, env, num_servers, server_capacity, dataset):
        self.env = env
        self.server = simpy.Resource(env, capacity=num_servers)
        self.server_capacity = server_capacity
        self.dataset = dataset

    def process_request(self, request):
        yield self.env.timeout(request["processing_time"])


def task_generator(env, data_center, dataset):
    for request in dataset:
        env.process(task(env, data_center, request))
        yield env.timeout(request["arrival_time"])


def task(env, data_center, request):
    with data_center.server.request() as req:
        yield req
        print(f"Request with processing time {request['processing_time']} is processing at time {env.now}")
        yield env.process(data_center.process_request(request))
        print(f"Request is completed at time {env.now}")


if __name__ == "__main__":
    env = simpy.Environment()
    dataset = [
        {"arrival_time": 0, "processing_time": 5},
        {"arrival_time": 2, "processing_time": 4},
        {"arrival_time": 4, "processing_time": 3},
        {"arrival_time": 7, "processing_time": 6},
        # 更多任务...
    ]
    data_center = DataCenter(env, num_servers=2, server_capacity=1000000, dataset=dataset)

    env.process(task_generator(env, data_center, dataset))
    env.run()
