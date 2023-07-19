import random
import numpy as np


class DataCenter:
    def __init__(self, num_servers, server_capacity):
        self.num_servers = num_servers
        self.server_capacity = server_capacity
        self.servers = [server_capacity] * num_servers

    def allocate_server(self, task):
        # min_server_load = min(self.servers)
        selected_server = self.servers.index(min(self.servers))
        self.servers[selected_server] += task["processing_time"]
        return selected_server


def generate_task():
    return {"id": random.randint(1, 100), "processing_time": random.randint(1, 10)}


def q_learning(env, data_center, num_episodes, learning_rate, discount_factor, exploration_prob):
    q_table = {server_id: [0] * data_center.server_capacity for server_id in range(data_center.num_servers)}

    for episode in range(num_episodes):
        state = 0
        total_reward = 0

        while True:
            if np.random.random() < exploration_prob:
                action = random.choice(range(data_center.num_servers))
                print(action)
                print('----------')
            else:
                action = q_table[state].index(max(q_table[state]))
                print(action)
                print(q_table[state])
                print('/////////////')
            print('+++++++++++++')
            next_state = action
            print(action)
            print('xxxxxxxxxxxxx')
            print(data_center.servers)
            reward = -data_center.servers[action]
            total_reward += reward

            td_error = reward + discount_factor * np.nanmax(q_table[next_state]) - q_table[state][action]
            q_table[state][action] += learning_rate * td_error

            if total_reward >= 0:
                break

            state = next_state
    return q_table


if __name__ == "__main__":
    num_servers = 4
    server_capacity = 100
    num_episodes = 10000
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_prob = 0.2

    data_center = DataCenter(num_servers, server_capacity)

    for _ in range(num_episodes):
        task = generate_task()
        server_id = data_center.allocate_server(task)
        task["server_id"] = server_id

        q_table = q_learning(None, data_center, num_episodes, learning_rate, discount_factor, exploration_prob)

        # 打印每个服务器的Q值
        for i, q_values in enumerate(q_table.items()):
            print(f"Server {i} Q-values: {q_values}")
