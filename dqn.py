import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import random


class DataCenterEnvironment:
    def __init__(self, num_servers, server_capacity):
        self.num_servers = num_servers
        self.server_capacity = server_capacity
        self.servers = [server_capacity] * num_servers

    def reset(self):
        self.servers = [self.server_capacity] * self.num_servers

    def step(self, action):
        # 模拟任务到达
        task = {"id": random.randint(1, 100), "processing_time": random.randint(1, 10)}
        server_id = action

        # 模拟任务处理
        if self.servers[server_id] >= task["processing_time"]:
            self.servers[server_id] -= task["processing_time"]
            reward = -task["processing_time"]
        else:
            reward = -self.servers[server_id]
            self.servers[server_id] = 0

        done = all(server == 0 for server in self.servers)

        return None, reward, done, {}


# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义经验回放缓存
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones


# 定义DQN训练函数
def train_dqn(env, dqn, target_dqn, buffer, batch_size, gamma, optimizer, loss_fn, num_episodes, device, epsilon):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = select_action(dqn, state, epsilon, device)
            next_state, reward, done, _ = env.step(action)

            buffer.add(state, action, reward, next_state, done)

            if len(buffer.buffer) >= batch_size:
                train_dqn_step(dqn, target_dqn, buffer, batch_size, gamma, optimizer, loss_fn)

            state = next_state
            episode_reward += reward

        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        print(f"Episode {episode}, Reward: {episode_reward}")


# 定义DQN选择动作函数
def select_action(dqn, state, epsilon, device):
    if np.random.random() < epsilon:
        action = np.random.randint(4)
    else:
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = dqn(state).argmax().item()
    return action


# 定义DQN训练一步函数
def train_dqn_step(dqn, target_dqn, buffer, batch_size, gamma, optimizer, loss_fn):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32)

    q_values = dqn(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    next_q_values = target_dqn(next_states_tensor).max(1)[0].detach()

    targets = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
    loss = loss_fn(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # 定义数据中心环境和DQN模型
    num_servers = 4
    server_capacity = 10
    env = DataCenterEnvironment(num_servers, server_capacity)

    input_dim = num_servers
    hidden_dim = 64
    output_dim = num_servers
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dqn = DQN(input_dim, hidden_dim, output_dim).to(device)
    target_dqn = DQN(input_dim, hidden_dim, output_dim)

    # 定义经验回放缓存、优化器和损失函数
    buffer_size = 10000
    batch_size = 64
    gamma = 0.99
    epsilon = 0.01
    buffer = ReplayBuffer(buffer_size)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 训练DQN模型
    num_episodes = 1000
    train_dqn(env, dqn, target_dqn, buffer, batch_size, gamma, optimizer, loss_fn, num_episodes, device, epsilon)
