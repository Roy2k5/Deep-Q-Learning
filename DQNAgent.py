import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
import numpy as np

env = gym.make("CartPole-v1", render_mode = "rgb_array")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class Agent(object):
    def __init__(self, lr, n_state, n_action, epsilon, final_epsilon, epsilon_decay, gamma, TAU = 0.1):
        self.TAU = TAU
        self.lr = lr
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.main_model = DQN(n_state, n_action).to(self.device)
        self.target_model = DQN(n_state, n_action).to(self.device)
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.replay_buffer = deque(maxlen = 1000)
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.lr)
        self.training_error = []
    def get_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = torch.argmax(self.main_model(state), dim = 1)
            return action.item()
    def push(self, state, action, reward, next_state, terminated):
        self.replay_buffer.append((state, action, reward, next_state, terminated))
    def get_sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        state_batch = torch.from_numpy(np.array([b[0] for b in batch])).float().to(self.device)
        action_batch = torch.from_numpy(np.array([b[1] for b in batch])).long().unsqueeze(1).to(self.device)
        reward_batch = torch.from_numpy(np.array([b[2] for b in batch])).float().unsqueeze(1).to(self.device)
        next_state_batch = torch.from_numpy(np.array([b[3] for b in batch])).float().to(self.device)
        terminated_batch = torch.from_numpy(np.array(([b[4] for b in batch]))).float().unsqueeze(1).to(self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, terminated_batch

    def update_main(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.get_sample(batch_size)
        q_values = self.main_model(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - terminated_batch) * self.gamma * next_q_values
        loss = F.mse_loss(q_values, target_q_values)
        self.training_error.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def update_target(self):
        new_state_dict = self.target_model.state_dict()
        for key in self.main_model.state_dict():
            new_state_dict[key] = self.TAU * self.main_model.state_dict()[key] + (1 - self.TAU)* self.target_model.state_dict()[key]
        self.target_model.load_state_dict(new_state_dict)
    def decay_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.final_epsilon
    def save_model(self, path):
        torch.save(self.main_model.state_dict(), path)
    def load_model(self, path):
        self.main_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))