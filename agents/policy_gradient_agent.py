from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env

from agents.agent import Agent
from utils.memory import Memory
from utils.mlp import MLP


class PolicyGradient(Agent):
    def __init__(self, env: Env, lr: float, gamma: float = 0.99, layers=(128, 128)):
        super().__init__(env)
        self.gamma = gamma

        if self.action_space.discrete:
            head = nn.Softmax(dim=-1)
        else:
            head = nn.Tanh()

        self.model = MLP(self.state_space.shape[0], self.action_space.shape[0], layers, head)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.reset()

    def setup_memory(self) -> None:
        columns = ["states", "next_states", "actions", "log_probs", "rewards"]
        self.episode_memory = Memory(columns)
        self.epoch_memory = Memory(columns)

    def reset(self):
        self.episode_memory.reset()
        self.epoch_memory.reset()

    def act(self, state: List) -> Tuple:
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.model(state)
        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()
        return action.data.numpy(), distribution.log_prob(action)

    def update(self) -> None:
        logs_probs, cumulated_rewards = self.epoch_memory.get_columns(["log_probs", "cumulated_rewards"])
        loss = - torch.sum(torch.mul(torch.stack(logs_probs), torch.Tensor(cumulated_rewards)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()

    def cumulate_rewards(self):
        cumulated_reward = 0
        cumulated_rewards = []
        log_probs, rewards = self.episode_memory.get_columns(["log_probs", "rewards"])
        for i in range(len(rewards) - 1, -1, -1):
            cumulated_reward = self.gamma * cumulated_reward + rewards[i]
            cumulated_rewards.append(cumulated_reward)
        cumulated_rewards = torch.Tensor(cumulated_rewards[::-1])
        cumulated_rewards = (cumulated_rewards - cumulated_rewards.mean()) / (
                cumulated_rewards.std() + np.finfo(np.float32).eps)
        self.episode_memory.extend_column("cumulated_rewards", cumulated_rewards)
