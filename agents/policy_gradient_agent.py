from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env

from agents.agent import Agent
from utils.mlp import MLP


class PolicyGradient(Agent):
    def __init__(self, env: Env, lr: float, gamma: float = 0.99, layers=(128, 128)):
        super().__init__(env)
        self.gamma = gamma

        if self.action_space.discrete:
            head = nn.Softmax(dim=-1)
            output = self.action_space.shape[0]
        else:
            head = None
            output = 2 * self.action_space.shape[0]

        self.model = MLP(self.state_space.shape[0], output, layers, head)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.reset()

    def reset(self):
        self.reward_memory = torch.Tensor([])
        self.episode_memory.reset()
        self.epoch_memory.reset()

    def act(self, state: List) -> Tuple:
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.model(state)
        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()
        return action.data.numpy(), distribution.log_prob(action)

    def update(self) -> None:
        loss = - torch.sum(self.reward_memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()

    def cumulate_rewards(self, input: list):
        cumulated_reward = 0
        cumulated_rewards = []
        log_probs, rewards = input
        for i in range(len(rewards) - 1, -1, -1):
            cumulated_reward = self.gamma * cumulated_reward + rewards[i]
            cumulated_rewards.append(cumulated_reward)
        cumulated_rewards = torch.Tensor(cumulated_rewards[::-1])
        self.reward_history.append(cumulated_rewards.mean())
        cumulated_rewards = (cumulated_rewards - cumulated_rewards.mean()) / (
                cumulated_rewards.std() + np.finfo(np.float32).eps)
        results = torch.mul(cumulated_rewards, torch.stack(log_probs))
        return results
