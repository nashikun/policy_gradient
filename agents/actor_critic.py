from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env

from agents.agent import Agent
from utils.mlp import MLP


class ActorCritic(Agent):
    def __init__(self, env: Env, lr: float, gamma: float = 0.99, policy_layers=(128, 128), value_layers=(128, 128)):
        super().__init__(env)
        self.gamma = gamma

        if self.action_space.discrete:
            policy_head = nn.Softmax(dim=-1)
        else:
            policy_head = nn.Tanh()

        self.policy_model = MLP(self.state_space.shape[0], self.action_space.shape[0], policy_layers, policy_head)
        self.value_model = MLP(self.state_space.shape[0], 1, value_layers, None)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=lr)
        self.value_loss = nn.MSELoss()
        self.reset()

    def reset(self):
        self.episode_memory.reset()
        self.epoch_memory.reset()

    def act(self, state: List) -> Tuple:
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.policy_model(state)
        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()
        return action.data.numpy(), distribution.log_prob(action)

    def update(self) -> None:
        states, next_states, _, _, rewards = self.epoch_memory.get_values(reverse=True)
        with torch.no_grad():
            target = rewards + self.gamma * self.value_model(torch.Tensor(next_states))
        values = self.value_model(states)
        value_loss = self.value_loss(values, target)
        policy_loss = - torch.sum()
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        self.reset()

    def cumulate_rewards(self, input: List):
        states, _, _, log_probs, rewards = list(zip(*input))

        rewards = (rewards - rewards.mean()) / (
                rewards.std() + np.finfo(np.float32).eps)
        with torch.no_grad():
            values = self.value_model(torch.Tensor(states)).T
        results = torch.mul(values, torch.stack(log_probs)).T
        return results

    def get_rewards(self, inputs: list):
        cumulated_reward = 0
        cumulated_rewards = []
        log_probs, rewards = list(zip(*inputs))[-2:]
        for i in range(len(rewards) - 1, -1, -1):
            cumulated_reward = self.gamma * cumulated_reward + rewards[i]
            cumulated_rewards.append(cumulated_reward)
        cumulated_rewards = torch.Tensor(cumulated_rewards[::-1])
        cumulated_rewards = (cumulated_rewards - cumulated_rewards.mean()) / (
                cumulated_rewards.std() + np.finfo(np.float32).eps)
        results = torch.mul(cumulated_rewards, torch.stack(log_probs))
        return results
