from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env
from torch.optim.lr_scheduler import ExponentialLR

from agents.agent import Agent
from utils.memory import Memory
from utils.mlp import MLP


class GeneralizedAdvantageEstimation(Agent):
    def __init__(self, env: Env, policy_lr: float, value_lr: float, gamma: float = 0.99, falloff=0.99, value_iter=50,
                 policy_layers=(128, 128), value_layers=(128, 128), verbose=False, save=True, policy_path=None,
                 value_path=None):
        super().__init__(env, verbose, save)
        self.gamma = gamma
        self.falloff = falloff
        self.policy_path = policy_path
        self.value_path = value_path
        if self.action_space.discrete:
            policy_head = nn.Softmax(dim=-1)
        else:
            policy_head = nn.Tanh()

        self.policy_model = MLP(self.state_space.shape[0], self.action_space.shape[0], policy_layers, policy_head)
        self.value_model = MLP(self.state_space.shape[0], 1, value_layers, None)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=value_lr)
        self.value_loss = nn.MSELoss()
        self.reset()
        self.value_iter = value_iter

    def setup_memory(self) -> None:
        columns = ["states", "next_states", "actions", "log_probs", "rewards"]
        self.episode_memory = Memory(columns)
        self.epoch_memory = Memory(columns)

    def act(self, state: List) -> Tuple:
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.policy_model(state)
        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()
        return action.data.numpy(), distribution.log_prob(action)

    def update(self) -> None:
        states, next_states, rewards, cumulated_advantages, log_probs = self.epoch_memory.get_columns(
            ["states", "next_states", "rewards", "cumulated_advantages", "log_probs"])

        # Train the value function a cetrain number of iterations
        for _ in range(self.value_iter):
            values = self.value_model(torch.Tensor(states)).squeeze()
            value_loss = self.value_loss(values, torch.stack(cumulated_advantages))
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1)
            self.value_optimizer.step()

        print(f"Value Loss: {value_loss.item()}")
        # Compute the policy loss using th previous value function
        policy_loss = - torch.mean(torch.mul(torch.stack(log_probs), torch.stack(cumulated_advantages)))
        print(f"Policy Loss: {policy_loss.item()}")
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1)
        self.policy_optimizer.step()
        self.reset()

    def save_model(self) -> None:
        torch.save(self.policy_model.state_dict(), self.policy_path)
        torch.save(self.value_model.state_dict(), self.value_path)

    def load_model(self, policy_path, value_path) -> None:
        self.policy_model.load_state_dict(torch.load(policy_path))
        self.value_model.load_state_dict(torch.load(value_path))
        self.policy_model.eval()
        self.value_model.eval()

    def setup_schedulers(self, n_epochs: int) -> None:
        policy_scheduler = ExponentialLR(self.policy_optimizer, 0.97)
        value_scheduler = ExponentialLR(self.value_optimizer, 0.97)
        self.schedulers.append(policy_scheduler)
        self.schedulers.append(value_scheduler)

    def cumulate_rewards(self) -> None:
        rewards, states, next_states = self.episode_memory.get_columns(["rewards", "states", "next_states"])
        with torch.no_grad():
            advantages = torch.Tensor(rewards) + self.gamma * (
                    self.value_model(torch.Tensor(next_states)) - self.value_model(torch.Tensor(states))).squeeze()
        cumulated_advantage = 0
        cumulated_advantages = []
        for i in range(len(advantages) - 1, -1, -1):
            cumulated_advantage = self.falloff * self.gamma * cumulated_advantage + advantages[i]
            cumulated_advantages.append(cumulated_advantage)
        self.episode_memory.extend_column("cumulated_advantages", cumulated_advantages[::-1])
