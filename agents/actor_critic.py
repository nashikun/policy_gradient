from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env

from agents.agent import Agent
from utils.memory import Memory
from utils.mlp import MLP


class ActorCritic(Agent):
    def __init__(self, env: Env, policy_lr: float, value_lr: float, gamma: float = 0.99, value_iter=50,
                 policy_layers=(128, 128),
                 value_layers=(128, 128)):
        super().__init__(env)
        self.gamma = gamma

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
        self.counter = 10

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
        states, next_states, rewards, cumulated_rewards, log_probs = self.epoch_memory.get_columns(
            ["states", "next_states", "rewards", "cumulated_rewards", "log_probs"])
        # Compute the advantge for the previous Value function
        with torch.no_grad():
            advantages = torch.Tensor(rewards) + (
                    self.gamma * self.value_model(torch.Tensor(next_states)) - self.value_model(
                torch.Tensor(states))).squeeze()

        # Train the value function a cetrain number of iterations
        for _ in range(self.value_iter):
            values = self.value_model(torch.Tensor(states)).squeeze()
            value_loss = self.value_loss(values, torch.stack(cumulated_rewards))
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        print(f"Volicy Loss: {value_loss.item()}")
        # Compute the policy loss using th previous value function
        policy_loss = - torch.sum(torch.mul(torch.stack(log_probs), advantages))
        policy_loss /= self.epoch_episodes
        print(f"Policy Loss: {policy_loss.item()}")
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.reset()
