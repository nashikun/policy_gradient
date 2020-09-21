from typing import List, Tuple
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

    def act(self, state: List) -> Tuple:
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.model(state)

        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()
        return action.data.numpy(), distribution.log_prob(action)

    def update(self) -> None:
        logs_probs, cumulated_rewards = self.epoch_memory.get_columns(["log_probs", "cumulated_rewards"])
        loss = - torch.sum(torch.mul(torch.stack(logs_probs), torch.Tensor(cumulated_rewards))) / self.epoch_episodes
        print(f"Loss is {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()
