from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.agent import Agent
from utils.memory import Memory
from utils.mlp import MLP


class PolicyGradient(Agent):
    def __init__(self, env: Env, lr: float, gamma: float = 0.99, layers=(128, 128), verbose=False, model_path=None,
                 save=False):
        super().__init__(env, verbose, save)
        self.gamma = gamma
        self.model_path = model_path
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

    def act(self, state: List, train: bool = True) -> Tuple:
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.model(state)

        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()
        if train:
            return action.data.numpy(), distribution.log_prob(action)
        else:
            return torch.argmax(action_probs).data.numpy(),

    def update(self) -> None:
        self.optimizer.zero_grad()
        loss, = self.epoch_memory.get_columns(["loss"])
        loss = torch.mean(torch.stack(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        print(f"Value Loss: {loss.item()}")
        self.reset()

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self, model_path: str) -> None:
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def setup_schedulers(self, n_epochs: int) -> None:
        scheduler = CosineAnnealingLR(self.optimizer, n_epochs)
        self.schedulers.append(scheduler)

    def cumulate_rewards(self) -> None:
        cumulated_reward = 0
        cumulated_rewards = []
        rewards, log_probs = self.episode_memory.get_columns(["rewards", "log_probs"])
        for i in range(len(rewards) - 1, -1, -1):
            cumulated_reward = self.gamma * cumulated_reward + rewards[i]
            cumulated_rewards.append(cumulated_reward)

        cumulated_rewards = cumulated_rewards[::-1]
        loss = - torch.sum(torch.mul(torch.stack(log_probs), torch.Tensor(cumulated_rewards)))
        self.episode_memory.append_column("loss", loss)
        self.episode_memory.extend_column("cumulated_rewards", cumulated_rewards)
