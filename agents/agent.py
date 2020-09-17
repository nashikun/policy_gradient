from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
from gym import Env

from utils.converters import Converter
from utils.memory import Memory


class Agent(metaclass=ABCMeta):
    def __init__(self, env: Env, *args):
        self.env = env
        self.action_space = Converter(env.action_space)
        self.state_space = Converter(env.observation_space)
        self.epoch_memory = Memory()
        self.episode_memory = Memory(reverse=True)
        self.reward_memory = torch.tensor([])
        self.reward_history = []
        self.loss_history = []
        self.episode_length = []

    def store_step(self, *args):
        self.episode_memory.store(args)

    def store_episode(self) -> None:
        episode = self.episode_memory.get_values()
        rewards = self.cumulate_rewards(episode[3:])
        self.reward_memory = torch.cat([self.reward_memory, rewards])
        self.epoch_memory.store(episode[:3])
        self.episode_memory.reset()

    def train(self, n_epochs: int, n_episodes: int, n_steps: int, render: bool) -> None:
        for epoch in range(n_epochs):
            for episode in range(n_episodes):
                state = self.env.reset()

                for time in range(n_steps):
                    action = self.act(state)

                    if render:
                        self.env.render()

                    next_state, reward, done, _ = self.env.step(action[0])
                    self.store_step(state, next_state, *action, reward)
                    state = next_state
                    if done:
                        break
                self.store_episode()
            self.update()
        self.env.close()

    @abstractmethod
    def cumulate_rewards(self, rewards: list) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def act(self, state: torch.Tensor) -> Tuple:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError("Not implemented")
