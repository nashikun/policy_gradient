from abc import ABCMeta, abstractmethod

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

    def store_episode(self):
        episode = self.episode_memory.get_values()
        rewards = self.cumulate_rewards(episode[3:])
        self.reward_memory = torch.cat([self.reward_memory, rewards])
        self.epoch_memory.store(episode[:3])
        self.episode_memory.reset()

    @abstractmethod
    def cumulate_rewards(self, rewards: list):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def act(self, state: torch.Tensor):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self):
        raise NotImplementedError("Not implemented")
