from abc import ABCMeta, abstractmethod

import torch
from gym import Env

from utils.converters import Converter


class Agent(metaclass=ABCMeta):
    def __init__(self, env: Env, *args):
        self.env = env
        self.action_space = Converter(env.action_space)
        self.state_space = Converter(env.action_space)
        self.episode_rewards = []
        self.loss_history = []

    @abstractmethod
    def train(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def act(self, state: torch.Tensor):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self):
        raise NotImplementedError("Not implemented")
