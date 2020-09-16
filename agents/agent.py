from abc import ABCMeta, abstractmethod
import torch

from utils.converters import Converter
from gym import Env


class Agent(metaclass=ABCMeta):
    def __init__(self, env: Env):
        self.env = env
        self.action_space = Converter(env.action_space)
        self.state_space = Converter(env.action_space)

    @abstractmethod
    def train(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def act(self, state: torch.Tensor):
        raise NotImplementedError("Not implemented")
