import torch

from agents.agent import Agent


class RandomAgent(Agent):

    def act(self, state: torch.Tensor):
        return self.action_space.sample()

    def update(self):
        pass
