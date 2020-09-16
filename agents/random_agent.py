from agents.agent import Agent
import torch


class RandomAgent(Agent):

    def train(self):
        pass

    def act(self, state: torch.Tensor):
        return self.action_space.sample()
