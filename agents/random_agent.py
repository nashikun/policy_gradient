from typing import Tuple

import torch

from agents.agent import Agent


class RandomAgent(Agent):

    def setup_memory(self) -> None:
        pass

    def act(self, state: torch.Tensor) -> Tuple:
        return self.action_space.sample(),

    def update(self):
        pass

    def save(self):
        pass

    def store_step(self, *args):
        pass

    def store_episode(self):
        pass
