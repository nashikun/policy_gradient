from typing import Tuple

import torch

from agents.agent import Agent


class RandomAgent(Agent):

    def act(self, state: torch.Tensor) -> Tuple:
        return self.action_space.sample(),

    def update(self):
        pass

    def save(self):
        pass

    def store_step(self, *args):
        pass

    def end_episode(self):
        pass

    def setup_schedulers(self, n_epochs: int):
        pass

    def cumulate_rewards(self):
        pass

    def setup_memory(self) -> None:
        pass
