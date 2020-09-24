from typing import Tuple

import torch

from agents.agent import Agent


class RandomAgent(Agent):

    def act(self, state: torch.Tensor, train: bool = True) -> Tuple:
        return self.action_space.sample(),

    def update(self) -> None:
        pass

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass

    def store_step(self, *args) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def setup_schedulers(self, n_epochs: int) -> None:
        pass

    def cumulate_rewards(self) -> None:
        pass

    def setup_memory(self) -> None:
        pass
