from typing import Tuple

import gym.spaces as spaces
import torch
import torch.distributions as D
from gym import Space


class DiscreteConverter:
    def __init__(self, space: spaces.Discrete) -> None:
        self.space = space

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.space.n,)

    def distribution(self, probabilities: torch.Tensor) -> D.Categorical:
        return D.Categorical(probabilities)

    def action(self, probabilties: torch.Tensor) -> torch.Tensor:
        return self.distribution(probabilties).sample()

    @property
    def discrete(self) -> bool:
        return True

    def sample(self):
        return self.space.sample()


class BoxConverter:
    def __init__(self, space):
        self.space = space
        self.max = torch.Tensor(space.high)
        self.min = torch.Tensor(space.low)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.space.shape

    def distribution(self, logits: torch.Tensor) -> D.Normal:
        scale = logits  # [:-1]
        loc = torch.ones_like(logits) * 0.01
        # loc = torch.eye(logits.size(0) - 1) * logits[-1]
        # loc = loc * loc + torch.finfo(torch.float32).eps
        return D.Normal(scale, loc)

    def action(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.min(self.min, torch.max(self.max, self.distribution(logits).sample()))

    @property
    def discrete(self) -> bool:
        return False

    def sample(self):
        return self.space.sample()


def Converter(space: Space):
    if isinstance(space, spaces.Discrete):
        return DiscreteConverter(space)
    elif isinstance(space, spaces.Box):
        return BoxConverter(space)
    raise NotImplementedError("converter is only implemented for discrete and box for now")
