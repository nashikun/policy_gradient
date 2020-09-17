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

    def distribution(self, logits: torch.Tensor) -> D.MultivariateNormal:
        assert not logits.size(0) % 2
        mid = logits.size // 2
        scale = logits[:, :mid]
        loc = logits[:, mid:]
        return D.MultivariateNormal(scale, loc)

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
