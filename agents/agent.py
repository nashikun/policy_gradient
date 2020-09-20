import time
from abc import ABCMeta, abstractmethod
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from gym import Env

from utils.converters import Converter
from utils.memory import Memory


class Agent(metaclass=ABCMeta):
    def __init__(self, env: Env, *args):
        self.env = env
        self.action_space = Converter(env.action_space)
        self.state_space = Converter(env.observation_space)
        self.epoch_memory: Optional[Memory] = None
        self.episode_memory: Optional[Memory] = None
        self.setup_memory()

    def store_step(self, *args):
        self.episode_memory.store(args)

    def store_episode(self) -> None:
        self.cumulate_rewards()
        self.epoch_memory.extend(self.episode_memory)
        self.episode_memory.reset()

    def train(self, n_epochs: int, n_episodes: int, n_steps: int, render: bool) -> None:
        for epoch in range(n_epochs):
            returned_rewards = []
            episode_steps = []
            for episode in range(n_episodes):
                state = self.env.reset()
                ep_reward = 0
                for step in range(n_steps):
                    action = self.act(state)

                    if render:
                        self.env.render()

                    next_state, reward, done, _ = self.env.step(action[0])
                    ep_reward += reward
                    if not done:
                        self.store_step(state, next_state, *action, reward)
                        state = next_state
                    else:
                        break
                self.store_episode()
                returned_rewards.append(ep_reward)
                episode_steps.append(step + 1)
            print(f"Epoch {epoch + 1} / {n_epochs}: Average returned reward: {np.mean(returned_rewards)}")
            print(f"Epoch {epoch + 1} / {n_epochs}: Average episode length: {np.mean(episode_steps)}")
            self.update()
        self.env.close()

    def evaluate(self, n_episodes: int, n_steps: int, render: bool) -> None:
        returned_rewards = []
        for episode in range(n_episodes):
            state = self.env.reset()
            ep_reward = 0
            for step in range(n_steps):
                action = self.act(state)

                if render:
                    self.env.render()
                    time.sleep(0.001)

                next_state, reward, done, _ = self.env.step(action[0])
                self.store_step(state, next_state, *action, reward)
                ep_reward += reward
                state = next_state
                if done:
                    break
            returned_rewards.append(ep_reward)
        print(f"Average reward: {np.mean(returned_rewards)}")
        self.env.close()

    @abstractmethod
    def cumulate_rewards(self, inputs: list) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def act(self, state: torch.Tensor) -> Tuple:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def setup_memory(self) -> None:
        raise NotImplementedError("Not implemented")
