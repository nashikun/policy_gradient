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
    def __init__(self, env: Env, verbose=False, *args):
        self.env = env
        self.verbose = verbose
        self.schedulers = []
        self.action_space = Converter(env.action_space)
        self.state_space = Converter(env.observation_space)
        self.epoch_memory: Optional[Memory] = None
        self.episode_memory: Optional[Memory] = None
        self.epoch_episodes: int = 0
        self.setup_memory()

    def store_step(self, *args):
        self.episode_memory.store(args)

    def end_episode(self) -> None:
        self.epoch_episodes += 1
        self.episode_memory.normalize_columns(["rewards"])
        self.cumulate_rewards()
        self.epoch_memory.extend(self.episode_memory)
        self.episode_memory.reset()
        self.epoch_episodes += 1

    def reset(self):
        self.epoch_episodes = 0
        self.episode_memory.reset()
        self.epoch_memory.reset()

    def train(self, n_epochs: int, n_episodes: int, n_steps: int, render: bool) -> None:
        self.setup_schedulers(n_epochs)
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
                self.end_episode()
                returned_rewards.append(ep_reward)
                episode_steps.append(step + 1)
            print("-" * 100 + f"\nEpoch {epoch + 1} / {n_epochs}: ")
            print(f"Average returned reward: {np.mean(returned_rewards)}")
            print(f"Average episode length: {np.mean(episode_steps)}")
            self.update()
            for idx, scheduler in enumerate(self.schedulers):
                scheduler.step()
                if self.verbose:
                    print(f"Scheduler {idx+1}: {scheduler.get_lr()}")
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
                    time.sleep(0.01)

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
    def cumulate_rewards(self):
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

    @abstractmethod
    def setup_schedulers(self, n_epochs: int) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def save(self, *args):
        raise NotImplementedError("Not Implemented")
