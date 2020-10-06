import time
from abc import ABCMeta, abstractmethod
from typing import Optional, List
from typing import Tuple

import numpy as np
import torch
from gym import Env

from utils.converters import Converter
from utils.memory import Memory


class Agent(metaclass=ABCMeta):
    def __init__(self, env: Env, verbose=False, save=True, *args):
        self.env = env
        self.verbose = verbose
        self.save = save
        self.schedulers = []
        self.action_space = Converter(env.action_space)
        self.state_space = Converter(env.observation_space)
        self.epoch_memory: Optional[Memory] = None
        self.episode_memory: Optional[Memory] = None
        self.setup_memory()

    def store_step(self, *args):
        self.episode_memory.store(args)

    def end_episode(self) -> None:
        # self.episode_memory.normalize_columns(["rewards"])
        self.counter += 1
        self.cumulate_rewards()
        # self.episode_memory.normalize_columns(["cumulated_rewards"])
        self.epoch_memory.extend(self.episode_memory)
        self.episode_memory.reset()

    def reset(self):
        self.episode_memory.reset()
        self.epoch_memory.reset()

    def train(self, n_epochs: int, n_episodes: int, n_steps: int, render: bool) -> List[float]:
        self.setup_schedulers(n_epochs)
        best_reward = - np.inf
        history_rewards = []
        losses = []
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
                    self.store_step(state, next_state, *action, reward, done)
                    state = next_state
                    if done:
                        break
                self.end_episode()
                returned_rewards.append(ep_reward)
                episode_steps.append(step + 1)
            history_rewards.extend(returned_rewards)
            mean_returned_rewards = np.mean(returned_rewards)
            losses.extend(self.epoch_memory.get_columns(["loss"])[0])
            print("-" * 100 + f"\nEpoch {epoch + 1} / {n_epochs}: ")
            print(f"Maximum returned reward: {np.max(returned_rewards)}")
            print(f"Average returned reward: {mean_returned_rewards}")
            print(f"Average episode length: {np.mean(episode_steps)}")
            if self.save and mean_returned_rewards > best_reward:
                best_reward = mean_returned_rewards
                self.save_model()
            self.update()
            for idx, scheduler in enumerate(self.schedulers):
                scheduler.step()
                if self.verbose:
                    print(f"Scheduler {idx + 1}: {scheduler.get_lr()}")
        self.env.close()
        return history_rewards, [x.data.numpy() for x in losses]

    def evaluate(self, n_episodes: int, n_steps: int, render: bool) -> List[float]:
        returned_rewards = []
        for episode in range(n_episodes):
            state = self.env.reset()
            ep_reward = 0
            for step in range(n_steps):
                action = self.act(state, train=False)

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
        return returned_rewards

    @staticmethod
    def get_max_grad(model):
        ave_grads = []
        layers = []
        for n, p in model.named_parameters():
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        return ave_grads

    @abstractmethod
    def cumulate_rewards(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def act(self, state: torch.Tensor, train: bool = True) -> Tuple:
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
    def save_model(self, *args) -> None:
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def load_model(self, *args) -> None:
        raise NotImplementedError("Not Implemented")
