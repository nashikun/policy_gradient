import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.agent import Agent


class PolicyGradient(Agent):
    def __init__(self, env, lr, gamma=0.99):
        super().__init__(env)
        # super(nn.Module, self).__init__()
        self.gamma = gamma

        if self.action_space.discrete:
            head = nn.Softmax(dim=-1)
        else:
            head = nn.Linear()
        num_hidden = 128

        self.model = torch.nn.Sequential(
            nn.Linear(self.state_space.shape[0], num_hidden, bias=False),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(num_hidden, self.action_space.shape[0], bias=False),
            head
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.forward(state)
        distribution = self.action_space.distribution(action_probs)
        action = distribution.sample()

        self.episode_actions = torch.cat([self.episode_actions, distribution.log_prob(action).reshape(1)])

        return action.data.numpy()

    def update(self):
        total_reward = 0
        rewards = []

        for r in self.episode_rewards[::-1]:
            total_reward = r + self.gamma * total_reward
            rewards.insert(0, total_reward)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        loss = (torch.sum(torch.mul(self.episode_actions, rewards).mul(-1), -1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.loss_history.append(loss.item())
        self.reward_history.append(np.sum(self.episode_rewards))
        self.reset()

    def train(self):
        pass
