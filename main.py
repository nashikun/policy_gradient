import argparse
import ast
import random
from typing import List

import gym
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

from agents.actor_critic import ActorCritic
from agents.generalized_advantage_estimation import GeneralizedAdvantageEstimation
from agents.policy_gradient_agent import PolicyGradient
from agents.random_agent import RandomAgent


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="The openai gym", default="LunarLander-v2", required=False)
    parser.add_argument('--episodes', type=int, help="The number of episodes per epoch", default=50, required=False)
    parser.add_argument('--epochs', type=int, help="The number of epochs", default=50, required=False)
    parser.add_argument('--steps', type=int, help="The number of steps", default=300, required=False)
    parser.add_argument('--render', type=ast.literal_eval, help="Whether to render the environment", default=True,
                        required=False)
    parser.add_argument('--train', type=ast.literal_eval, help="Whether to update the agent's model", default=True,
                        required=False)
    parser.add_argument('--evaluate', type=ast.literal_eval, help="Whether to evaluate the agent's model", default=True,
                        required=False)
    parser.add_argument('--save', type=ast.literal_eval, help="Whether to save the model", default=True,
                        required=False)
    parser.add_argument('--load', type=ast.literal_eval, help="Whether to load the model", default=False,
                        required=False)
    parser.add_argument('--verbose', type=ast.literal_eval, help="Whether to print extra details", default=False,
                        required=False)
    subparsers = parser.add_subparsers(dest="model")
    random_agent = subparsers.add_parser("rnd")
    policy_gradient = subparsers.add_parser("pg")
    policy_gradient.add_argument('--lr', type=float, help="The learning rate", default=0.01, required=False)
    policy_gradient.add_argument('--layers', nargs='+', type=int, default=[128, 128])
    policy_gradient.add_argument('--model_path', type=str, help="The model path", default="./pg_model.h5",
                                 required=False)
    ac_gae_parent = argparse.ArgumentParser(add_help=False)
    ac_gae_parent.add_argument('--value_iter', type=int, default=50)
    ac_gae_parent.add_argument('--policy_lr', type=float, help="The policy learning rate", default=0.01, required=False)
    ac_gae_parent.add_argument('--value_lr', type=float, help="The value learning rate", default=0.01, required=False)
    ac_gae_parent.add_argument('--policy_layers', nargs='+', type=int, default=[128, 128])
    ac_gae_parent.add_argument('--value_layers', nargs='+', type=int, default=[128, 128])
    ac_gae_parent.add_argument('--policy_model_path', type=str, help="The policy model path",
                               default="./ac_policy_model.h5", required=False)
    ac_gae_parent.add_argument('--value_model_path', type=str, help="The value model path",
                               default="./ac_value_model.h5", required=False)
    actor_critic = subparsers.add_parser("ac", parents=[ac_gae_parent])
    gae_agent = subparsers.add_parser("gae", parents=[ac_gae_parent])
    gae_agent.add_argument('--falloff', type=float, default=0.97, required=False)
    return parser


def get_configs(args):
    if args.model == "pg":
        model_config = {"lr": args.lr, "layers": args.layers, "verbose": args.verbose, "save": args.save,
                        "model_path": args.model_path}
        load_config = {"model_path": args.model_path}
    elif args.model in ["gae", "ac"]:
        model_config = {"policy_lr": args.policy_lr, "value_lr": args.value_lr, "policy_layers": args.policy_layers,
                        "value_layers": args.value_layers, "value_iter": args.value_iter, "verbose": args.verbose,
                        "save": args.save, "policy_path": args.policy_model_path, "value_path": args.value_model_path}
        load_config = {"policy_path": args.policy_model_path, "value_path": args.value_model_path}
        if args.model == "gae":
            model_config["falloff"] = args.falloff
    elif args.model == "rnd":
        model_config = {}
        load_config = {}
    else:
        raise ValueError("Unknown model")

    train_config = {"n_epochs": args.epochs, "n_episodes": args.episodes, "n_steps": args.steps,
                    "render": args.render}

    return model_config, train_config, load_config


def plot_rewards(input_series: List[float], window=100):
    plt.plot(input_series)
    ret = np.cumsum(input_series, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    moving_average = ret[window-1:] / window
    plt.plot(moving_average)
    plt.show()


def main():
    parser = setup_parser()
    args = parser.parse_args()
    env = gym.make(args.env)
    model_type = args.model

    if not model_type:
        raise ValueError("Please specify the model")

    model_config, train_config, load_config = get_configs(args)

    if model_type == "pg":
        model = PolicyGradient(env, **model_config)
    elif model_type == "ac":
        model = ActorCritic(env, **model_config)
    elif model_type == "gae":
        model = GeneralizedAdvantageEstimation(env, **model_config)
    elif model_type == "rnd":
        model = RandomAgent(env, **model_config)

    if args.load:
        model.load_model(**load_config)
    if args.train:
        reward_history, loss = model.train(**train_config)
        plot_rewards(reward_history)
    if args.evaluate:
        model.evaluate(n_episodes=10, n_steps=1000, render=args.render)
        plot_rewards(evaluation_results)
    # plt.plot(model.loss_history)


if __name__ == "__main__":
    main()
