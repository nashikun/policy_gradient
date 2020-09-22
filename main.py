import argparse
import ast

import gym

from agents.actor_critic import ActorCritic
from agents.policy_gradient_agent import PolicyGradient
from agents.random_agent import RandomAgent


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="The openai gym", default="LunarLander-v2", required=False)
    parser.add_argument('--episodes', type=int, help="The number of episodes per epoch", default=50, required=False)
    parser.add_argument('--epochs', type=int, help="The number of epochs", default=30, required=False)
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
    ac_age_parent = argparse.ArgumentParser(add_help=False)
    ac_age_parent.add_argument('--value_iter', type=int, default=50)
    ac_age_parent.add_argument('--policy_lr', type=float, help="The policy learning rate", default=0.01, required=False)
    ac_age_parent.add_argument('--value_lr', type=float, help="The value learning rate", default=0.05, required=False)
    ac_age_parent.add_argument('--policy_layers', nargs='+', type=int, default=[128, 128])
    ac_age_parent.add_argument('--value_layers', nargs='+', type=int, default=[128, 128])
    ac_age_parent.add_argument('--policy_model_path', type=str, help="The policy model path",
                               default="./ac_policy_model.h5", required=False)
    ac_age_parent.add_argument('--value_model_path', type=str, help="The value model path",
                               default="./ac_value_model.h5", required=False)
    actor_critic = subparsers.add_parser("ac", parents=[ac_age_parent])
    age_agent = subparsers.add_parser("age", parents=[ac_age_parent])
    age_agent.add_argument('--falloff', type=float, default=0.99, required=False)
    return parser


def get_configs(args):
    if args.model_type == "pg":
        model_config = {"lr": args.lr, "layers": args.layers, "verbose": args.verbose}
        load_config = {"model_path": args.model_path}
        save_config = {"model_path": args.model_path}
    elif args.model_type in ["age", "ac"]:
        model_config = {"policy_lr": args.policy_lr, "value_lr": args.value_lr, "policy_layers": args.policy_layers,
                        "value_layers": args.value_layers, "value_iter": args.value_iter, "verbose": args.verbose}
        load_config = {"policy_model_path": args.policy_model_path, "value_model_path": args.value_model_path}
        save_config = {"model_path": args.model_path}
    elif args.model_type == "rnd":
        model_config = {}
        load_config = {}
        save_config = {}
    else:
        raise ValueError("Unknown model")

    train_config = {}

    return model_config, train_config, load_config, save_config


def main():
    parser = setup_parser()
    args = parser.parse_args()
    env = gym.make(args.env)
    model_type = args.model
    if not model_type:
        raise ValueError("Please specify the model")

    model_config, train_config, load_config, save_config = get_configs(args)

    if model_type == "pg":
        model = PolicyGradient(env, **model_config)
    elif model_type == "ac":
        model = ActorCritic(env, **model_config)
    elif model_type == "rnd":
        model = RandomAgent(env, **model_config)

    if args.load:
        model.load(**load_config)
    if args.train:
        model.train(**train_config)
    if args.save:
        model.save(**save_config)

    if args.evaluate:
        model.evaluate(n_episodes=10, n_steps=1000, render=args.render)
    # plt.plot(model.loss_history)


if __name__ == "__main__":
    main()
