import argparse

import gym
import matplotlib.pyplot as plt

from agents.actor_critic import ActorCritic
from agents.policy_gradient_agent import PolicyGradient


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="The openai gym", default="LunarLander-v2", required=False)
    parser.add_argument("--episodes", type=int, help="The number of episodes per epoch", default=5, required=False)
    parser.add_argument("--epochs", type=int, help="The number of epochs", default=50, required=False)
    parser.add_argument("--lr", type=float, help="The learning rate", default=0.01, required=False)
    parser.add_argument("--render", type=bool, help="Whether to render the agent's actions", default=True,
                        required=False)
    parser.add_argument("--train", type=bool, help="Whether to update the agent's model", default=True, required=False)
    subparsers = parser.add_subparsers(dest="model")
    policy_gradient = subparsers.add_parser("pg")
    policy_gradient.add_argument('--layers', nargs='+', type=int, default=[128, 128, 128, 64])
    actor_critic = subparsers.add_parser("ac")
    actor_critic.add_argument('--policy_layers', nargs='+', type=int, default=[128, 128, 128, 64])
    actor_critic.add_argument('--value_layers', nargs='+', type=int, default=[128, 128, 128, 64])
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    env = gym.make(args.env)
    episodes = args.episodes
    epochs = args.epochs
    lr = args.lr
    render = args.render
    model_type = args.model
    if not model_type:
        raise ValueError("Please specify the model")
    if model_type == "pg":
        model = PolicyGradient(env, lr, layers=args.layers)
    elif model_type == "ac":
        model = ActorCritic(env, lr, policy_layers=args.policy_layers, value_layers=args.value_layers)
    # model.model.load_state_dict(torch.load("./model.h5"))
    # model.model.eval()
    model.train(n_epochs=epochs, n_episodes=episodes, n_steps=100, render=render)
    # model.evaluate(n_episodes=3, n_steps=1000, render=render)
    # torch.save(model.model.state_dict(), "./model.h5")
    plt.plot(model.loss_history)
    return


if __name__ == "__main__":
    main()
