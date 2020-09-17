import argparse

import gym

from agents.policy_gradient_agent import PolicyGradient


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="The openai gym", default="LunarLander-v2", required=False)
    parser.add_argument("--episodes", type=int, help="The number of episodes per epoch", default=500, required=False)
    parser.add_argument("--epochs", type=int, help="The number of epochs", default=10, required=False)
    parser.add_argument("--lr", type=float, help="The learning rate", default=0.01, required=False)
    parser.add_argument("--render", type=bool, help="Whether to render the agent's actions", default=True,
                        required=False)
    parser.add_argument("--train", type=bool, help="Whether to update the agent's model", default=True, required=False)
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    env = gym.make(args.env)
    episodes = args.episodes
    epochs = args.epochs
    lr = args.lr
    render = args.render
    train = args.train
    model = PolicyGradient(env, lr)

    for epoch in range(epochs):
        for episode in range(episodes):
            state = env.reset()

            for time in range(100):
                action = model.act(state)

                if render:
                    env.render()

                state, reward, done, _ = env.step(action)
                if done:
                    break
            if train:
                model.update()

    env.close()
    return


if __name__ == "__main__":
    main()
