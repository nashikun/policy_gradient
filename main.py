import argparse
import gym
import numpy as np

from models.policy_gradient import PolicyGradient


def main():
    env = gym.make('CartPole-v1')
    episodes = 1000
    lr = 0.1
    model = PolicyGradient(env.observation_space, env.action_space, lr)

    scores = []
    for episode in range(episodes):
        # Reset environment and record the starting state
        state = env.reset()

        for time in range(1000):
            action = model.predict(state)

            # Uncomment to render the visual state in a window
            env.render()

            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data.numpy())

            # Save reward
            model.episode_rewards.append(reward)
            if done:
                break

        model.update()

        # Calculate score to determine when the environment has been solved
        scores.append(time)
        mean_score = np.mean(scores[-100:])
        mean_loss = np.mean(model.loss_history[-100:])

        if episode % 50 == 0:
            print('Episode {}\tAverage length (last 100 episodes): {:.2f}'.format(
                episode, mean_score))
            print('Episode {}\tAverage loss (last 100 episodes): {:.2f}'.format(
                episode, mean_loss))

    #         if mean_score > env.spec.reward_threshold:
    #             print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
    #                   .format(episode, mean_score, time))
    #             break
    return


if __name__ == "__main__":
    main()
