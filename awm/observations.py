from dataclasses import dataclass
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np

TARGET_DIRECTORY: Path = Path("data")
NUMBER_OF_EPISODES: int = 1
STEPS_PER_EPISODE: int = 2000
ACTION_EVERY_STEPS: int = 10


@dataclass
class State:
    filename: str
    observation: np.array
    action: int
    reward: float
    done: bool


def gather_observations(
    game,
    target_directory=TARGET_DIRECTORY,
    number_of_episodes=NUMBER_OF_EPISODES,
    steps_per_episode=STEPS_PER_EPISODE,
    action_every_steps=ACTION_EVERY_STEPS,
):
    """ Play the given game for a number of episodes. Each episode lasts a
    given number of steps. Every N steps a random action is taken.

    An episode ends if either the game ends or the number of steps is reached.
    After each episode the collected observations of the screen are saved as
    images to a target directory.
    """

    target_directory /= game
    target_directory.mkdir(parents=True, exist_ok=True)

    def padding(number):
        return "{:0%d}" % len(str(number))

    padded_episodes = padding(number_of_episodes)
    padded_states = padding(steps_per_episode)
    # A .format() style string with nice padding
    filename = padded_episodes + "-" + padded_states + ".png"

    env = gym.make(game)

    for episode in range(number_of_episodes):
        env.reset()
        states = []

        for step in range(steps_per_episode):
            env.render()

            # Choose a random action
            if step % action_every_steps == 0:
                action = env.action_space.sample()

            # Take a game step
            observation, reward, done, _ = env.step(action)

            # Save observed state
            state = State(
                filename=filename.format(episode, step),
                observation=observation,
                action=action,
                reward=reward,
                done=done,
            )
            states.append(state)

            if done:
                break

        # Write observations to disk
        for state in states:
            plt.imsave(target_directory / state.filename, state.observation)

    env.close()
