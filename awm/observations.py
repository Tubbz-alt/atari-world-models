import datetime
import math
import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from xvfbwrapper import Xvfb

from . import logger
from .utils import spread

TARGET_DIRECTORY: Path = Path("data")
NUMBER_OF_EPISODES: int = 1
STEPS_PER_EPISODE: int = 2000
ACTION_EVERY_STEPS: int = 10
SHOW_SCREEN: bool = False
CPUS_TO_USE: int = multiprocessing.cpu_count()


@dataclass
class State:
    filename: str
    observation: np.array
    action: int
    reward: float
    done: bool


def gather_observations_pooled(
    game,
    show_screen=SHOW_SCREEN,
    target_directory=TARGET_DIRECTORY,
    number_of_episodes=NUMBER_OF_EPISODES,
    steps_per_episode=STEPS_PER_EPISODE,
    action_every_steps=ACTION_EVERY_STEPS,
    cpus_to_use=CPUS_TO_USE,
):
    pool = Pool(cpus_to_use)

    episode_split = spread(number_of_episodes, cpus_to_use)
    logger.debug("episode_split: %s", episode_split)

    def build_args(episodes):
        return (
            game,
            show_screen,
            target_directory,
            episodes,
            steps_per_episode,
            action_every_steps,
        )

    work = []
    for episodes in episode_split:
        args = build_args(episodes)
        logger.debug("Starting worker with %s", args)
        work.append(pool.apply_async(gather_observations, args))

    while not all(result.ready() for result in work):
        time.sleep(0.1)


def gather_observations(
    game,
    show_screen,
    target_directory,
    number_of_episodes,
    steps_per_episode,
    action_every_steps,
):
    """ Play the given game for a number of episodes. Each episode lasts a
    given number of steps. Every N steps a random action is taken.

    An episode ends if either the game ends or the number of steps is reached.
    After each episode the collected observations of the screen are saved as
    images to a target directory.
    """

    def padding(number):
        return "{:0%d}" % len(str(number))

    padded_episodes = padding(number_of_episodes)
    padded_states = padding(steps_per_episode)
    # A .format() style string with nice padding
    filename = padded_episodes + "-" + padded_states + ".png"

    name = multiprocessing.current_process().name
    stamp = datetime.datetime.now().isoformat() + "-" + name
    target_directory /= game / Path(stamp)
    target_directory.mkdir(parents=True, exist_ok=True)

    if not show_screen:
        vdisplay = Xvfb()
        vdisplay.start()

    env = gym.make(game)

    for episode in range(number_of_episodes):
        env.reset()
        states = []

        for step in range(steps_per_episode):
            if step % 100 == 0:
                logger.debug("%s: e=%d s=%d", name, episode, step)
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

    if not show_screen:
        vdisplay.stop()


def load_observations(game, source_directory=TARGET_DIRECTORY):
    source_directory /= game
    dataset = datasets.ImageFolder(
        root=str(source_directory),
        transform=transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor(),]
        ),
        # transform=transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader, dataset
