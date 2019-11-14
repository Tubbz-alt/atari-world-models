import dataclasses
import datetime
import math
import multiprocessing
import time
import typing
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
ACTION_EVERY_STEPS: int = 20
SHOW_SCREEN: bool = False
CPUS_TO_USE: int = multiprocessing.cpu_count()


@dataclass
class State:
    filename: str
    observation: np.array
    action: np.array
    reward: float
    done: bool

    FILE_EXTENSION: typing.ClassVar = ".npy"

    def save(self, target_directory):
        np.save(
            target_directory / (self.filename + self.FILE_EXTENSION),
            dataclasses.asdict(self),
        )

    @staticmethod
    def load_as_dict(filename):
        return np.load(filename, allow_pickle=True).item()


def gather_observations_pooled(
    game,
    show_screen=SHOW_SCREEN,
    target_directory=TARGET_DIRECTORY,
    number_of_episodes=NUMBER_OF_EPISODES,
    steps_per_episode=STEPS_PER_EPISODE,
    action_every_steps=ACTION_EVERY_STEPS,
    cpus_to_use=CPUS_TO_USE,
):
    episode_split = spread(number_of_episodes, cpus_to_use)
    logger.debug("episode_split: %s", episode_split)

    # Actual number of CPUs required after splitting games across CPUs
    pool = Pool(len(episode_split))

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

    logger.debug("All results ready")

    # There are not actual results to get, but this reraises exceptions
    # in the workers
    for result in work:
        result.get()

    pool.close()
    pool.join()
    logger.debug("Gathering observations done")


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
    filename = padded_episodes + "-" + padded_states

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

        logger.info("Writing states to disk")
        for state in states:
            state.save(target_directory)

    env.close()

    if not show_screen:
        vdisplay.stop()


def load_observations(game, source_directory=TARGET_DIRECTORY):
    def load_and_transform(filename):
        state_dict = State.load_as_dict(filename)
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor(),]
        )
        state_dict["observation"] = transform(state_dict["observation"])
        return state_dict

    source_directory /= game
    dataset = datasets.DatasetFolder(
        root=str(source_directory),
        loader=load_and_transform,
        extensions=State.FILE_EXTENSION,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, dataset
