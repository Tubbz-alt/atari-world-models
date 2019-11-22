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
OBSERVATION_DIRECTORY = TARGET_DIRECTORY
NUMBER_OF_PLAYS: int = 1
STEPS_PER_PLAY: int = 2000
ACTION_EVERY_STEPS: int = 20
SHOW_SCREEN: bool = False
CPUS_TO_USE: int = multiprocessing.cpu_count()


@dataclass
class Observation:
    filename: str
    screen: np.array
    action: np.array
    reward: float
    done: bool

    # Filled in later by precompute-z-values
    # FIXME: This is a typing violation
    z: np.array = 0.0

    disk_location: str = ""

    FILE_EXTENSION: typing.ClassVar = ".npy"

    def save(self, target_directory):
        self.disk_location = str(target_directory / (self.filename + self.FILE_EXTENSION))
        np.save(
            self.disk_location, dataclasses.asdict(self),
        )

    @staticmethod
    def load_as_dict(filename):
        return np.load(filename, allow_pickle=True).item()


def gather_observations_pooled(
    game,
    show_screen=SHOW_SCREEN,
    target_directory=TARGET_DIRECTORY,
    number_of_plays=NUMBER_OF_PLAYS,
    steps_per_play=STEPS_PER_PLAY,
    action_every_steps=ACTION_EVERY_STEPS,
    cpus_to_use=CPUS_TO_USE,
):
    play_split = spread(number_of_plays, cpus_to_use)
    logger.debug("play_split: %s", play_split)

    # Actual number of CPUs required after splitting games across CPUs
    pool = Pool(len(play_split))

    def build_args(plays):
        return (
            game,
            show_screen,
            target_directory,
            plays,
            steps_per_play,
            action_every_steps,
        )

    work = []
    for plays in play_split:
        args = build_args(plays)
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
    number_of_plays,
    steps_per_play,
    action_every_steps,
):
    """ Play the given game for a number of plays. Each play lasts at most a
    given number of steps. Every N steps a random action is taken.

    A play ends if either the game ends or the number of steps is reached.
    After each play the collected observations are saved to a target directory.
    """

    def padding(number):
        return "{:0%d}" % len(str(number))

    padded_plays = padding(number_of_plays)
    padded_states = padding(steps_per_play)
    # A .format() style string with nice padding
    filename = padded_plays + "-" + padded_states

    name = multiprocessing.current_process().name
    stamp = datetime.datetime.now().isoformat() + "-" + name
    target_directory /= game / Path(stamp)
    target_directory.mkdir(parents=True, exist_ok=True)

    if not show_screen:
        vdisplay = Xvfb()
        vdisplay.start()

    env = gym.make(game)

    for play in range(number_of_plays):
        env.reset()
        observations = []

        for step in range(steps_per_play):
            if step % 100 == 0:
                logger.debug("%s: e=%d s=%d", name, play, step)
            env.render()

            # Choose a random action
            if step % action_every_steps == 0:
                action = env.action_space.sample()

            # Take a game step
            screen, reward, done, _ = env.step(action)

            observation = Observation(
                filename=filename.format(play, step),
                screen=screen,
                action=action,
                reward=reward,
                done=done,
            )
            observations.append(observation)

            if done:
                break

        logger.info("Writing observations to disk")
        for observation in observations:
            observation.save(target_directory)

    env.close()

    if not show_screen:
        vdisplay.stop()


def load_observations(game, source_directory=TARGET_DIRECTORY, batch_size=32):
    def load_and_transform(filename):
        obs_dict = Observation.load_as_dict(filename)
        transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor(),]
        )
        obs_dict["screen"] = transform(obs_dict["screen"])
        return obs_dict

    source_directory /= game
    dataset = datasets.DatasetFolder(
        root=str(source_directory),
        loader=load_and_transform,
        extensions=Observation.FILE_EXTENSION,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset
