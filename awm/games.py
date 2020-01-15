from functools import partial
from typing import Callable

from gym.wrappers.atari_preprocessing import AtariPreprocessing

from .hyperparams import (
    ControllerParams,
    HyperParams,
    MDNRNNParams,
    ObservationsParams,
    PlayGameParams,
    VAEParams,
)

SUPPORTED_GAMES = {}


class RegisterGame(type):
    def __new__(cls, name, bases, attrs):
        klass = super().__new__(cls, name, bases, attrs)
        # Prevent registration of the base class *GymGame*
        if bases:
            SUPPORTED_GAMES[klass.key] = klass
        return klass


class GymGame(metaclass=RegisterGame):
    key: str
    action_vector_size: int
    hyperparams: HyperParams
    wrapper: Callable
    color_channels: int

    @staticmethod
    def transform_overall_reward(overall_reward):
        raise NotImplementedError()

    @staticmethod
    def transform_action(action):
        # action:
        # 0: steering -1 -> 1
        # 1: gas 0 -> 1
        # 2: break 0 -> 1
        return action[0], (action[1] + 1) / 2, (action[2] + 1) / 2


class CarRacing(GymGame):
    wrapper = None
    key = "CarRacing-v0"
    action_vector_size = 3
    color_channels = 3

    hyperparams = HyperParams(
        observations=ObservationsParams(
            number_of_plays=100,
            # All games will stop after 999 steps - play full games for gathering
            # observations.
            steps_per_play=1000,
            # When gathering observations we must not be "hyperactive" - this is
            # controlled by the action_every_steps parameter
            action_every_steps=40,
        ),
        vae=VAEParams(number_of_epochs=100, no_improvement_threshold=10),
        mdnrnn=MDNRNNParams(number_of_epochs=100, no_improvement_threshold=10),
        controller=ControllerParams(
            reward_threshold=-600, step_limit=0, average_over=1, population_size=5
        ),
        play_game=PlayGameParams(step_limit=0,),
    )

    @staticmethod
    def transform_overall_reward(overall_reward):
        return -overall_reward


class Pong(GymGame):
    wrapper = partial(AtariPreprocessing, grayscale_obs=True)
    key = "Pong-v0"
    action_vector_size = 1
    color_channels = 1

    hyperparams = HyperParams(
        observations=ObservationsParams(
            number_of_plays=100, steps_per_play=0, action_every_steps=1,
        ),
        vae=VAEParams(number_of_epochs=100, no_improvement_threshold=10),
        mdnrnn=MDNRNNParams(number_of_epochs=100, no_improvement_threshold=10),
        controller=ControllerParams(
            reward_threshold=20, step_limit=0, average_over=1, population_size=5
        ),
        play_game=PlayGameParams(step_limit=0,),
    )

    @staticmethod
    def transform_overall_reward(overall_reward):
        return -overall_reward

    @staticmethod
    def transform_action(action):
        return int(round(action[0]))
