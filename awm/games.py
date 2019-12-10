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

    def transform_overall_reward(self, overall_reward):
        raise NotImplementedError()


class CarRacing(GymGame):
    key = "CarRacing-v0"
    action_vector_size = 3
    hyperparams = HyperParams(
        # When gathering observations we must not be "hyperactive" - this is
        # controlled by the action_every_steps parameter
        # Most games will not reach 2000 steps in the observation phase
        observations=ObservationsParams(
            number_of_plays=100, steps_per_play=2000, action_every_steps=40,
        ),
        vae=VAEParams(number_of_epochs=100, no_improvement_threshold=10),
        mdnrnn=MDNRNNParams(number_of_epochs=100, no_improvement_threshold=5),
        controller=ControllerParams(
            reward_threshold=600, step_limit=0, average_over=5, population_size=5
        ),
        play_game=PlayGameParams(step_limit=0,),
    )

    def transform_overall_reward(self, overall_reward):
        return 1000 - overall_reward
