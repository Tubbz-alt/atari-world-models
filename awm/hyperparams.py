from dataclasses import dataclass


@dataclass
class ObservationsParams:
    number_of_plays: int
    steps_per_play: int
    action_every_steps: int


@dataclass
class VAEParams:
    number_of_epochs: int
    no_improvement_threshold: int


@dataclass
class MDNRNNParams:
    number_of_epochs: int


@dataclass
class ControllerParams:
    reward_threshold: int
    step_limit: int


@dataclass
class PlayGameParams:
    step_limit: int


@dataclass
class HyperParams:
    observations: ObservationsParams
    vae: VAEParams
    mdnrnn: MDNRNNParams
    controller: ControllerParams
    play_game: PlayGameParams