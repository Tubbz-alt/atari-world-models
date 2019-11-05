import argparse
from argparse import RawDescriptionHelpFormatter

from . import SUPPORTED_GAMES, VERSION, observations
from .observations import gather_observations_pooled
from .vae import train_vae


def parse():
    description = """
awm - {}

Atari-World-Models - system to train a world-model style NN to play
Atari games.
    """.format(
        ".".join(map(str, VERSION))
    )
    parser = argparse.ArgumentParser(
        "awm", description=description, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("game", help="Name of game", choices=SUPPORTED_GAMES)
    subparsers = parser.add_subparsers(
        dest="subcommand", title="Subcommands", description="", help="",
    )
    subparsers.required = True

    observations_parser = subparsers.add_parser(
        "gather-observations",
        help="Gather random observations",
        description="Gather random observations",
    )
    observations_parser.add_argument(
        "--show-screen",
        action="store_true",
        default=observations.SHOW_SCREEN,
        help="Show the gym screen when playing (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--cpus-to-use",
        type=int,
        default=observations.CPUS_TO_USE,
        help="CPUs to use in gathering observations (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--target-directory",
        default=observations.TARGET_DIRECTORY,
        help="Path where the images should be saved (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--number-of-episodes",
        type=int,
        default=observations.NUMBER_OF_EPISODES,
        help="Number of episodes to play (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=observations.STEPS_PER_EPISODE,
        help="Steps per episode to take (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--action-every-steps",
        type=int,
        default=observations.ACTION_EVERY_STEPS,
        metavar="N",
        help="Take an action every N steps (default: %(default)s)",
    )
    observations_parser.set_defaults(func=gather_observations_pooled)

    train_vae_parser = subparsers.add_parser(
        "train-vae", help="Train the VAE", description="Train the VAE"
    )
    train_vae_parser.add_argument(
        "--source-directory",
        default=observations.TARGET_DIRECTORY,
        help="Path to load the images from (default: %(default)s)",
    )
    train_vae_parser.set_defaults(func=train_vae)

    args = vars(parser.parse_args())

    # Remove keys not mapping directly to a kwarg
    func = args.pop("func")
    game = args.pop("game")
    del args["subcommand"]

    func(game, **args)


parse()
