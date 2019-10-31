import argparse
from argparse import RawDescriptionHelpFormatter

from . import VERSION, observations
from .observations import gather_observations
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
    parser.add_argument("game", help="Name of game")
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
        "--number_of_episodes",
        default=observations.NUMBER_OF_EPISODES,
        help="Number of episodes to play (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--steps-per-episode",
        default=observations.STEPS_PER_EPISODE,
        help="Steps per episode to take (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--action-every-steps",
        default=observations.ACTION_EVERY_STEPS,
        metavar="N",
        help="Take an action every N steps (default: %(default)s)",
    )
    observations_parser.set_defaults(func=gather_observations)

    train_vae_parser = subparsers.add_parser(
        "train-vae", help="Train the VAE", description="Train the VAE"
    )
    train_vae_parser.set_defaults(func=train_vae)

    args = vars(parser.parse_args())

    # Remove keys not mapping directly to a kwarg
    func = args.pop("func")
    game = args.pop("game")
    del args["subcommand"]

    func(game, **args)


parse()
