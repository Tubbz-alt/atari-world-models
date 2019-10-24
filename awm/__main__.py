import argparse

from . import VERSION


def parse():
    description = "awm - {}".format(".".join(map(str, VERSION)))
    parser = argparse.ArgumentParser(description=description)
    args = parser.parse_args()


parse()
