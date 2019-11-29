import subprocess


def test_commandline_args():
    """ This is just a small sanity check, that the module does not immediately
    throw an error when called on the commandline.
    """

    def run(args):
        return subprocess.run(
            ["python", "-m", "awm"] + args,
            encoding="ascii",
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )

    result = run([])
    assert "arguments are required: game, subcommand" in result.stdout

    result = run(["INVALID-GAME"])
    assert "argument game: invalid choice" in result.stdout

    result = run(["CarRacing-v0", "invalid-subcommand"])
    assert "argument subcommand: invalid choice" in result.stdout
