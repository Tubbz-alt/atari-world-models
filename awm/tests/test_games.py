from ..games import SUPPORTED_GAMES, GymGame


def test_registry_works():
    assert "some-cool-game" not in SUPPORTED_GAMES

    class SomeCoolGame(GymGame):
        key = "some-cool-game"

    # SomeCoolGame should have autoregistered
    assert "some-cool-game" in SUPPORTED_GAMES

    # Be nice - clean up
    del SUPPORTED_GAMES["some-cool-game"]
