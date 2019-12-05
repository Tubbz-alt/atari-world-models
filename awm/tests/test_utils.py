from dataclasses import dataclass

from awm.utils import merge_args_with_hyperparams, spread


def test_spread():
    assert spread(10, 1) == [10]
    assert spread(4, 2) == [2, 2]
    assert spread(3, 3) == [1, 1, 1]
    assert spread(2, 3) == [1, 1]  # Note: Does not return [1, 1, 0]
    assert spread(3, 2) == [2, 1]


def test_merge_args_with_hyperparams():
    args = {"hyper_A": None, "hyper_B": "override-B", "extra": "extra"}

    @dataclass
    class H:
        hyper_A: str
        hyper_B: str

    hyperparams = H(hyper_A="A", hyper_B="B")

    actual = merge_args_with_hyperparams(args, hyperparams)
    expected = {"hyper_A": "A", "hyper_B": "override-B", "extra": "extra"}
    assert actual == expected
