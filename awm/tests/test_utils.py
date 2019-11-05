from awm.utils import spread


def test_spread():
    assert spread(10, 1) == [10]
    assert spread(4, 2) == [2, 2]
    assert spread(3, 3) == [1, 1, 1]
    assert spread(2, 3) == [0, 0, 2]
    assert spread(3, 2) == [1, 2]