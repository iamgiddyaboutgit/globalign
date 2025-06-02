#!/usr/bin/env python3
import pytest

from src.globalign import globalign as ga

def test_check_symmetric():
    assert ga.check_symmetric(
        {
            "a": {"a": 4, "b": 3},
            "b": {"a": 3, "b": 4}
        }
    )

    assert ga.check_symmetric(
        {
            "a": {"a": 4, "b": 3, "c": 0},
            "b": {"a": 3, "b": 4, "c": 7},
            "c": {"a": 0, "b": 7, "c": 1}
        }
    )

    assert not ga.check_symmetric(
        {
            "a": {"a": 4, "b": 3, "c": 0},
            "b": {"a": 3, "b": 4, "c": 7},
            "c": {"a": 0, "b": 17, "c": 1}
        }
    )

    assert not ga.check_symmetric(
        {
            "a": {"a": 4, "b": 3, "c": 0},
            "b": {"a": 3, "b": 4, "c": 7},
            "d": {"a": 0, "b": 7, "c": 1}
        }
    )

    with pytest.raises(AttributeError):
        ga.check_symmetric(0)

    with pytest.raises(AttributeError):
        ga.check_symmetric(None)

    with pytest.raises(AttributeError):
        ga.check_symmetric([[1, 4], [4, 1]])