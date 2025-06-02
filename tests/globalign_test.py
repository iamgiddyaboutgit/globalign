#!/usr/bin/env python3
import pytest

from src.globalign import globalign as ga

@pytest.mark.parametrize(
    argnames=("test_input", "expected"),
    argvalues=(
        (
            {
                "a": {"a": 4, "b": 3},
                "b": {"a": 3, "b": 4}
            },
            True
        ),
        (
            {
                "a": {"a": 4, "b": 3, "c": 0},
                "b": {"a": 3, "b": 4, "c": 7},
                "c": {"a": 0, "b": 7, "c": 1}
            },
            True
        ),
        (
            {
                "a": {"a": 4, "b": 3, "c": 0},
                "b": {"a": 3, "b": 4, "c": 7},
                "c": {"a": 0, "b": 17, "c": 1}
            },
            False
        ),
        (
            {
                "a": {"a": 4, "b": 3, "c": 0},
                "b": {"a": 3, "b": 4, "c": 7},
                "d": {"a": 0, "b": 7, "c": 1}
            },
            False
        )
    )
)
def test_check_symmetric_valid_input(test_input, expected):
    assert ga.check_symmetric(test_input) == expected


@pytest.mark.parametrize(
    argnames=("test_input", "expected"),
    argvalues=(
        (
            0,
            AttributeError
        ),
        (
            None,
            AttributeError
        ),
        (
            [[1, 4], [4, 1]],
            AttributeError
        ),
    )
)
def test_check_symmetric_invalid_input(test_input, expected):
    with pytest.raises(expected):
        ga.check_symmetric(test_input)

