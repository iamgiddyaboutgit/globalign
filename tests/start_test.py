#!/usr/bin/env python3
import pytest

from src.globalign import start

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
    assert start.check_symmetric(test_input) == expected


@pytest.mark.parametrize(
    argnames=("test_input", "expected"),
    argvalues=[
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
    ]
)
def test_check_symmetric_invalid_input(test_input, expected):
    with pytest.raises(expected):
        start.check_symmetric(test_input)


@pytest.mark.parametrize(
    argnames="alphabet, min_len, max_len, seed, expected",
    argvalues=[
        (
            ["A", "C", "T", "G"],
            7,
            10,
            19,
            "GTTCGCA"
        ),
        (
            ["A", "C", "T", "G"],
            5,
            8,
            345,
            "AGACGAC"
        ),
        (
            [""],
            7,
            10,
            19,
            ""
        ),
        (
            ["the", "fat", "cat"],
            7,
            10,
            19,
            "catfatfatfatcatthethe"
        )
    ]
)
def test_draw_random_seq(
    alphabet, 
    min_len, 
    max_len, 
    seed, 
    expected
):
    res = start.draw_random_seq(
        alphabet,
        min_len,
        max_len,
        seed
    ) 

    assert res == expected


@pytest.mark.parametrize(
    argnames="alphabet, min_len, max_len, seed, expected",
    argvalues=[
        (
            [],
            7,
            10,
            19,
            IndexError
        ),
        (
            54646,
            7,
            10,
            19,
            TypeError
        ),
        (
            ["the", "fat", "cat", 9],
            7,
            10,
            19,
            TypeError
        ),
         (
            [1, 0],
            20,
            20,
            19,
            TypeError
        ),
        (
            ["a", "b"],
            7,
            3,
            19,
            ValueError
        ),
        (
            ["a", "b"],
            -7,
            -3,
            19,
            ValueError
        )
    ]
)
def test_draw_random_seq_invalid_input(
    alphabet, 
    min_len, 
    max_len, 
    seed, 
    expected
):
    with pytest.raises(expected):
        start.draw_random_seq(
            alphabet,
            min_len,
            max_len,
            seed
        )