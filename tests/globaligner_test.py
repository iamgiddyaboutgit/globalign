#!/usr/bin/env python3
import pytest

from src.globalign import globaligner

@pytest.mark.parametrize(
    argnames="dp_array, seq_1, seq_2, costing_mat, gap_open_cost, expected",
    argvalues=[(
        # dp_array
        [
            [(0, 7, 7), (6, 3, 9), (5, 5, 11)],
            [(4, 10, 4), None, None],
            [(10, 13, 7), None, None]
        ],
        # seq_1
        "AG",
        # seq_2
        "GA",
        # costing_mat
        {
            "A": {"A": 0, "G": 3, "-": 3},
            "G": {"A": 3, "G": 0, "-": 3},
            "-": {"A": 2, "G": 2, "-": 0},
        },
        # gap_open_cost
        1,
        # expected
        [
            [(0, 7, 7), (6, 3, 9), (5, 5, 11)],
            [(4, 10, 4), (3, 7, 7), (3, 6, 9)],
            [(10, 13, 7), (4, 10, 7), (6, 7, 7)]
        ]  
    )]
)
def test_dp_array_forward(dp_array, seq_1, seq_2, costing_mat, gap_open_cost, expected):
    globaligner.dp_array_forward(dp_array, seq_1, seq_2, costing_mat, gap_open_cost)
    assert dp_array == expected