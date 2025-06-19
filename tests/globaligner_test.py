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


@pytest.mark.parametrize(
    argnames=(
        "input_fasta",
        "output",
        "seq_1",
        "seq_2",
        "scoring_mat_name",
        "scoring_mat_path",
        "match_score",
        "mismatch_score",
        "mismatch_cost",
        "gap_open_score",
        "gap_open_cost",
        "gap_extension_score",
        "gap_extension_cost",
        "expected_score",
        "expected_cost"
    ),
    argvalues=[
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "TT",
            # seq_2
            "TA",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            3,
            # mismatch_score
            -4,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -1,
            # expected_cost
            7
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "TAAAGCTAA",
            # seq_2
            "TAGCTC",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            2,
            # mismatch_score
            -3,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -9,
            # expected_cost
            24
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "TGGATGAGGCTCCACGCACTAA",
            # seq_2
            "GATTGGTGAGGCTCAGCAT",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            2,
            # mismatch_score
            -3,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -15,
            # expected_cost
            56
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "CGGTCTTAGCATATGTTGGCATAC",
            # seq_2
            "ATTAGCATCATAGTGGA",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            2,
            # mismatch_score
            -3,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -21,
            # expected_cost
            62
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "CGGTCTTAGCATATGTTGGCATAC",
            # seq_2
            "ATTAGCATCATAGTGGA",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            4,
            # mismatch_score
            -5,
            # mismatch_cost
            None,
            # gap_open_score
            -3,
            # gap_open_cost
            None,
            # gap_extension_score
            -5,
            # gap_extension_cost
            None,
            # expected_score
            -20,
            # expected_cost
            102
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "GTAGGCGGTC",
            # seq_2
            "CAGCTGC",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            1,
            # mismatch_score
            -2,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -18,
            # expected_cost
            28
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "CTGTACCG",
            # seq_2
            "CGGAACAGTCCGAT",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            1,
            # mismatch_score
            -2,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -18,
            # expected_cost
            26
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "GGAGGACGTT",
            # seq_2
            "GAG",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            1,
            # mismatch_score
            -2,
            # mismatch_cost
            None,
            # gap_open_score
            -5,
            # gap_open_cost
            None,
            # gap_extension_score
            -2,
            # gap_extension_cost
            None,
            # expected_score
            -21,
            # expected_cost
            31
        ),
        (
            # input_fasta
            None,
            # output
            None,
            # seq_1
            "GGAGGACGTT",
            # seq_2
            "GAG",
            # scoring_mat_name
            None,
            # scoring_mat_path
            None,
            # match_score
            "1",
            # mismatch_score
            "-2",
            # mismatch_cost
            None,
            # gap_open_score
            "-5",
            # gap_open_cost
            None,
            # gap_extension_score
            "-2",
            # gap_extension_cost
            None,
            # expected_score
            -21,
            # expected_cost
            31
        ),
    ]
)
def test_find_global_alignment(
    input_fasta,
    output,
    seq_1,
    seq_2,
    scoring_mat_name,
    scoring_mat_path,
    match_score,
    mismatch_score,
    mismatch_cost,
    gap_open_score,
    gap_open_cost,
    gap_extension_score,
    gap_extension_cost,
    expected_score,
    expected_cost
):
    alignment_results = globaligner.find_global_alignment(
        input_fasta,
        output,
        seq_1,
        seq_2,
        scoring_mat_name,
        scoring_mat_path,
        match_score,
        mismatch_score,
        mismatch_cost,
        gap_open_score,
        gap_open_cost,
        gap_extension_score,
        gap_extension_cost
    )

    assert alignment_results.score == expected_score
    assert alignment_results.cost == expected_cost