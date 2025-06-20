#!/usr/bin/env python3
"""Perform optimal global alignment of two nucleotide \
or amino acid sequences.
"""

import sys
import argparse
from pathlib import Path
import random
from importlib.metadata import version

from globalign.start import (
    validate_and_transform_args,
    get_max_val,
    make_matrix
)
from globalign.conclude import (
    final_cost_to_score,
    AlignmentResults
)


def main():
    usage = "Perform optimal global alignment of two nucleotide \
or amino acid sequences."
    # Create an object to store arguments passed from the command 
    # line.
    parser = argparse.ArgumentParser(
        description=usage
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"{version('globalign')}",
        help="Prints the version and exits."
    )

    parser.add_argument(
        "-i",
        "--input_fasta",
        required=False,
        help="File path to a FASTA file containing two sequences to align.  Do not include if seq_1 and seq_2 are provided.  If the file contains more than 2 sequences, only the first 2 will be used."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output file path to which a file containing the global alignment will be written.  If not provided, then the alignment will be written to stdout."
    ) 

    parser.add_argument(
        "--seq_1", 
        required=False,
        help="First sequence to align.  Do not include if input_fasta is provided."
    ) 

    parser.add_argument(
        "--seq_2", 
        required=False,
        help="Second sequence to align.  Do not include if input_fasta is provided."
    ) 

    parser.add_argument(
        "--scoring_mat_name", 
        required=False,
        choices=["BLOSUM50", "BLOSUM62"],
        help="Either 'BLOSUM50' or 'BLOSUM62'.  Do not include this option if you would like to use a different scoring scheme or if you are aligning nucleotide sequences.  If set, then none of the other options with scores or costs should be set, except for the gap_open options."
    ) 

    parser.add_argument(
        "--scoring_mat_path", 
        required=False,
        help="File path to a custom scoring matrix file.  If set, then none of the other options with scores or costs should be set, except for the gap_open options."
    ) 

    parser.add_argument(
        "--match_score",
        required=False,
        help="Score for a match.  Should be positive.  Only used if scoring_mat is not specified.  If set, then none of the options with costs should be set.  Default: 2."
    ) 

    parser.add_argument(
        "--mismatch_score",
        required=False,
        help="Score for a mismatch.  Should be negative.  Only used if scoring_mat is not specified.  If set, then none of the options with costs should be set.  Default: -3."
    ) 

    parser.add_argument(
        "--mismatch_cost",
        required=False,
        help="Cost for a mismatch.  Should be positive.  If set, then none of the options with scores should be set.  Default: 5."
    ) 

    parser.add_argument(
        "--gap_open_score",
        required=False,
        help="Score for opening a run of gaps.  It is accumulated even for a run with just one gap in it.  Should be non-positive.  Only used if scoring_mat is not specified.  If set, then none of the options with costs should be set.  Default: -4."
    ) 

    parser.add_argument(
        "--gap_open_cost",
        required=False,
        help="Cost for opening a run of gaps.  It is accumulated even for a run with just one gap in it.  Should be non-negative.  If set, then none of the options with scores should be set.  Default: 4."
    ) 

    parser.add_argument(
        "--gap_extension_score",
        required=False,
        help="Score for extending a run of gaps.  It is accumulated even for a run with just one gap in it.  Should be negative.  Only used if scoring_mat is not specified.  If set, then none of the options with costs should be set.  Default: -2."
    ) 

    parser.add_argument(
        "--gap_extension_cost",
        required=False,
        help="Cost for extending a run of gaps.  It is accumulated even for a run with just one gap in it.  Should be positive.  If set, then none of the options with scores should be set.  Default: 3."
    ) 

    cmd_line_args = parser.parse_args()

    # https://stackoverflow.com/a/33637806/8423001
    alignment_results = find_global_alignment(
        **vars(cmd_line_args)
    )
    
    alignment_results.write()

    return None


def find_global_alignment(
    input_fasta: str|Path=None,
    output: str|Path=None,
    seq_1: str=None,
    seq_2: str=None,
    scoring_mat_name: str=None,
    scoring_mat_path: str|Path=None,
    match_score: str|int=None,
    mismatch_score: str|int=None,
    mismatch_cost: str|int=None,
    gap_open_score: str|int=None,
    gap_open_cost: str|int=None,
    gap_extension_score: str|int=None,
    gap_extension_cost: str|int=None
) -> AlignmentResults:
    """
    Args:
        input_fasta: File path to a FASTA file containing two 
            sequences to align.  Do not include if seq_1 and 
            seq_2 are provided.  If the file contains more than 
            2 sequences, only the first 2 will be used.
        output: Output file path to which a file containing 
            the global alignment will be written.  If not 
            provided, then the alignment will be written to 
            stdout.
        seq_1: First sequence to align.  Do not include if 
            input_fasta is provided.
        seq_2: Second sequence to align.  Do not include 
            if input_fasta is provided.
        scoring_mat_name: Either 'BLOSUM50' or 'BLOSUM62'.  
            Do not include this option if you would like 
            to use a different scoring scheme or if you 
            are aligning nucleotide sequences.  If set, 
            then none of the other options with scores 
            or costs should be set, except for the 
            gap_open options.
        scoring_mat_path: File path to a custom scoring 
            matrix file.  If set, then none of the other 
            options with scores or costs should be set, 
            except for the gap_open options.
        match_score: Score for a match.  Should be positive.  
            Only used if scoring_mat is not specified.  
            If set, then none of the options with costs should 
            be set.  Default: 2.
        mismatch_score: Score for a mismatch.  Should be negative.  
            Only used if scoring_mat is not specified.  If set, 
            then none of the options with costs should be set.  
            Default: -3.
        mismatch_cost: Cost for a mismatch.  Should be positive.  
            If set, then none of the options with scores should 
            be set.  Default: 5.
        gap_open_score: Score for opening a run of gaps.  It is 
            accumulated even for a run with just one gap in it.  
            Should be non-positive.  Only used if scoring_mat 
            is not specified.  If set, then none of the options 
            with costs should be set.  Default: -4.
        gap_open_cost: Cost for opening a run of gaps.  It is 
            accumulated even for a run with just one gap in it.  
            Should be non-negative.  If set, then none of the 
            options with scores should be set.  
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment. Note that an alignment like

                    A--CG
                    |   |
                    ATT-G

            incurs the gap_open_cost twice.
            Default: 4.

    Returns:
        AlignmentResults instance with attributes of:
            seq_1_aligned,
            middle_part,
            seq_2_aligned,
            cost,
            score,
            scoring_mat,
            costing_mat,
            gap_open_score,
            gap_open_cost,
            output
    """
    good_args = validate_and_transform_args(
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
    (
        seq_1, 
        seq_2, 
        scoring_mat, 
        costing_mat, 
        gap_open_score, 
        gap_open_cost, 
        output
    ) = good_args
    # Imagine a 3-d parking garage.
    # Movement through this "parking garage"
    # is movement through the alignment graph.
    # The parking garage has 3 levels and we
    # can teleport vertically between levels.
    # On a bird's eye view, we are trying to get
    # from the top left to the bottom right.
    # Progressions that end with you on levels 0, 1, and 2 
    # (from a bird's eye view) are for matches/mismatches,
    # gaps in seq_1, and gaps in seq_2, respectively.
    # For a given bird's eye view position,
    # there are 3 ways that you could have gotten there:
    # from a match/mismatch, from a gap in seq_1,
    # or from a gap in seq_2.
    # This becomes important in the traceback.

    # Create the dynamic programming array (dp_array).
    # Initialize the dp_array.
    max_cost = get_max_val(costing_mat)

    dp_array = make_dp_array(
        seq_1=seq_1,
        seq_2=seq_2,
        costing_mat=costing_mat,
        max_cost=max_cost,
        gap_open_cost=gap_open_cost
    )

    # Loop through the dp_array and write the
    # best costs to get to each position.
    dp_array_forward(
        dp_array=dp_array,
        seq_1=seq_1,
        seq_2=seq_2,
        costing_mat=costing_mat,
        gap_open_cost=gap_open_cost
    )

    # Traceback the dp_array to determine
    # the sequence of moves in reverse
    # order needed to produce an optimal alignment.
    traceback_results = dp_array_backward(
        dp_array=dp_array,
        seq_1=seq_1,
        seq_2=seq_2,
        costing_mat=costing_mat,
        gap_open_cost=gap_open_cost
    )
    (
        seq_1_aligned,
        middle_part,
        seq_2_aligned,
        cost
    ) = traceback_results

    max_score = get_max_val(scoring_mat)

    score = final_cost_to_score(
        cost=cost,
        m=len(seq_1),
        n=len(seq_2),
        max_score=max_score
    )
    return AlignmentResults(
        seq_1_aligned,
        middle_part,
        seq_2_aligned,
        cost,
        score,
        scoring_mat,
        costing_mat,
        gap_open_score,
        gap_open_cost,
        output
    )


def get_next_best_costs(
    dp_array: list[list[list]],
    i: int,
    j: int,
    seq_1: str,
    seq_2: str,
    costing_mat: dict[dict],
    gap_open_cost: int
) -> tuple[int]:
    seq_1_index = i - 1
    seq_2_index = j - 1
    
    # The following are the previous costs
    # and step cost if the current node is in level 0.
    previous_costs_0 = (
        dp_array[i - 1][j - 1][0],
        dp_array[i - 1][j - 1][1],
        dp_array[i - 1][j - 1][2]
    )
    step_cost_0 = costing_mat[seq_1[seq_1_index]][seq_2[seq_2_index]]

    ###############################################
    # The following are the previous costs
    # plus relevant gap_open_costs 
    # and step cost if the current node is in level 1.
    previous_costs_plus_gap_1 = (
        dp_array[i][j - 1][0] + gap_open_cost,
        dp_array[i][j - 1][1],
        dp_array[i][j - 1][2] + gap_open_cost
    )
    step_cost_1 = costing_mat["-"][seq_2[seq_2_index]]

    # The following are the previous costs
    # plus relevant gap_open_costs 
    # and step cost if the current node is in level 2.
    previous_costs_plus_gap_2 = (
        dp_array[i - 1][j][0] + gap_open_cost,
        dp_array[i - 1][j][1] + gap_open_cost,
        dp_array[i - 1][j][2]
    )
    step_cost_2 = costing_mat[seq_1[seq_1_index]]["-"]
    #################################################
    return (
        min(previous_costs_0) + step_cost_0,
        min(previous_costs_plus_gap_1) + step_cost_1,
        min(previous_costs_plus_gap_2) + step_cost_2
    )


def dp_array_forward(
    dp_array:list[list[None]],
    seq_1:str,
    seq_2:str,
    costing_mat:dict[dict],
    gap_open_cost:int|float
):
    """
    Modifies dp_array in-place.
    """
    # Prepare for loop.
    dim_1 = len(seq_1) + 1
    dim_2 = len(seq_2) + 1
    
    for i in range(1, dim_1):
        for j in range(1, dim_2):
            dp_array[i][j] = get_next_best_costs(
                dp_array=dp_array,
                i=i,
                j=j,
                seq_1=seq_1,
                seq_2=seq_2,
                costing_mat=costing_mat,
                gap_open_cost=gap_open_cost
            )

    return None


def dp_array_backward(
    dp_array: list[list[tuple[int]]],
    seq_1: str,
    seq_2: str,
    costing_mat: dict[dict],
    gap_open_cost:int
) -> tuple:
    """
    Traces backward through the dp_array

    to determine which alignment moves are best.

    Returns:
        tuple with entries of:
            seq_1_aligned,
            middle_part,
            seq_2_aligned,
            cost
    """
    
    seq_1_aligned = []
    middle_part = [] 
    seq_2_aligned = []
    
    # Handle the bottom-right bird's eye-view
    # cell of dp_array before the loop.
    dim_1 = len(seq_1) + 1
    dim_2 = len(seq_2) + 1
    # The minimum of this cell
    # is the ultimate cost.
    cost = min(dp_array[dim_1 - 1][dim_2 - 1])
    
    i = dim_1 - 1
    j = dim_2 - 1
    seq_1_index = i - 1
    seq_2_index = j - 1
    costs_to_compare = dp_array[i][j]
    # Find a minimum of the dp_array values compared.
    # Randomly break ties.
    # https://stackoverflow.com/a/53661474/8423001
    cost_ranks = [sorted(costs_to_compare).index(x) for x in costs_to_compare]
    is_match = (seq_1[seq_1_index] == seq_2[seq_2_index])
    # Figure out the move to make in the alignment graph.
    move, delta_i, delta_j, level = cost_ranks_dispatcher(
        cost_ranks=cost_ranks, 
        is_match=is_match
    )

    move_params = dict(
        seq_1 = seq_1,
        seq_2 = seq_2, 
        seq_1_index = seq_1_index,
        seq_2_index = seq_2_index,
        seq_1_aligned = seq_1_aligned,
        middle_part = middle_part,
        seq_2_aligned = seq_2_aligned
    )

    # Make the move in the alignment graph.
    move(**move_params)

    # Prepare indices for going to the next cell.
    i += delta_i
    j += delta_j

    if i == 0 and j == 0:
        seq_1_aligned="".join(seq_1_aligned)
        middle_part="".join(middle_part)
        seq_2_aligned="".join(seq_2_aligned)
        
        return (
            seq_1_aligned,
            middle_part,
            seq_2_aligned,
            cost
        )

    # Prepare for loop.
    max_num_additional_alignment_moves = dim_1 + dim_2 - 2

    for h in range(max_num_additional_alignment_moves):
        seq_1_index = i - 1
        seq_2_index = j - 1
        # Find the dp_array values to compare.
        # This depends on which cell was selected 
        # as the best last time. 
        # Decisions of how to move through the alignment
        # graph are made based on values plus gap stuff
        # and what level was selected in the last
        # iteration.  We need to know which level
        # was selected in the last iteration
        # to figure out the gap stuff.
        costs_to_compare_1 = dp_array[i][j]
        # Based on what the level was in the last iteration,
        # there are different costs_to_add to costs_to_compare_1.
        if level == 0:
            costs_to_add = [
                costing_mat[seq_1[seq_1_index]][seq_2[seq_2_index]]
            ]*3
        elif level == 1:
            costs_to_add = [
                gap_open_cost + costing_mat["-"][seq_2[seq_2_index]],
                costing_mat["-"][seq_2[seq_2_index]],
                gap_open_cost + costing_mat["-"][seq_2[seq_2_index]]
            ]
        else:
            costs_to_add = [
                gap_open_cost + costing_mat["-"][seq_2[seq_2_index]],
                gap_open_cost + costing_mat["-"][seq_2[seq_2_index]],
                costing_mat["-"][seq_2[seq_2_index]]
            ]
        # costs_to_add contains the costs that 
        # would have to be added to get to the cell
        # that we know has the best cost.
        costs_to_compare_2 = [sum(cs) for cs in zip(costs_to_compare_1, costs_to_add)]
        
        # Find a minimum of the dp_array values compared.
        # Randomly break ties.
        # https://stackoverflow.com/a/53661474/8423001
        cost_ranks = [sorted(costs_to_compare_2).index(x) for x in costs_to_compare_2]
        is_match = (seq_1[seq_1_index] == seq_2[seq_2_index])
        # Figure out the move to make in the alignment graph.
        move, delta_i, delta_j, level = cost_ranks_dispatcher(
            cost_ranks=cost_ranks, 
            is_match=is_match
        )

        move_params = dict(
            seq_1 = seq_1,
            seq_2 = seq_2, 
            seq_1_index = seq_1_index,
            seq_2_index = seq_2_index,
            seq_1_aligned = seq_1_aligned,
            middle_part = middle_part,
            seq_2_aligned = seq_2_aligned
        )

        # Make the move in the alignment graph.
        move(**move_params)

        # Prepare indices for going to the next cell.
        i += delta_i
        j += delta_j

        if i == 0 and j == 0:
            break

    seq_1_aligned.reverse()
    seq_2_aligned.reverse()
    middle_part.reverse()

    return (
        "".join(seq_1_aligned),
        "".join(middle_part),
        "".join(seq_2_aligned),
        cost
    )

def cost_ranks_dispatcher(cost_ranks: list|tuple, is_match: bool):
    cost_ranks_with_is_match = (tuple(cost_ranks), is_match)
    # Note that the result of random.choice is "permanent".
    move_dipatch_dict = {
        ((0, 0, 0), True): random.choice((take_match, take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((0, 0, 1), True): take_gap_in_seq_2,
        ((0, 1, 0), True): take_gap_in_seq_1,
        ((1, 0, 0), True): take_match,
        
        ((0, 0, 2), True): random.choice((take_match, take_gap_in_seq_1)),
        ((0, 2, 0), True): random.choice((take_match, take_gap_in_seq_2)),
        ((2, 0, 0), True): random.choice((take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((0, 1, 1), True): take_match,
        ((1, 0, 1), True): take_gap_in_seq_1,
        ((1, 1, 0), True): take_gap_in_seq_2,

        ((0, 2, 2), True): take_match,
        ((2, 0, 2), True): take_gap_in_seq_1,
        ((2, 2, 0), True): take_gap_in_seq_2,
        
        ((0, 1, 2), True): take_match,
        ((1, 0, 2), True): take_gap_in_seq_1,
        ((1, 2, 0), True): take_gap_in_seq_2,
        ((0, 2, 1), True): take_match,
        ((2, 0, 1), True): take_gap_in_seq_1,
        ((2, 1, 0), True): take_gap_in_seq_2,
        
        ((1, 1, 1), True): random.choice((take_match, take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((1, 1, 2), True): random.choice((take_match, take_gap_in_seq_1)),
        ((1, 2, 1), True): random.choice((take_match, take_gap_in_seq_2)),
        ((2, 1, 1), True): random.choice((take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((1, 2, 2), True): take_match,
        ((2, 1, 2), True): take_gap_in_seq_1,
        ((2, 2, 1), True): take_gap_in_seq_2,
        
        ((2, 2, 2), True): random.choice((take_match, take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((0, 0, 0), False): random.choice((take_mismatch, take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((0, 0, 1), False): take_gap_in_seq_2,
        ((0, 1, 0), False): take_gap_in_seq_1,
        ((1, 0, 0), False): take_mismatch,
        
        ((0, 0, 2), False): random.choice((take_mismatch, take_gap_in_seq_1)),
        ((0, 2, 0), False): random.choice((take_mismatch, take_gap_in_seq_2)),
        ((2, 0, 0), False): random.choice((take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((0, 1, 1), False): take_mismatch,
        ((1, 0, 1), False): take_gap_in_seq_1,
        ((1, 1, 0), False): take_gap_in_seq_2,

        ((0, 2, 2), False): take_mismatch,
        ((2, 0, 2), False): take_gap_in_seq_1,
        ((2, 2, 0), False): take_gap_in_seq_2,
        
        ((0, 1, 2), False): take_mismatch,
        ((1, 0, 2), False): take_gap_in_seq_1,
        ((1, 2, 0), False): take_gap_in_seq_2,
        ((0, 2, 1), False): take_mismatch,
        ((2, 0, 1), False): take_gap_in_seq_1,
        ((2, 1, 0), False): take_gap_in_seq_2,
        
        ((1, 1, 1), False): random.choice((take_mismatch, take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((1, 1, 2), False): random.choice((take_mismatch, take_gap_in_seq_1)),
        ((1, 2, 1), False): random.choice((take_mismatch, take_gap_in_seq_2)),
        ((2, 1, 1), False): random.choice((take_gap_in_seq_1, take_gap_in_seq_2)),
        
        ((1, 2, 2), False): take_mismatch,
        ((2, 1, 2), False): take_gap_in_seq_1,
        ((2, 2, 1), False): take_gap_in_seq_2,
        
        ((2, 2, 2), False): random.choice((take_mismatch, take_gap_in_seq_1, take_gap_in_seq_2)),
    }

    # The last value in the tuple of values
    # is the level.
    delta_dispatch_dict = {
        take_match: (-1, -1, 0),
        take_mismatch: (-1, -1, 0),
        take_gap_in_seq_1: (0, -1, 1),
        take_gap_in_seq_2: (-1, 0, 2)
    }

    move = move_dipatch_dict[cost_ranks_with_is_match]
    delta_i, delta_j, level = delta_dispatch_dict[move]
    return (move, delta_i, delta_j, level)


def take_match(
    seq_1:str, 
    seq_2:str, 
    seq_1_index:int, 
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    """Modifies the lists in-place."""
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append("|")
    seq_2_aligned.append(seq_2[seq_2_index])

    return None


def take_mismatch(
    seq_1:str, 
    seq_2:str, 
    seq_1_index:int, 
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    """Modifies the lists in-place."""
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append("*")
    seq_2_aligned.append(seq_2[seq_2_index])

    return None


def take_gap_in_seq_1( 
    seq_1:str,
    seq_2:str, 
    seq_1_index:int,
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    """Modifies the lists in-place."""
    seq_1_aligned.append("-")
    middle_part.append(" ")
    seq_2_aligned.append(seq_2[seq_2_index])

    return None


def take_gap_in_seq_2( 
    seq_1:str,
    seq_2:str, 
    seq_1_index:int,
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    """Modifies the lists in-place."""
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append(" ")
    seq_2_aligned.append("-")

    return None


def make_dp_array(
    seq_1: str, 
    seq_2: str,
    costing_mat: dict[dict],
    max_cost: int,
    gap_open_cost: int|float
    ) -> list[list]:
    # Create the array.
    seq_1_len = len(seq_1)
    seq_2_len = len(seq_2)
    dim_1 = seq_1_len + 1
    dim_2 = seq_2_len + 1

    dp_array = make_matrix(
        num_rows=dim_1,
        num_cols=dim_2,
        fill_val=None
    )
    # Initialize dp_array.
    # To avoid floating point infinities,
    # set some values really high.
    big_num = (max_cost + 1) * max(seq_1_len, seq_2_len)
    dp_array[0][0] = (0, 0, 0)
    try:
        dp_array[0][1] = (
            big_num,
            gap_open_cost + costing_mat["-"][seq_2[0]],
            big_num
        )
    except IndexError:
        return dp_array
    
    try:
        dp_array[1][0] = (
            big_num,
            big_num,
            gap_open_cost + costing_mat[seq_1[0]]["-"]
        )
    except IndexError:
        return dp_array

    # Now that the top-left corner is filled in,
    # the rest of the 0-th row and 0-th column entries
    # can be filled in.
    
    # Fill in the 0-th row.
    for j in range(2, dim_2):
        seq_2_index = j - 1

        dp_array[0][j] = (
            big_num,  # level 0
            dp_array[0][j - 1][1] + costing_mat["-"][seq_2[seq_2_index]],  # level 1
            big_num  # level 2
        )

    # Fill in the 0-th column.
    for i in range(2, dim_1):
        seq_1_index = i - 1

        dp_array[i][0] = (
            big_num,  # level 0
            big_num,  # level 1
            dp_array[i - 1][0][2] + costing_mat[seq_1[seq_1_index]]["-"]  # level 2
        )

    return dp_array


if __name__ == "__main__":
    sys.exit(main())