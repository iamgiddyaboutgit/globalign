#!/usr/bin/env python3

import math
from pathlib import Path
from typing import NamedTuple
import sys

class AlignmentResults(NamedTuple):
    seq_1_aligned: str
    middle_part: str
    seq_2_aligned: str
    cost: int
    score: int
    scoring_mat: dict[dict]
    costing_mat: dict[dict]
    gap_open_score: int
    gap_open_cost: int
    output: Path


def final_cost_to_score(
    cost:int|float, 
    m:int,
    n:int,
    max_score:int|float,
    delta_d:int|float=None, 
    delta_i:int|float=None
) -> int|float:
    """https://curiouscoding.nl/posts/alignment-scores-transform/

    https://www.biorxiv.org/content/10.1101/2022.01.12.476087v1.full.pdf

    Args:
        m: length of seq_1
        n: length of seq_2
        max_score: A maximum score in the original
            scoring matrix.
    """
    b = max_score
    if delta_d is None:
        delta_d = math.floor(b/2)
    if delta_i is None:
        delta_i = math.ceil(b/2)
    return n*delta_d + m*delta_i - cost

def final_score_to_cost(
    score:int|float, 
    m:int,
    n:int,
    max_score:int|float,
    delta_d:int|float=None, 
    delta_i:int|float=None
) -> int|float:
    """https://curiouscoding.nl/posts/alignment-scores-transform/

    https://www.biorxiv.org/content/10.1101/2022.01.12.476087v1.full.pdf

    Args:
        score: The conventional score for the alignment
            using some conventional scoring scheme.
        max_score: A maximum score in the original
            scoring matrix.
    """
    b = max_score
    if delta_d is None:
        delta_d = math.floor(b/2)
    if delta_i is None:
        delta_i = math.ceil(b/2)
    return -score + n*delta_d + m*delta_i 

def print_nested_list_aligned(nested_list: list[list[int|float|str]]):
    """Pretty-prints a nested list.
    
    Args:
        nested_list: Let's call each entry in nested_list 
            a 'row'.  Each 'row' is a list of the same length.
    """
    # Determine how wide each column should be
    # based on the length of the string representation
    # of each cell.
    widths = []
    # Because we assume that each entry in nested_list
    # is the same length, we can get the number of columns
    # from the number of entries in the 0-th row
    # of nested_list.
    num_cols = len(nested_list[0])

    # Loop through the "columns" of the nested list.
    for j in range(num_cols):
        # If each cell in the j-th "column" of the nested list,
        # is given a string representation, what is the 
        # length of the longest representation?
        width_required = 0
        for row in nested_list:
            # Is the length of the string representation of the 
            # current cell longer than the longest such 
            # representation found so far for column j?
            # If it is, then save it and check the next cell
            # in column j.
            width_required = max(width_required, len(str(row[j])))
        widths.append(width_required)

    # Print the numbers with formatting
    for row in nested_list:
        row_2 = ""
        for j, cell in enumerate(row):
            # The format specification of :>{width}
            # right-aligns the string within the width.
            row_2 += f"{cell:>{widths[j] + 1}}"
        print(row_2)
    
    return None


def print_alignment(
    alignment_results: AlignmentResults,
    desc_1: str="seq_1", 
    desc_2: str="seq_2", 
    chars_per_line: int=70
) -> None:
    print(desc_1)
    print(desc_2)
    print("")

    # Handle long alignments with proper line breaking.
    alignment_len = len(alignment_results.middle_part)
    num_sets_needed = math.ceil(alignment_len / chars_per_line)
    
    # Prep for loop
    lower = 0
    if num_sets_needed == 1:
        upper = alignment_len
    else:   
        upper = chars_per_line

    for u in range(num_sets_needed):
        # Loop body
        print(alignment_results.seq_1_aligned[lower:upper])
        print(alignment_results.middle_part[lower:upper])
        print(alignment_results.seq_2_aligned[lower:upper])
        # Prep for next iteration
        print("")
        lower = upper
        upper = lower + chars_per_line
    
    return None

    
def write_alignment(
    alignment_results: AlignmentResults,
    desc_1: str="seq_1", 
    desc_2: str="seq_2", 
    chars_per_line: int=70,
    output: Path|str=None
) -> None:
    """
    Args:
        output: The path to a file to which the results will be written.
            If output is specified, then it overwrites
            alignment_results.output; otherwise, alignment_results.output
            will be used.
            Use the string "stdout", to write to sys.stdout. 
            If alignment_results.output
            is None, then the results will be written to sys.stdout.
    """
    if (output is None and alignment_results.output is None) or output == "stdout":
        print_alignment(
            alignment_results=alignment_results,
            desc_1=desc_1,
            desc_2=desc_2,
            chars_per_line=chars_per_line
        )
        return None
    elif output is None and alignment_results.output is not None:
        file = alignment_results.output
    else:
        file = output
    
    # Handle long alignments with proper line breaking.
    alignment_len = len(alignment_results.middle_part)
    num_sets_needed = math.ceil(alignment_len / chars_per_line)
    
    # Prep for loop
    lower = 0
    if num_sets_needed == 1:
        upper = alignment_len
    else:   
        upper = chars_per_line

    with open(file=file, mode="w+") as f:
        print(desc_1, file=f)
        print(desc_2, file=f)
        print("", file=f)

        for u in range(num_sets_needed):
            # Loop body
            print(alignment_results.seq_1_aligned[lower:upper], file=f)
            print(alignment_results.middle_part[lower:upper], file=f)
            print(alignment_results.seq_2_aligned[lower:upper], file=f)
            # Prep for next iteration
            print("", file=f)
            lower = upper
            upper = lower + chars_per_line
    
    return None