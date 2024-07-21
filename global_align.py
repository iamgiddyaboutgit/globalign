#!/usr/bin/env python3
"""Perform optimal global alignment of two nucleotide \
or amino acid sequences using the Needleman-Wunsch algorithm.

References:
https://web.stanford.edu/class/cs262/archives/presentations/lecture3.pdf
https://ocw.mit.edu/courses/6-096-algorithms-for-computational-biology-spring-2005/01f55f348ea1e95f7015bd1b40586012_lecture5.pdf
"""

import sys
import argparse
from pathlib import Path

# The cost for a gap just to exist.
# This cost should be non-negative.
# This cost can be incurred multiple times
# if there are multiple runs of gaps in the
# alignment.
GAP_EXISTENCE_COST = 0

def main():
    usage = "Perform optimal global alignment of two nucleotide \
or amino acid sequences using the Needleman-Wunsch algorithm."
    # Create an object to store arguments passed from the command 
    # line.
    parser = argparse.ArgumentParser(description=usage)
    # The nargs="+" here allows all command-line args present 
    # to be gathered into a list. Additionally, an error message 
    # will be generated if there wasn’t at least one command-line 
    # argument present.
    parser.add_argument(
        "-s", 
        required=True,
        help="file path to scoring matrix text file"
    ) 

    cmd_line_args = parser.parse_args()

    path_to_score_matrix_file = Path(cmd_line_args.s)
    
    # Read in FASTA file.
    # Verify FASTA file is in the correct format.
    # Extract sequences.
    # Verify sequences are formatted correctly.
    # Handle sequences of 0 length.
    # Check that the product of the lengths of the sequences does
    # not exceed 400_000_000. If it does, then error.
    # Read in scoring matrix file.
    read_scoring_mat(scoring_mat_path=path_to_score_matrix_file)
    # Verify format of scoring matrix file.
    # Get the data from the scoring matrix into a nested dictionary
    # with codes for the letters as keys.
    # Check that the scoring matrix is symmetric.
    # For each row, the entry on the main diagonal
    # should be greater than or equal to the other entries in the row.

    # Determine whether we are aligning amino acid residues
    # or nucleotides by checking both sequences and the
    # scoring matrix.  If there's a mismatch, then 
    # raise an exception.  In other words, the sequences
    # should only contain letters present in the scoring matrix.

    # Perform the alignment, insert gaps, and compute the score.

    # Write the outputs to a file.


    ...

def read_scoring_mat(scoring_mat_path:Path) -> dict[dict]:
    """Read in scoring matrix."""
    if not scoring_mat_path.is_file():
        raise FileNotFoundError("scoring_mat_path does not point to a valid file.")
    
    with scoring_mat_path.open() as f:
        header = f.readline()
        letters = header.upper().split()
        scoring_mat = dict.fromkeys(letters)

        for line in f:
            split_line = line.split()

            outer_dict_letter = split_line[0]
            # Make inner dict for this line's outer_dict_letter.
            scoring_mat[outer_dict_letter] = dict.fromkeys(letters)
            # prep for loop
            letter_id = 0
            for inner_dict_letter in letters:
                # prep for iteration
                letter_id += 1
                # loop body
                inner_dict_letter_2 = inner_dict_letter.upper()
                # Get the score for outer_dict_letter paired 
                # with inner_dict_letter.
                score = int(split_line[letter_id])
                
                # Place values into inner dict for the current inner_dict_letter.
                scoring_mat[outer_dict_letter][inner_dict_letter_2] = score
             
    return scoring_mat

def align(
    seq_1:str, 
    seq_2:str, 
    scoring_mat:dict[dict], 
    gap_existence_cost:int
) -> tuple[str, str, str, int]:
    m = len(seq_1)
    n = len(seq_2)
    dynamic_prog_num_rows = m + 1
    dynamic_prog_num_cols = n + 1

    # Initialize matrices to hold the current best scores
    # for different alignments assuming that a certain move
    # was the last move.
    # 
    # partial_A_mat[i][j] holds the best scores for when seq_2[j]
    # aligns with a new gap (or another gap in a run of gaps) in seq_1.
    #
    # partial_B_mat[i][j] holds the best scores for when seq_1[i]
    # aligns with a new gap (or another gap in a run of gaps) in seq_2.
    #
    # partial_C_mat[i][j] holds the best scores for when seq_1[i]
    # aligns with seq_2[j].
    # 
    # To be find the best score up to a certain point, we consider
    # the max(partial_A_mat[i][j], partial_B_mat[i][j], partial_C_mat[i][j]).
    partial_A_mat, partial_B_mat, partial_C_mat = (init_partial_dynamic_prog_matrix(
        gap_existence_cost=gap_existence_cost,
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        dynamic_prog_num_cols=dynamic_prog_num_cols
    ) for u in range(3)) 

    # Go one row at a time through partial_A_mat, partial_B_mat, and 
    # partial_C_mat (starting at row index 1, col index 1 and always 
    # skipping col index 0).  Simultaneously, fill out the 
    # best_paths_mat.  (Note that we need to save the entirety of 
    # the best_paths_mat.)  We only need to keep two rows each of 
    # partial_A_mat, partial_B_mat, and partial_C_mat in memory 
    # at a time.
    best_paths_mat = [[0]*dynamic_prog_num_cols for i in range(dynamic_prog_num_rows)]

    # Get 1's in the beginning of each row.
    for i in range(1, dynamic_prog_num_rows):
        best_paths_mat[i][0] = 1

    # Pre loop
    i = 1
    partial_mat_prev_row_id = 0
    partial_mat_cur_row_id = 1

    for j in range(1, dynamic_prog_num_cols):
        # prep for this iteration
        seq_1_index = i - 1
        seq_2_index = j - 1

        # body of loop
        # Consider partial_A_mat
        partial_A_mat[partial_mat_cur_row_id][j] = max(
            partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
            partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        )

        # Consider partial_B_mat
        partial_B_mat[partial_mat_cur_row_id][j] = max(
            partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]],
            partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        )

        # Consider partial_C_mat
        prev_best = max(
            partial_A_mat[partial_mat_prev_row_id][j - 1],
            partial_B_mat[partial_mat_prev_row_id][j - 1],
            partial_C_mat[partial_mat_prev_row_id][j - 1]
        )
        partial_C_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
        
        # Choose the best move.
        possible_new_scores = [
            partial_A_mat[partial_mat_cur_row_id][j],
            partial_B_mat[partial_mat_cur_row_id][j],
            partial_C_mat[partial_mat_cur_row_id][j],
        ]
        max_possible_new_score = max(possible_new_scores)
        
        best_type_of_path = possible_new_scores.index(max_possible_new_score)
        
        best_paths_mat[i][j] = best_type_of_path

    for i in range(2, dynamic_prog_num_rows):
        # Prep for a new row iteration.
        # https://stackoverflow.com/a/14836456
        # Do some swapping.
        partial_mat_prev_row_id, partial_mat_cur_row_id = partial_mat_cur_row_id, partial_mat_prev_row_id
        # Update the 0th columns based on how gaps are penalized.
        partial_A_mat[partial_mat_cur_row_id][0] = partial_A_mat[partial_mat_prev_row_id][0] + scoring_mat[seq_1[seq_1_index]]["-"]
        partial_B_mat[partial_mat_cur_row_id][0] = partial_A_mat[partial_mat_prev_row_id][0] + scoring_mat[seq_1[seq_1_index]]["-"]
        partial_C_mat[partial_mat_cur_row_id][0] = partial_A_mat[partial_mat_prev_row_id][0] + scoring_mat[seq_1[seq_1_index]]["-"]
        
        for j in range(1, dynamic_prog_num_cols):
            # prep for this iteration
            seq_1_index = i - 1
            seq_2_index = j - 1

            # body of loop
            # Consider partial_A_mat
            partial_A_mat[partial_mat_cur_row_id][j] = max(
                partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
                partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
            )

            # Consider partial_B_mat
            partial_B_mat[partial_mat_cur_row_id][j] = max(
                partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]],
                partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
            )

            # Consider partial_C_mat
            prev_best = max(
                partial_A_mat[partial_mat_prev_row_id][j - 1],
                partial_B_mat[partial_mat_prev_row_id][j - 1],
                partial_C_mat[partial_mat_prev_row_id][j - 1]
            )
            partial_C_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
            
            # Choose the best move.
            possible_new_scores = [
                partial_A_mat[partial_mat_cur_row_id][j],
                partial_B_mat[partial_mat_cur_row_id][j],
                partial_C_mat[partial_mat_cur_row_id][j],
            ]
            max_possible_new_score = max(possible_new_scores)
            
            best_type_of_path = possible_new_scores.index(max_possible_new_score)
            
            best_paths_mat[i][j] = best_type_of_path

    score = max_possible_new_score

    # traceback
    # Prepare for loop.
    seq_1_aligned = []
    seq_2_aligned = []
    middle_part = []

    num_alignment_moves = max(dynamic_prog_num_rows, dynamic_prog_num_cols) - 1

    # Start at the bottom-right.
    seq_1_index = m - 1
    seq_2_index = n - 1

    for w in range(num_alignment_moves):
        # Prep for this iteration.
        # Because of the initial row and column in
        # best_paths_mat that doesn't align with
        # any parts of the two sequence, the indices
        # are off by one.
        best_paths_mat_row_index = seq_1_index + 1
        best_paths_mat_col_index = seq_2_index + 1

        path_indicator = best_paths_mat[best_paths_mat_row_index][best_paths_mat_col_index]

        if path_indicator == 0:
            middle_part.append(" ")
            seq_1_aligned.append("-")
            seq_2_aligned.append(seq_2[seq_2_index])
            seq_2_index -= 1
        elif path_indicator == 1:
            middle_part.append(" ")
            seq_1_aligned.append(seq_1[seq_1_index])
            seq_1_index -= 1
            seq_2_aligned.append("-")
        else:
            seq_1_letter = seq_1[seq_1_index]
            seq_2_letter = seq_2[seq_2_index]
            if seq_1_letter == seq_2_letter:
                # There was a match.
                middle_part.append("|")
            else:
                # There was not a match.
                middle_part.append("*")

            seq_1_aligned.append(seq_1[seq_1_index])
            seq_1_index -= 1
            seq_2_aligned.append(seq_2[seq_2_index])
            seq_2_index -= 1

    seq_1_aligned.reverse()
    middle_part.reverse()
    seq_2_aligned.reverse()

    seq_1_aligned_out = "".join(seq_1_aligned)
    middle_part_out = "".join(middle_part)
    seq_2_aligned_out = "".join(seq_2_aligned)

    return (
        seq_1_aligned_out,
        middle_part_out,
        seq_2_aligned_out,
        score
    )


def init_partial_dynamic_prog_matrix(
    gap_existence_cost:int, 
    seq_1:str,
    seq_2:str,
    scoring_mat:dict[dict], 
    dynamic_prog_num_cols:int
) -> list[list]:
    
    # https://www.freecodecamp.org/news/list-within-a-list-in-python-initialize-a-nested-list/
    mat = [
        [0]*(dynamic_prog_num_cols) for i in range(2)
    ]
    # Take care of initialization with gap scores.
  
    # Loop prep
    # Start in column 1
    j = 1
    gap_score_cum_sum = 0

    seq_2_index = j - 1
    cur_gap_score = -gap_existence_cost + scoring_mat["-"][seq_2[seq_2_index]]
    mat[0][j] = cur_gap_score

    for j in range(2, dynamic_prog_num_cols):
        # Prep for this iteration
        # The sequence indices are always one behind
        # the row/column indices.
        seq_2_index = j - 1
        gap_score_cum_sum += cur_gap_score

        # body of loop
        cur_gap_score = cur_gap_score + scoring_mat["-"][seq_2[seq_2_index]]
        mat[0][j] = cur_gap_score

    mat[1][0] = -gap_existence_cost + scoring_mat[seq_1[0]]["-"]
    return mat


if __name__ == "__main__":
    sys.exit(main())