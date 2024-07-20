#!/usr/bin/env python3
"""Perform optimal global alignment of two nucleotide \
or amino acid sequences using the Needleman-Wunsch algorithm.

References:
https://web.stanford.edu/class/cs262/archives/presentations/lecture3.pdf
https://ocw.mit.edu/courses/6-096-algorithms-for-computational-biology-spring-2005/01f55f348ea1e95f7015bd1b40586012_lecture5.pdf
"""

import sys

# The cost for a gap just to exist.
# This cost should be non-negative.
# This cost can be incurred multiple times
# if there are multiple runs of gaps in the
# alignment.
GAP_EXISTENCE_COST = 0

def main():
    # Read in FASTA file.
    # Verify FASTA file is in the correct format.
    # Extract sequences.
    # Verify sequences are formatted correctly.
    # Handle sequences of 0 length.
    # Check that the product of the lengths of the sequences does
    # not exceed 400_000_000. If it does, then error.
    # Read in scoring matrix file.
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

def align():
    ...



def moves_to_result(moves:list) -> tuple[str]:
    """Given the 'moves' performed in the
    'game' of the alignment of the two sequences,

    """

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