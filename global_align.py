#!/usr/bin/env python3
"""Perform optimal global alignment of two nucleotide \
or amino acid sequences using the Needleman-Wunsch algorithm.
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

if __name__ == "__main__":
    sys.exit(main())