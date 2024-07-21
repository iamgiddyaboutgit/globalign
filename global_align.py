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
        help="File path to a scoring matrix text file."
    ) 

    parser.add_argument(
        "-i", 
        required=True,
        help="File path to a FASTA file containing two sequences to align."
    ) 

    parser.add_argument(
        "-g", 
        required=False,
        default=0,
        help="Cost for opening a run of gaps. Should be non-negative."
    ) 

    cmd_line_args = parser.parse_args()

    path_to_score_matrix_file = Path(cmd_line_args.s)
    path_to_fasta_file = Path(cmd_line_args.i)
    gap_existence_cost = int(cmd_line_args.g)
    
    # Read in descriptions and sequences from FASTA file.
    counter = 0
    for desc_and_seq in read_seq_from_fasta(fasta_path=path_to_fasta_file):
        counter += 1
        if counter == 1:
            desc_1, seq_1 = desc_and_seq
        elif counter == 2:
            desc_2, seq_2 = desc_and_seq
        else:
            break
   
    print("seq_2")
    print(seq_2)
    # Verify FASTA file is in the correct format.
    # Extract sequences.
    # Verify sequences are formatted correctly.
    # Handle sequences of 0 length.
    # Check that the product of the lengths of the sequences does
    # not exceed 400_000_000. If it does, then error.
    # Read in scoring matrix file.
    scoring_mat = read_scoring_mat(scoring_mat_path=path_to_score_matrix_file)
    
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
    scoring_mat_letters = scoring_mat.keys()
    if not all([letter in scoring_mat_letters for letter in seq_1]):
        raise RuntimeError("There were letters in seq_1 not present in scoring_mat.")
    if not all([letter in scoring_mat_letters for letter in seq_2]):
        raise RuntimeError("There were letters in seq_2 not present in scoring_mat.")
    # Perform the alignment, insert gaps, and compute the score.
    alignment = align(
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        gap_existence_cost=gap_existence_cost
    )
    print_alignment(
        desc_1=desc_1,
        desc_2=desc_2,
        alignment=alignment
    )
   
    # Write the outputs to a file.


    ...

def read_scoring_mat(scoring_mat_path:Path) -> dict[dict]:
    """Read in scoring matrix."""
    if not scoring_mat_path.is_file():
        raise FileNotFoundError("scoring_mat_path does not point to a valid file.")
    
    with scoring_mat_path.open() as f:
        header = f.readline()
        letters = header.upper().split()
        # Check that we do have single characters in letters.
        if not all([len(letter) == 1 for letter in letters]):
            raise RuntimeError("The header row did not have single letters spaced apart.")
        scoring_mat = dict.fromkeys(letters)

        # Prep for loop
        outer_dict_letter_id = -1
        for line in f:
            # Prep for this iteration
            outer_dict_letter_id += 1
            # Body of loop
            split_line = line.split()

            outer_dict_letter = split_line[0]
            # Check that the outer_dict_letter was also 
            # present in the header in the same
            # relative position.
            if not (outer_dict_letter == letters[outer_dict_letter_id]):
                raise RuntimeError("Row headers do not match column headers.")

            # Make inner dict for this line's outer_dict_letter.
            scoring_mat[outer_dict_letter] = dict.fromkeys(letters)
            # prep for loop
            inner_dict_letter_id = 0
            for inner_dict_letter in letters:
                # prep for iteration
                inner_dict_letter_id += 1
                # loop body
                inner_dict_letter_2 = inner_dict_letter.upper()
                # Get the score for outer_dict_letter paired 
                # with inner_dict_letter.
                score = int(split_line[inner_dict_letter_id])
                
                # Place values into inner dict for the current inner_dict_letter.
                scoring_mat[outer_dict_letter][inner_dict_letter_2] = score
             
    return scoring_mat

def read_seq_from_fasta(fasta_path:Path):
    """Read in a FASTA file. 

    Yields:
        2-tuples where the 0th element is the description
        and the 1st element is the sequence
    
    See: NCBI FASTA specification
    https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
    """
    if not fasta_path.is_file():
        raise FileNotFoundError("fasta_path does not point to a valid file.")
    
    
    with fasta_path.open() as f:
        desc_and_seq_complete = False
        seq_list = []
        line = f.readline()
        line_stripped = line.strip()

        if not line_stripped.startswith(">"):
            raise RuntimeError("Invalid FASTA format. Expected the first line to start with '>'.")
            
        desc = line_stripped

        for line in f:
            line_stripped = line.strip()

            if line_stripped.startswith(">"):
                # We have reached a description
                # other than the first one.
                # We are ready to yield.
                seq = "".join(seq_list)
                if not (len(seq) > 0):
                    raise RuntimeError("Empty sequence detected in FASTA.")
                yield (desc, seq)
                # Prepare for the next yield
                # with the description we just
                # found.
                desc = line_stripped
                # Clear seq_list for the next seq.
                seq_list.clear()
            elif len(line_stripped) > 0:
                # Append the sequence on the line_stripped.
                seq_list.append(line_stripped)

        # We have reached the end of the file.
        # We are ready to yield.
        seq = "".join(seq_list)
        if not (len(seq) > 0):
            raise RuntimeError("Empty sequence detected in FASTA.")
        
        yield (desc, seq)

        
    # with fasta_path.open() as f:
    #     # Prep for loop
    #     desc = None
    #     seq_list = []

    #     # Get the first desc.
    #     line = f.readline()
    #     line_stripped = line.strip()

    #     is_desc = line_stripped.startswith(">")
    #     if not is_desc:
    #         raise RuntimeError("Invalid FASTA format. Expected the first line to start with '>'.")
        
    #     desc = line_stripped
  
    #     for line in f:
    #         line_stripped = line.strip()
    #         is_desc = line_stripped.startswith(">")
    #         is_seq = (not is_desc) and (line_stripped != "")
            
    #         if is_seq:
    #             # Append what is there and then
    #             # go to the next line to possibly
    #             # append more.
    #             seq_list.append(line_stripped)
    #             continue
    #         elif is_desc:
    #             desc = line_stripped
    #             seq = "".join(seq_list)
    #             yield (desc, seq)

    #     # We have consumed the entire FASTA file.        
    #     seq = "".join(seq_list)
    #     yield (desc, seq)
        
        ###########

        #     is_seq = (not is_desc) and line_stripped.isalpha() 
            
        #     if is_seq:
        #         # Append what is there and then
        #         # go to the next line to possibly
        #         # append more.
        #         seq_list.append(line_stripped)
        #         continue
        #     elif is_desc:
        #         desc = line_stripped

        #     seq = "".join(seq_list)
        #     if desc is not None and seq is not None:
        #         yield (desc, seq)
        #         desc = None
        #         seq = None

        # for line in f:
        #     line_stripped = line.strip()
        #     is_desc = line_stripped.startswith(">")
        #     is_seq = (not is_desc) and line_stripped.isalpha() 
            
        #     if is_seq:
        #         # Append what is there and then
        #         # go to the next line to possibly
        #         # append more.
        #         seq_list.append(line_stripped)
        #         continue
        #     elif is_desc:
        #         desc = line_stripped

        #     seq = "".join(seq_list)
        #     if desc is not None and seq is not None:
        #         yield (desc, seq)
        #         desc = None
        #         seq = None


    


def align(
    seq_1:str, 
    seq_2:str, 
    scoring_mat:dict[dict], 
    gap_existence_cost:int
) -> tuple[str, str, str, int]:
    """
    Args:
        gap_existence_cost: The cost for a gap just to exist.
            This cost should be non-negative.
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment.

    Returns:
        (
            seq_1_aligned_out,
            middle_part_out,
            seq_2_aligned_out,
            score
        )
    """
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

def print_alignment(desc_1:str, desc_2:str, alignment:tuple[str, str, str, int]):
    # TODO: Handle long alignments with proper line breaking.
    print(desc_1)
    print(desc_2)
    print("")
    print(alignment[0])
    print(alignment[1])
    print(alignment[2])
    print(f"score={str(alignment[3])}")


if __name__ == "__main__":
    sys.exit(main())