#!/usr/bin/env python3
"""Perform optimal global alignment of two nucleotide \
or amino acid sequences using the Needleman-Wunsch algorithm.

References:
1. https://web.stanford.edu/class/cs262/archives/presentations/lecture3.pdf
2. https://ocw.mit.edu/courses/6-096-algorithms-for-computational-biology-spring-2005/01f55f348ea1e95f7015bd1b40586012_lecture5.pdf
3. Martin Mann, Mostafa M Mohamed, Syed M Ali, and Rolf Backofen
     Interactive implementations of thermodynamics-based RNA structure and RNA-RNA interaction prediction approaches for example-driven teaching
     PLOS Computational Biology, 14 (8), e1006341, 2018.
4. Martin Raden, Syed M Ali, Omer S Alkhnbashi, Anke Busch, Fabrizio Costa, Jason A Davis, Florian Eggenhofer, Rick Gelhausen, Jens Georg, Steffen Heyne, Michael Hiller, Kousik Kundu, Robert Kleinkauf, Steffen C Lott, Mostafa M Mohamed, Alexander Mattheis, Milad Miladi, Andreas S Richter, Sebastian Will, Joachim Wolff, Patrick R Wright, and Rolf Backofen
     Freiburg RNA tools: a central online resource for RNA-focused research and teaching
     Nucleic Acids Research, 46(W1), W25-W29, 2018.
5. https://doi.org/10.1016/0022-2836(82)90398-9
6. http://www.cs.cmu.edu/~durand/03-711/2017/Lectures/Sequence-Alignment-2017.pdf
7. https://bioboot.github.io/bimm143_W20/class-material/nw/
"""

import sys
import argparse
from pathlib import Path
import math

import numpy as np

def main():
    usage = "Perform optimal global alignment of two nucleotide \
or amino acid sequences using the Needleman-Wunsch algorithm."
    # Create an object to store arguments passed from the command 
    # line.
    parser = argparse.ArgumentParser(description=usage)

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

    parser.add_argument(
        "-o", 
        required=True,
        help="Output file path to write a FASTA file containing the global alignment."
    )

    cmd_line_args = parser.parse_args()

    # Transform and validate command line arguments.
    path_to_score_matrix_file = Path(cmd_line_args.s)
    path_to_fasta_file = Path(cmd_line_args.i)
    if not path_to_fasta_file.is_file():
        raise FileNotFoundError("path_to_fasta_file does not point to a valid file.")
    gap_existence_cost = int(cmd_line_args.g)
    path_to_output = Path(cmd_line_args.o)
    if not path_to_output.parent.exists():
        raise FileNotFoundError("The parent directory of path_to_output does not exist.")
    
    #################################################################
    #################################################################

    # Read in descriptions and sequences from FASTA file.
    # Verify FASTA file is in the correct format.
    # Extract sequences.
    # Verify sequences are formatted correctly.
    # Handle sequences of 0 length.
    counter = 0
    for desc_and_seq in read_seq_from_fasta(fasta_path=path_to_fasta_file):
        counter += 1
        if counter == 1:
            desc_1, seq_1 = desc_and_seq
        elif counter == 2:
            desc_2, seq_2 = desc_and_seq
        else:
            break

    
    # Check that the product of the lengths of the sequences is
    # less than 20_000_000.  
    m = len(seq_1)
    n = len(seq_2)
    seq_len_prod = m*n
    if not seq_len_prod < 20_000_000:
        raise RuntimeError(f"Your sequences are too long.  They have lengths of {m} and {n}")
    # Read in scoring matrix file.
    # Verify format of scoring matrix file.
    # Get the data from the scoring matrix into a nested dictionary
    # with codes for the letters as keys.
    scoring_mat = read_scoring_mat(scoring_mat_path=path_to_score_matrix_file)
    
    # Check that the scoring matrix is symmetric.
    if not check_symmetric(mat=scoring_mat):
        raise RuntimeError("The scoring matrix is not symmetric.")
    
    # For each row, the entry on the main diagonal
    # should be greater than or equal to the other entries in the row.
    if not check_big_main_diag(mat=scoring_mat):
        raise RuntimeError("The scoring matrix does not make sense because the maximum for each row does not occur on the main diagonal.")
    # Check that the sequences
    # only contain letters present in the scoring matrix.
    scoring_mat_letters = scoring_mat.keys()
    seq_1_letter_ok = [letter in scoring_mat_letters for letter in seq_1]
    if not all(seq_1_letter_ok):
        not_ok_letters = [letter for letter in seq_1 if letter not in scoring_mat_letters]
        raise RuntimeError(f"There were letters in seq_1 not present in scoring_mat, i.e. {not_ok_letters}")
    
    seq_2_letter_ok = [letter in scoring_mat_letters for letter in seq_2]
    if not all(seq_2_letter_ok):
        not_ok_letters = [letter for letter in seq_2 if letter not in scoring_mat_letters]
        raise RuntimeError(f"There were letters in seq_2 not present in scoring_mat, i.e. {not_ok_letters}")
    
    # Perform the alignment, insert gaps, and compute the score.
    alignment = align(
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        gap_existence_cost=gap_existence_cost
    )
    print_alignment(
        *alignment,
        desc_1=desc_1,
        desc_2=desc_2
    )
   
    # Write the outputs to a file.
    write_alignment(
        out_path=path_to_output,
        desc_1=desc_1,
        desc_2=desc_2,
        alignment=alignment
    )
    return None

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
                seq = "".join(seq_list).upper()
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
        seq = "".join(seq_list).upper()
        if not (len(seq) > 0):
            raise RuntimeError("Empty sequence detected in FASTA.")
        
        yield (desc, seq)


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
    # partial_A_mat[i][j] holds the best scores for when seq_1[i]
    # aligns with seq_2[j].
    # 
    # partial_B_mat[i][j] holds the best scores for when seq_2[j]
    # aligns with a new gap (or another gap in a run of gaps) in seq_1.
    # 
    # partial_C_mat[i][j] holds the best scores for when seq_1[i]
    # aligns with a new gap (or another gap in a run of gaps) in seq_2.
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

    best_paths_mat = init_best_paths_matrix(
        dynamic_prog_num_rows=dynamic_prog_num_rows,
        dynamic_prog_num_cols=dynamic_prog_num_cols
    )

    partial_A_mat, partial_B_mat, partial_C_mat, best_paths_mat, score = warmup_align(
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        gap_existence_cost=gap_existence_cost,
        dynamic_prog_num_cols=dynamic_prog_num_cols,
        partial_A_mat=partial_A_mat,
        partial_B_mat=partial_B_mat,
        partial_C_mat=partial_C_mat,
        best_paths_mat=best_paths_mat
    )
    print("after warmup_align")
    print("partial_A_mat")
    print(partial_A_mat)
    print("partial_B_mat")
    print(partial_B_mat)
    print("partial_C_mat")
    print(partial_C_mat)
    print("best_paths_mat")
    print(best_paths_mat)

    partial_A_mat, partial_B_mat, partial_C_mat, best_paths_mat, score = do_core_align(
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        gap_existence_cost=gap_existence_cost,
        dynamic_prog_num_rows=dynamic_prog_num_rows,
        dynamic_prog_num_cols=dynamic_prog_num_cols,
        partial_A_mat=partial_A_mat,
        partial_B_mat=partial_B_mat,
        partial_C_mat=partial_C_mat,
        best_paths_mat=best_paths_mat,
        score=score
    )

    print("in align")
    print("best_paths_mat before traceback")
    print(best_paths_mat)

    seq_1_aligned_out, middle_part_out, seq_2_aligned_out = traceback(
        best_paths_mat=best_paths_mat,
        seq_1=seq_1,
        seq_2=seq_2
    )
    
    return (
        seq_1_aligned_out,
        middle_part_out,
        seq_2_aligned_out,
        score
    )









    ##############################

    
    print("beginning partial matrices")
    print(partial_A_mat)

    # Go one row at a time through partial_A_mat, partial_B_mat, and 
    # partial_C_mat (starting at row index 1, col index 1 and always 
    # skipping col index 0).  Simultaneously, fill out the 
    # best_paths_mat.  (Note that we need to save the entirety of 
    # the best_paths_mat.)  We only need to keep two rows each of 
    # partial_A_mat, partial_B_mat, and partial_C_mat in memory 
    # at a time.
    # There are 3 possible values for each entry in the best_paths_mat
    # to indicate one of the following alignment "moves":
    # 0 = ↖ (match/mismatch)
    # 1 = ← (new gap or continuation of run of gaps in seq_1)
    # 2 = ↑ (new gap or continuation of run of gaps in seq_2)
    best_paths_mat = init_best_paths_matrix(
        dynamic_prog_num_rows=dynamic_prog_num_rows,
        dynamic_prog_num_cols=dynamic_prog_num_cols
    )
    
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
        # Always do the max operations with partial_A_mat first
        # because a max there is better than the same max somewhere else.
        # TODO: remove score_choices
        score_choices = [
            partial_A_mat[partial_mat_prev_row_id][j - 1],
            partial_B_mat[partial_mat_prev_row_id][j - 1],
            partial_C_mat[partial_mat_prev_row_id][j - 1]
        ]
        prev_best = max(score_choices)
        # prev_best = max(
        #     partial_A_mat[partial_mat_prev_row_id][j - 1],
        #     partial_B_mat[partial_mat_prev_row_id][j - 1],
        #     partial_C_mat[partial_mat_prev_row_id][j - 1]
        # )
        # TODO: delete if
        
        if sum([prev_best == score_choices]) > 1:
            viable_moves = np.where([prev_best == score_choices])
            print(f"Tie. viable_moves: {viable_moves}")
            
        partial_A_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
        
        if j == 1:
            ...
            # print('partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost')
            # print(str(partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost))
            # print('partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]]')
            # print(str(partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]]))
            # print('partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost')
            # print(partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost)
        # Consider partial_B_mat
        score_choices = [
            partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
            partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        ]
        # partial_B_mat[partial_mat_cur_row_id][j] = max(
        #     partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        #     partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
        #     partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        # )
        partial_B_mat[partial_mat_cur_row_id][j] = max(
            score_choices
        )

        if sum([partial_B_mat[partial_mat_cur_row_id][j] == score_choices]) > 1:
            viable_moves = np.where([partial_B_mat[partial_mat_cur_row_id][j] == score_choices])
            print(f"Tie. viable_moves: {viable_moves}")

        # Consider partial_C_mat
        score_choices = [
            partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] 
        ]

        # partial_C_mat[partial_mat_cur_row_id][j] = max(
        #     partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        #     partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        #     partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] 
        # )
        partial_C_mat[partial_mat_cur_row_id][j] = max(
            score_choices
        )

        if sum([partial_C_mat[partial_mat_cur_row_id][j] == score_choices]) > 1:
            viable_moves = np.where([partial_C_mat[partial_mat_cur_row_id][j] == score_choices])
            print(f"Tie. viable_moves: {viable_moves}")
        
        # Choose the best move.
        possible_new_scores = [
            partial_A_mat[partial_mat_cur_row_id][j],
            partial_B_mat[partial_mat_cur_row_id][j],
            partial_C_mat[partial_mat_cur_row_id][j]
        ]
        max_possible_new_score = max(possible_new_scores)
        # TODO: DELETE
        if sum([max_possible_new_score == possible_new_scores]) > 1:
            viable_moves = np.where([max_possible_new_score == possible_new_scores])
            print(f"Overall Tie. viable_moves: {viable_moves}")
        best_type_of_path = possible_new_scores.index(max_possible_new_score)
        
        best_paths_mat[i][j] = best_type_of_path

    print("Before going to the index=2 row of the dynamic programming matrix ensemble, we have:")
    print("partial_A_mat")
    print(partial_A_mat)
    print("partial_B_mat")
    print(partial_B_mat)
    print("partial_C_mat")
    print(partial_C_mat)

    for i in range(2, dynamic_prog_num_rows):
        # Prep for a new row iteration.
        # https://stackoverflow.com/a/14836456
        # Do some swapping.
        partial_mat_prev_row_id, partial_mat_cur_row_id = partial_mat_cur_row_id, partial_mat_prev_row_id
        # Update the 0th columns based on how gaps are penalized.
        partial_A_mat[partial_mat_cur_row_id][0] = partial_A_mat[partial_mat_prev_row_id][0] + scoring_mat[seq_1[seq_1_index]]["-"]
        partial_B_mat[partial_mat_cur_row_id][0] = partial_B_mat[partial_mat_prev_row_id][0] + scoring_mat[seq_1[seq_1_index]]["-"]
        partial_C_mat[partial_mat_cur_row_id][0] = partial_C_mat[partial_mat_prev_row_id][0] + scoring_mat[seq_1[seq_1_index]]["-"]
        
        for j in range(1, dynamic_prog_num_cols):
            # prep for this iteration
            seq_1_index = i - 1
            seq_2_index = j - 1

            # body of loop
            # Consider partial_A_mat
            # Always do the max operations with partial_A_mat first
            # because a max there is better than the same max somewhere else.
            # TODO: remove score_choices
            score_choices = [
                partial_A_mat[partial_mat_prev_row_id][j - 1],
                partial_B_mat[partial_mat_prev_row_id][j - 1],
                partial_C_mat[partial_mat_prev_row_id][j - 1]
            ]
            prev_best = max(score_choices)
            # prev_best = max(
            #     partial_A_mat[partial_mat_prev_row_id][j - 1],
            #     partial_B_mat[partial_mat_prev_row_id][j - 1],
            #     partial_C_mat[partial_mat_prev_row_id][j - 1]
            # )
            # TODO: delete if
            import numpy as np
            if sum([prev_best == score_choices]) > 1:
                viable_moves = np.where([prev_best == score_choices])
                print(f"Tie. viable_moves: {viable_moves}")
                
            partial_A_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
            if j == 1:
                ...
                # print('partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost')
                # print(str(partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost))
                # print('partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]]')
                # print(str(partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]]))
                # print('partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost')
                # print(partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost)
            # Consider partial_B_mat
            score_choices = [
                partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
                partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
            ]
            # partial_B_mat[partial_mat_cur_row_id][j] = max(
            #     partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            #     partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
            #     partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
            # )
            partial_B_mat[partial_mat_cur_row_id][j] = max(
                score_choices
            )

            if sum([partial_B_mat[partial_mat_cur_row_id][j] == score_choices]) > 1:
                viable_moves = np.where([partial_B_mat[partial_mat_cur_row_id][j] == score_choices])
                print(f"Tie. viable_moves: {viable_moves}")

            # Consider partial_C_mat
            score_choices = [
                partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] 
            ]

            # partial_C_mat[partial_mat_cur_row_id][j] = max(
            #     partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            #     partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            #     partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] 
            # )
            partial_C_mat[partial_mat_cur_row_id][j] = max(
                score_choices
            )

            if sum([partial_C_mat[partial_mat_cur_row_id][j] == score_choices]) > 1:
                viable_moves = np.where([partial_C_mat[partial_mat_cur_row_id][j] == score_choices])
                print(f"Tie. viable_moves: {viable_moves}")
            
            # Choose the best move.
            possible_new_scores = [
                partial_A_mat[partial_mat_cur_row_id][j],
                partial_B_mat[partial_mat_cur_row_id][j],
                partial_C_mat[partial_mat_cur_row_id][j]
            ]
            max_possible_new_score = max(possible_new_scores)
            # TODO: DELETE
            if sum([max_possible_new_score == possible_new_scores]) > 1:
                viable_moves = np.where([max_possible_new_score == possible_new_scores])
                print(f"Overall Tie. viable_moves: {viable_moves}")
            best_type_of_path = possible_new_scores.index(max_possible_new_score)
            
            best_paths_mat[i][j] = best_type_of_path

            # print("partial_A_mat")
            # print(partial_A_mat)
            # print("partial_B_mat")
            # print(partial_B_mat)
            # print("partial_C_mat")
            # print(partial_C_mat)

    print("best_paths_mat")
    print(best_paths_mat)

    score = max_possible_new_score

    seq_1_aligned_out, middle_part_out, seq_2_aligned_out = traceback(
        best_paths_mat=best_paths_mat,
        seq_1=seq_1,
        seq_2=seq_2
    )
    
    return (
        seq_1_aligned_out,
        middle_part_out,
        seq_2_aligned_out,
        score
    )

def traceback(best_paths_mat:list[list], seq_1:str, seq_2:str) -> tuple[str, str, str]:
    """Perform traceback through best_paths_mat
    
    to find the alignment.
    There are 3 possible values for each entry in the best_paths_mat
    to indicate one of the following alignment "moves":
    0 = ↖ (match/mismatch)
    1 = ← (new gap or continuation of run of gaps in seq_1)
    2 = ↑ (new gap or continuation of run of gaps in seq_2)

    Args: 
        best_paths_mat: list of length len(seq_1) + 1
            where each element is a list of length 
            len(seq_2) + 1
    """
    # Prepare for loop.
    seq_1_aligned = []
    seq_2_aligned = []
    middle_part = []

    m = len(seq_1)
    n = len(seq_2)

    # http://www.cs.cmu.edu/~durand/03-711/2017/Lectures/Sequence-Alignment-2017.pdf
    max_num_alignment_moves = m + n

    # Start at the bottom-right.
    seq_1_index = m - 1
    seq_2_index = n - 1

    for w in range(max_num_alignment_moves):
        # Prep for this iteration.
        # Because of the initial row and column in
        # best_paths_mat that doesn't align with
        # any parts of the two sequences, the indices
        # are off by one.
        best_paths_mat_row_index = seq_1_index + 1
        best_paths_mat_col_index = seq_2_index + 1

        path_indicator = best_paths_mat[best_paths_mat_row_index][best_paths_mat_col_index]
        print(path_indicator)
        if path_indicator == 0:
            # match/mismatch is the best move
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
        elif path_indicator == 1:
            # gap in seq_1 is the best move
            middle_part.append(" ")
            seq_1_aligned.append("-")
            seq_2_aligned.append(seq_2[seq_2_index])
            seq_2_index -= 1
        else:
            # gap in seq_2 is the best move
            middle_part.append(" ")
            seq_1_aligned.append(seq_1[seq_1_index])
            seq_1_index -= 1
            seq_2_aligned.append("-")

        # Determine whether the loop should continue.
        if seq_1_index == -1 and seq_2_index == -1:
            print("seq_1_index")
            print(seq_1_index)
            print("seq_2_index")
            print(seq_2_index)
            break

    print("seq_1_index")
    print(seq_1_index)
    print("seq_2_index")
    print(seq_2_index)
    seq_1_aligned.reverse()
    middle_part.reverse()
    seq_2_aligned.reverse()

    seq_1_aligned_out = "".join(seq_1_aligned)
    middle_part_out = "".join(middle_part)
    seq_2_aligned_out = "".join(seq_2_aligned)

    return (
        seq_1_aligned_out,
        middle_part_out,
        seq_2_aligned_out
    )

def draw_random_seq(alphabet:list, min_len:int, max_len:int):
    # https://numpy.org/doc/stable/reference/random/index.html
    rng = np.random.default_rng()
    alphabet_len = len(alphabet)
    seq_len = rng.integers(low=min_len, high=max_len, endpoint=True, size=1)
    alphabet_indices = rng.integers(low=0, high=alphabet_len, endpoint=False, size=seq_len)
    return "".join([alphabet[x] for x in alphabet_indices])


def make_matrix(num_rows:int, num_cols:int, fill_val:int|float|str) -> list[list]:
    """Make a matrix as a nested list.
    
    See: https://www.freecodecamp.org/news/list-within-a-list-in-python-initialize-a-nested-list/
    """
    return [
        [fill_val]*(num_cols) for i in range(num_rows)
    ]

def update_best_paths_mat(
    best_paths_mat:list[list],
    partial_A_mat:list[list],
    partial_B_mat:list[list],
    partial_C_mat:list[list],
    partial_mat_cur_row_id:int,
    best_paths_mat_row_id:int,
    best_paths_mat_col_id:int
) -> tuple[list[list], int|float]:
    """Return best_paths_mat and score.
    """
    partial_dp_matrix_col_id = best_paths_mat_col_id
    # Choose the best move.
    possible_new_scores = [
        partial_A_mat[partial_mat_cur_row_id][partial_dp_matrix_col_id],
        partial_B_mat[partial_mat_cur_row_id][partial_dp_matrix_col_id],
        partial_C_mat[partial_mat_cur_row_id][partial_dp_matrix_col_id]
    ]
    max_possible_new_score = max(possible_new_scores)
    
    # Find the index corresponding to where the maximum
    # is first achieved.
    best_type_of_path = possible_new_scores.index(max_possible_new_score)
    
    best_paths_mat[best_paths_mat_row_id][best_paths_mat_col_id] = best_type_of_path

    return (best_paths_mat, max_possible_new_score)

def do_core_align(
    seq_1:str, 
    seq_2:str, 
    scoring_mat:dict[dict], 
    gap_existence_cost:int,
    dynamic_prog_num_rows:int,
    dynamic_prog_num_cols:int,
    partial_A_mat:list[list],
    partial_B_mat:list[list],
    partial_C_mat:list[list],
    best_paths_mat:list[list],
    score:int|float
) -> tuple[list[list], list[list], list[list], list[list], int]:
    """
    Find a global alignment of the subsequences
    
    seq_1[1:] (assuming len(seq_1) > 1) and seq_2.  
    Args:
        gap_existence_cost: The cost for a gap just to exist.
            This cost should be non-negative.
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment.
        partial_A_mat: already filled from an initial run
            of the algorithm.
        partial_B_mat: already filled from an initial run
            of the algorithm.
        partial_C_mat: already filled from an initial run
            of the algorithm.
        best_paths_mat: already filled for the first two
            rows from an initial run of the algorithm.

    Returns:
        (
            partial_A_mat,
            partial_B_mat,
            partial_C_mat,
            best_paths_mat,
            score
        )
    """
    # Pre loop
    partial_mat_prev_row_id = 0
    partial_mat_cur_row_id = 1

    for i in range(2, dynamic_prog_num_rows):
        # Prep for a new row iteration.
        # Take special care for the first two columns.
        j = 0
        # The best_paths_mat does not need its 0-index column
        # to be updated.  It should have already been
        # initialized correctly.
        # https://stackoverflow.com/a/14836456
        # Do some swapping.
        partial_mat_prev_row_id, partial_mat_cur_row_id = partial_mat_cur_row_id, partial_mat_prev_row_id
        # Update the 0-index columns based on how gaps are penalized.
        seq_1_index = i - 1
        
        partial_A_mat[partial_mat_cur_row_id][j] = partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat[seq_1[seq_1_index]]["-"]
        partial_B_mat[partial_mat_cur_row_id][j] = partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat[seq_1[seq_1_index]]["-"]
        partial_C_mat[partial_mat_cur_row_id][j] = partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat[seq_1[seq_1_index]]["-"]
        
   
        print("best_paths_mat line 817")
        print(best_paths_mat)
        # Update the 1-index columns based on how gaps are penalized.
        j = 1
        seq_2_index = j - 1
        # The gap existence cost is always paid for partial_B_mat
        # and partial_C_mat because j == 1.
        # There couldn't have been a pre-existing gap in seq_1.
        partial_A_mat[partial_mat_cur_row_id][j] = partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]]
        partial_B_mat[partial_mat_cur_row_id][j] = partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        partial_C_mat[partial_mat_cur_row_id][j] = partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        
        best_paths_mat, score = update_best_paths_mat(
            best_paths_mat=best_paths_mat,
            partial_A_mat=partial_A_mat,
            partial_B_mat=partial_B_mat,
            partial_C_mat=partial_C_mat,
            partial_mat_cur_row_id=partial_mat_cur_row_id,
            best_paths_mat_row_id=i,
            best_paths_mat_col_id=j
        )

        for j in range(2, dynamic_prog_num_cols):
            # prep for this iteration
            seq_1_index = i - 1
            seq_2_index = j - 1

            # body of loop
            # Consider partial_A_mat
            # Always do the max operations with partial_A_mat first
            # because a max there is better than the same max somewhere else.

            prev_best = max(
                partial_A_mat[partial_mat_prev_row_id][j - 1],
                partial_B_mat[partial_mat_prev_row_id][j - 1],
                partial_C_mat[partial_mat_prev_row_id][j - 1]
            )
            
            partial_A_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
           
            # Consider partial_B_mat
            partial_B_mat[partial_mat_cur_row_id][j] = max(
                partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
                partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
            )

            # Consider partial_C_mat
            partial_C_mat[partial_mat_cur_row_id][j] = max(
                partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
                partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] 
            )
            
            best_paths_mat, score = update_best_paths_mat(
                best_paths_mat=best_paths_mat,
                partial_A_mat=partial_A_mat,
                partial_B_mat=partial_B_mat,
                partial_C_mat=partial_C_mat,
                partial_mat_cur_row_id=partial_mat_cur_row_id,
                best_paths_mat_row_id=i,
                best_paths_mat_col_id=j
            )
    
  
    return (
        partial_A_mat,
        partial_B_mat,
        partial_C_mat,
        best_paths_mat,
        score
    )


def warmup_align(
    seq_1:str, 
    seq_2:str, 
    scoring_mat:dict[dict], 
    gap_existence_cost:int,
    dynamic_prog_num_cols:int,
    partial_A_mat:list[list],
    partial_B_mat:list[list],
    partial_C_mat:list[list],
    best_paths_mat:list[list]
) -> tuple[list[list], list[list], list[list], list[list], int|float]:
    """
    Find a global alignment of the subsequences
    
    seq_1[0] and seq_2.  
    Args:
        gap_existence_cost: The cost for a gap just to exist.
            This cost should be non-negative.
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment.
        partial_A_mat: already initialized for 0-index row and 
            0-index column.
        partial_B_mat: already initialized for 0-index row and 
            0-index column.
        partial_C_mat: already initialized for 0-index row and 
            0-index column.
        best_paths_mat: already initialized for 0-index row and 
            0-index column.

    Returns:
        (
            partial_A_mat,
            partial_B_mat,
            partial_C_mat,
            best_paths_mat,
            score
        )
    """
    i = 1
    j = 1
    partial_mat_prev_row_id = 0
    partial_mat_cur_row_id = 1

    # prep for this iteration
    seq_1_index = i - 1
    seq_2_index = j - 1

    # body of loop
    # Consider partial_A_mat
    # Always do the max operations with partial_A_mat first
    # because a max there is better than the same max somewhere else.
    prev_best = max(
        partial_A_mat[partial_mat_prev_row_id][j - 1],
        partial_B_mat[partial_mat_prev_row_id][j - 1],
        partial_C_mat[partial_mat_prev_row_id][j - 1]
    )

    partial_A_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
    
    # Consider partial_B_mat
    # The gap existence cost is always paid because j == 1.
    # There couldn't have been a pre-existing gap in seq_1.
    partial_B_mat[partial_mat_cur_row_id][j] = max(
        partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
    )

    # Consider partial_C_mat
    # The gap existence cost is always paid because i == 1.
    # There couldn't have been a pre-existing gap in seq_2.
    partial_C_mat[partial_mat_cur_row_id][j] = max(
        partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
        partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
    )
    
    # Choose one of the best moves.
    best_paths_mat, score = update_best_paths_mat(
        best_paths_mat=best_paths_mat,
        partial_A_mat=partial_A_mat,
        partial_B_mat=partial_B_mat,
        partial_C_mat=partial_C_mat,
        partial_mat_cur_row_id=partial_mat_cur_row_id,
        best_paths_mat_row_id=i,
        best_paths_mat_col_id=j
    )

    for j in range(2, dynamic_prog_num_cols):
        # prep for this iteration
        seq_1_index = i - 1
        seq_2_index = j - 1

        # body of loop
        # Consider partial_A_mat
        # Always do the max operations with partial_A_mat first
        # because a max there is better than the same max somewhere else.
        prev_best = max(
            partial_A_mat[partial_mat_prev_row_id][j - 1],
            partial_B_mat[partial_mat_prev_row_id][j - 1],
            partial_C_mat[partial_mat_prev_row_id][j - 1]
        )
   
        partial_A_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
        
        # Consider partial_B_mat
        partial_B_mat[partial_mat_cur_row_id][j] = max(
            partial_A_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]],
            partial_C_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        )

        # Consider partial_C_mat
        # The gap existence cost is always paid because i == 1.
        # There couldn't have been a pre-existing gap in seq_2.
        partial_C_mat[partial_mat_cur_row_id][j] = max(
            partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        )
        
        # Choose one of the best moves.
        best_paths_mat, score = update_best_paths_mat(
            best_paths_mat=best_paths_mat,
            partial_A_mat=partial_A_mat,
            partial_B_mat=partial_B_mat,
            partial_C_mat=partial_C_mat,
            partial_mat_cur_row_id=partial_mat_cur_row_id,
            best_paths_mat_row_id=i,
            best_paths_mat_col_id=j
        )

    return (
        partial_A_mat,
        partial_B_mat,
        partial_C_mat,
        best_paths_mat,
        score
    )

# def init_partial_A_mat(
#     gap_existence_cost:int, 
#     seq_1:str,
#     seq_2:str,
#     scoring_mat:dict[dict], 
#     dynamic_prog_num_cols:int
# ) -> list[list]:
#     partial_A_mat = init_partial_dynamic_prog_matrix(
#         gap_existence_cost=gap_existence_cost,
#         seq_1=seq_1,
#         seq_2=seq_2,
#         scoring_mat=scoring_mat,
#         dynamic_prog_num_cols=dynamic_prog_num_cols
#     )

#     return partial_A_mat

def init_partial_dynamic_prog_matrix(
    gap_existence_cost:int, 
    seq_1:str,
    seq_2:str,
    scoring_mat:dict[dict], 
    dynamic_prog_num_cols:int
) -> list[list]:
    """This is the base function used in init_partial_A_mat, 
    
    init_partial_B_mat, and init_partial_C_mat.  It initializes
    correctly for the 0-index rows and the 0-index columns.
    However, it leaves complete initialization of partial_A_mat,
    partial_B_mat, and partial_C_mat to other functions.
    """
    mat = make_matrix(
        num_rows=2,
        num_cols=dynamic_prog_num_cols,
        fill_val=0
    )
    # Take care of initialization with gap scores.
  
    # Loop prep
    # Start in column 1
    j = 1

    # The indices into our sequences are always 1 behind
    # the indices into our dynamic programming matrices.
    seq_2_index = j - 1

    # Pay a gap existence penalty for the entry in the 
    # 0th row and 1st column of each partial dynamic programming
    # matrix.
    cur_gap_score = -gap_existence_cost + scoring_mat["-"][seq_2[seq_2_index]]
    mat[0][j] = cur_gap_score

    # We do not have to pay the gap existence penalty for 
    # entries to the right of the 1-index column in the 
    # 0-index row.
    for j in range(2, dynamic_prog_num_cols):
        # Prep for this iteration
        # The sequence indices are always one behind
        # the row/column indices.
        seq_2_index = j - 1

        # body of loop
        # The cur_gap_score will have already incorporated
        # a gap existence penalty.
        cur_gap_score = cur_gap_score + scoring_mat["-"][seq_2[seq_2_index]]
        mat[0][j] = cur_gap_score

    # We also have to pay the gap existence penalty for 
    # the entry in the 1-index row and 0-index column
    # for partial_A_mat, partial_B_mat, and partial_C_mat.
    mat[1][0] = -gap_existence_cost + scoring_mat[seq_1[0]]["-"]
    return mat

def init_best_paths_matrix(
    dynamic_prog_num_rows,
    dynamic_prog_num_cols    
) -> list[list]:
    """Initialize a matrix where the entry in row i and column j
    
    indicates the best final 'move' to align the subsequences 
    seq_1[0:(i - 1)] and seq_2[0:(j - 1)].

    Args:
        dynamic_prog_num_rows: len(seq_1) + 1
        dynamic_prog_num_cols: len(seq_2) + 1
    Returns:
        best_paths_mat as a nested list.
        There are 3 possible values for each entry in the best_paths_mat
        to indicate one of the following alignment "moves":
        0 = ↖ (match/mismatch)
        1 = ← (new gap or continuation of run of gaps in seq_1)
        2 = ↑ (new gap or continuation of run of gaps in seq_2)
    """
    # Based on the order of arguments to every
    # call to max, we initialize with
    # 1's because 1 indicates moving left.
    best_paths_mat = make_matrix(
        num_rows=dynamic_prog_num_rows,
        num_cols=dynamic_prog_num_cols,
        fill_val=1
    )

    # Based on the order of arguments to every
    # call to max, we decide to put 
    # 2's in the beginning of each row
    # because 2 indicates moving up.
    for i in range(1, dynamic_prog_num_rows):
        best_paths_mat[i][0] = 2

    return best_paths_mat

def check_symmetric(mat:dict[dict]) -> bool:
    """Check if a matrix is symmetric.
    
    Args:
        mat: nested dictionary representing a matrix
    
    Returns:
        True if mat is symmetric and False otherwise.
    """
    # https://realpython.com/iterate-through-dictionary-python/#traversing-a-dictionary-directly
    for outer_key in mat.keys():
        # Assume that the outer and inner
        # keys are the same.
        for inner_key in mat.keys():
            try:
                has_eq_vals = mat[outer_key][inner_key] == mat[inner_key][outer_key]
            except KeyError:
                return False
            if not has_eq_vals:
                return False
    
    return True

def check_big_main_diag(mat:dict[dict]) -> bool:
    """Check if each row of a matrix has its maximum 
    in the main diagonal entry.
    
    Args:
        mat: nested dictionary representing a matrix
    
    Returns:
        True if each row of mat has its maximum 
        in the main diagonal entry; otherwise, False.
    """
    # https://realpython.com/iterate-through-dictionary-python/#traversing-a-dictionary-directly
    for outer_key in mat.keys():

        outer_key_max_val = max(mat[outer_key].values())
        try:
            # Test if the main diagonal entry of mat
            # is the same as the outer_key_max_val.
            has_max_in_main_diag = mat[outer_key][outer_key] == outer_key_max_val
        except KeyError:
            raise RuntimeError("mat is not a proper nested dict representation of a matrix.")
        
        if not has_max_in_main_diag:
            return False
        
    return has_max_in_main_diag 


def print_alignment(
    seq_1_aligned:str, 
    mid:str, 
    seq_2_aligned:str, 
    score:int|float|str=math.nan, 
    desc_1:str="seq_1", 
    desc_2:str="seq_2", 
    chars_per_line:int=70
):
    
    print(desc_1)
    print(desc_2)
    print("")

    # Handle long alignments with proper line breaking.
    alignment_len = len(seq_1_aligned)
    num_sets_needed = math.ceil(alignment_len / chars_per_line)
    
    # Prep for loop
    lower = 0
    if num_sets_needed == 1:
        upper = alignment_len
    else:   
        upper = chars_per_line

    for u in range(num_sets_needed):
        # Loop body
        print(seq_1_aligned[lower:upper])
        print(mid[lower:upper])
        print(seq_2_aligned[lower:upper])
        # Prep for next iteration
        print("")
        lower = upper
        upper = lower + chars_per_line

    print(f"score={str(score)}")


def write_alignment(out_path:Path, desc_1:str, desc_2:str, alignment:tuple[str, str, str, int]):
    seq_1_aligned, mid, seq_2_aligned, score = alignment
    with out_path.open(mode="w") as f:
        f.writelines([
            "".join([desc_1, "; score=", str(score)]),
            "\n",
            seq_1_aligned,
            "\n"
        ])
        f.writelines([
            "".join([desc_2, "; score=", str(score)]),
            "\n",
            seq_2_aligned,
            "\n"
        ])

if __name__ == "__main__":
    sys.exit(main())