#!/usr/bin/env python3
"""Perform optimal global alignment of two nucleotide \
or amino acid sequences.

References:
1. https://web.stanford.edu/class/cs262/archives/presentations/lecture3.pdf
2. https://ocw.mit.edu/courses/6-096-algorithms-for-computational-biology-spring-2005/01f55f348ea1e95f7015bd1b40586012_lecture5.pdf
3. Martin Mann, Mostafa M Mohamed, Syed M Ali, and Rolf Backofen
     Interactive implementations of thermodynamics-based RNA structure and RNA-RNA interaction prediction approaches for example-driven teaching
     PLOS Computational Biology, 14 (8), e1006341, 2018.
4. Martin Raden, Syed M Ali, Omer S Alkhnbashi, Anke Busch, Fabrizio Costa, Jason A Davis, Florian Eggenhofer, Rick Gelhausen, Jens Georg, Steffen Heyne, Michael Hiller, Kousik Kundu, Robert Kleinkauf, Steffen C Lott, Mostafa M Mohamed, Alexander Mattheis, Milad Miladi, Andreas S Richter, Sebastian Will, Joachim Wolff, Patrick R Wright, and Rolf Backofen
     Freiburg RNA tools: a central online resource for RNA-focused research and teaching
     Nucleic Acids Research, 46(W1), W25-W29, 2018.
5. An improved algorithm for matching biological sequences. Osamu Gotoh. https://doi.org/10.1016/0022-2836(82)90398-9
6. http://www.cs.cmu.edu/~durand/03-711/2017/Lectures/Sequence-Alignment-2017.pdf
7. https://bioboot.github.io/bimm143_W20/class-material/nw/
8. https://www.ncbi.nlm.nih.gov/CBBresearch/Przytycka/download/lectures/PCB_Lect02_Pairwise_allign.pdf
9. https://ics.uci.edu/~xhx/courses/CS284A-F08/lectures/alignment.pdf
10. https://link.springer.com/chapter/10.1007/978-3-319-90684-3_2
11. Optimal sequence alignment using affine gap costs. https://link.springer.com/content/pdf/10.1007/BF02462326.pdf
12. Optimal alignments in linear space. Eugene W. Myers, Webb Miller.  https://doi.org/10.1093/bioinformatics/4.1.11
13. Sequence alignment using FastLSA. https://webdocs.cs.ualberta.ca/~duane/publications/pdf/2000metmbs.pdf
14. MASA: A Multiplatform Architecture for Sequence Aligners
        with Block Pruning. https://doi.org/10.1145/2858656
15. https://community.gep.wustl.edu/repository/course_materials_WU/annotation/Introduction_Dynamic_Programming.pdf
16. Optimal gap-affine alignment in O(s) space. https://doi.org/10.1093/bioinformatics/btad074
17. Exact global alignment using A* with chaining seed heuristic and match pruning.
    https://doi.org/10.1093/bioinformatics/btae032
18. Transforming match bonus into cost. https://curiouscoding.nl/posts/alignment-scores-transform/
19. Improving the time and space complexity of the WFA algorithm and generalizing its scoring.
        https://doi.org/10.1101/2022.01.12.476087
20. A* PA2: up to 20 times faster exact global alignment.
        https://doi.org/10.1101/2024.03.24.586481
21. Notes on Dynamic-Programming Sequence Alignment.
        https://globin.bx.psu.edu/courses/fall2001/DP.pdf
22. Lecture 6: Affine gap penalty function.
        https://www.cs.hunter.cuny.edu/~saad/courses/compbio/lectures/lecture6.pdf
"""

import sys
import argparse
from pathlib import Path
import math
import random

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
        raise RuntimeError(f"Your sequences are too long.  The product of their lengths should be less than 20,000,000.  They have lengths of {m} and {n}")
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


def get_max_similarity_score(scoring_mat:dict[dict]) -> int|float:
    """Get the max similarity score
    
    from a scoring matrix.  
    Reference: https://curiouscoding.nl/posts/alignment-scores-transform/
    """
    # prep for loop
    cur_max = - math.inf
    for seq_1_letter, seq_2_scores in scoring_mat.items():
        # seq_1_letter is a key for scoring_mat.
        # seq_2_scores is the inner dict for the outer
        # key of seq_1_letter.
        new_possible_max = max(seq_2_scores.values())
        cur_max = max(cur_max, new_possible_max)

    return cur_max


def transform_scoring_mat_to_cost_mat(
    scoring_mat:dict[dict], 
    max_score:int|float,
    delta_d:int|float=None,
    delta_i:int|float=None
) -> dict[dict]:
    """Transform to a proper distance matrix.
    
    Args: 
        scoring_mat: Nested dict representation of 
            a similarity matrix
        max_score: Max in scoring_mat
        delta_d: amount to increase the cost of a
            horizontal step in the dynamic programming
            matrix. `delta_d + delta_i >= max_score`.
            Default: None.
        delta_i: amount to increase the cost of a
            vertical step in the dynamic programming
            matrix. 
            `delta_d + delta_i >= max_score`.
            Default: None.

    Returns:
        Nested dict representation of a distance matrix
            whose entries correspond to string edit costs
            for matches and mismatches.

    Reference: https://curiouscoding.nl/posts/alignment-scores-transform/
    """
    b = max_score
    if delta_d is None:
        delta_d = math.floor(b/2)
    if delta_i is None:
        delta_i = math.ceil(b/2)
    
    for seq_1_letter, seq_2_scores in scoring_mat.items():
        # seq_1_letter is a key for scoring_mat.
        # seq_2_scores is the inner dict for the outer
        # key of seq_1_letter.
        for seq_2_letter, score in seq_2_scores.items():
            # The scores are transformed differently
            # for insertions and deletions, than they
            # are for matches and mismatches.
            # if seq_1_letter == seq_2_letter and seq_1_letter != "-":
            #     # Update matches
            #     seq_2_scores[seq_2_letter] = -score
            # # elif seq_1_letter != seq_2_letter and seq_1_letter != "-":
            # #     # Update mis-matches
            # #     seq_2_scores[seq_2_letter] = score + b
            if seq_1_letter == "-" and seq_2_letter != "-":
                # Update deletions (horizontal steps)
                seq_2_scores[seq_2_letter] = -score + delta_d
            elif seq_2_letter == "-" and seq_1_letter != "-":
                # Update insertions (vertical steps)
                seq_2_scores[seq_2_letter] = -score + delta_i 
            else:
                # Update matches and mismatches.
                seq_2_scores[seq_2_letter] = -score + delta_d + delta_i
   
    cost_mat = scoring_mat
    return cost_mat

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
            alignment. Note that an alignment like

                    AT-CG
                    ||  |
                    ATT-G

            incurs the gap_existence_cost twice.

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

    # Initialize a matrix where the entry in row i and column j
    # indicates the best final score for the alignment of the 
    # subsequences seq_1[0:(i - 1)] and seq_2[0:(j - 1)].
    D_mat = init_D_mat(
        dynamic_prog_num_rows=dynamic_prog_num_rows,
        dynamic_prog_num_cols=dynamic_prog_num_cols,
        partial_A_mat=partial_A_mat,
        partial_B_mat=partial_B_mat,
        partial_C_mat=partial_C_mat
    )

    best_paths_mat = init_best_paths_matrix(
        dynamic_prog_num_rows=dynamic_prog_num_rows,
        dynamic_prog_num_cols=dynamic_prog_num_cols
    )

    partial_A_mat, partial_B_mat, partial_C_mat, D_mat, best_paths_mat, score = warmup_align(
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        gap_existence_cost=gap_existence_cost,
        dynamic_prog_num_cols=dynamic_prog_num_cols,
        partial_A_mat=partial_A_mat,
        partial_B_mat=partial_B_mat,
        partial_C_mat=partial_C_mat,
        D_mat=D_mat,
        best_paths_mat=best_paths_mat
    )
    # print("after warmup_align")
    # print("partial_A_mat")
    # print(partial_A_mat)
    # print("partial_B_mat")
    # print(partial_B_mat)
    # print("partial_C_mat")
    # print(partial_C_mat)
    # print("best_paths_mat")
    # print(best_paths_mat)

    partial_A_mat, partial_B_mat, partial_C_mat, D_mat, best_paths_mat, score = do_core_align(
        seq_1=seq_1,
        seq_2=seq_2,
        scoring_mat=scoring_mat,
        gap_existence_cost=gap_existence_cost,
        dynamic_prog_num_rows=dynamic_prog_num_rows,
        dynamic_prog_num_cols=dynamic_prog_num_cols,
        partial_A_mat=partial_A_mat,
        partial_B_mat=partial_B_mat,
        partial_C_mat=partial_C_mat,
        D_mat=D_mat,
        best_paths_mat=best_paths_mat,
        score=score
    )
    

    print("in align LINE 350")
    print("D_mat")
    print(D_mat)
    print("partial_A_mat")
    print(partial_A_mat)
    print("partial_B_mat")
    print(partial_B_mat)
    print("partial_C_mat")
    print(partial_C_mat)
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

def find_global_alignment(
    seq_1:str,
    seq_2:str,
    cost_mat:list[list],
    gap_open_cost:int|float
) -> tuple[str, str, str, int|float]:
    """
    Args:
        gap_open_cost: The cost for a gap just to exist.
            This cost should be non-negative.
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment. Note that an alignment like

                    AT-CG
                    ||  |
                    ATT-G

            incurs the gap_open_cost twice.

    Returns:
        (
            seq_1_aligned_out,
            middle_part_out,
            seq_2_aligned_out,
            cost
        )
    """
    middle_row_index = len(seq_1)
    best_paths_mat = init_best_paths_matrix(
        dynamic_prog_num_rows=middle_row_index + 1,
        dynamic_prog_num_cols=len(seq_2) + 1
    )
    dynamic_prog_num_cols = len(seq_2) + 1
    partial_dp_mat = init_partial_dynamic_prog_matrix_2(
        seq_1=seq_1,
        seq_2=seq_2,
        cost_mat=cost_mat,
        gap_open_cost=gap_open_cost,
        dynamic_prog_num_cols=dynamic_prog_num_cols
    )
    # The options for best_path_type are
    # coded as:
    # 0: match
    # 1: starting gap in seq_1
    # 2: continuing gap in seq_1
    # 3: starting gap in seq_2
    # 4: continuing gap in seq_2
    # 5: tie between 1 and 3
    # 6: tie between 1 and 4
    # 7: tie between 2 and 3
    # 8: tie between 2 and 4
    # 9: tie between 0 and 1
    # 10: tie between 0 and 2
    # 11: tie between 0 and 3
    # 12: tie between 0 and 4
    # 13: tie between 1 and 2
    # 14: tie between 3 and 4
    # 15: tie between 0, 1, and 3
    # 16: tie between 0, 1, and 4
    # 17: tie between 0, 2, and 3
    # 18: tie between 0, 2, and 4
    # 19: mismatch
    # 20: tie between 19 and 1
    # 21: tie between 19 and 2
    # 22: tie between 19 and 3
    # 23: tie between 19 and 4
    # 24: tie between 19, 1, and 3
    # 25: tie between 19, 1, and 4
    # 26: tie between 19, 2, and 3
    # 27: tie between 19, 2, and 4
    moves_for_gap_open_penalty_from_left = {0, 3, 4, 11, 12, 14, 19, 22, 23}
    moves_for_gap_open_penalty_from_up = {0, 1, 2, 9, 10, 13, 19, 20, 21}
    
    situation_mapper={
        # from_diag_best_path_type = 0
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (0, 1, 3)): 15,
        # 2-way ties for low
        ((0, 0, 2), (0, 1, 3)): 9,
        ((0, 2, 0), (0, 1, 3)): 11,
        ((2, 0, 0), (0, 1, 3)): 5,
        # 2-way ties for high
        ((0, 1, 1), (0, 1, 3)): 0,
        ((1, 0, 1), (0, 1, 3)): 1,
        ((1, 1, 0), (0, 1, 3)): 3,
        # no ties
        ((0, 1, 2), (0, 1, 3)): 0,
        ((1, 0, 2), (0, 1, 3)): 1,
        ((1, 2, 0), (0, 1, 3)): 3,
        ((0, 2, 1), (0, 1, 3)): 0,
        ((2, 0, 1), (0, 1, 3)): 1,
        ((2, 1, 0), (0, 1, 3)): 3,
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (0, 1, 4)): 16,
        # 2-way ties for low
        ((0, 0, 2), (0, 1, 4)): 9,
        ((0, 2, 0), (0, 1, 4)): 12,
        ((2, 0, 0), (0, 1, 4)): 6,
        # 2-way ties for high
        ((0, 1, 1), (0, 1, 4)): 0,
        ((1, 0, 1), (0, 1, 4)): 1,
        ((1, 1, 0), (0, 1, 4)): 4,
        # no ties
        ((0, 1, 2), (0, 1, 4)): 0,
        ((1, 0, 2), (0, 1, 4)): 1,
        ((1, 2, 0), (0, 1, 4)): 4,
        ((0, 2, 1), (0, 1, 4)): 0,
        ((2, 0, 1), (0, 1, 4)): 1,
        ((2, 1, 0), (0, 1, 4)): 4,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (0, 2, 3)): 17,
        # 2-way ties for low
        ((0, 0, 2), (0, 2, 3)): 10,
        ((0, 2, 0), (0, 2, 3)): 11,
        ((2, 0, 0), (0, 2, 3)): 7,
        # 2-way ties for high
        ((0, 1, 1), (0, 2, 3)): 0,
        ((1, 0, 1), (0, 2, 3)): 2,
        ((1, 1, 0), (0, 2, 3)): 3,
        # no ties
        ((0, 1, 2), (0, 2, 3)): 0,
        ((1, 0, 2), (0, 2, 3)): 2,
        ((1, 2, 0), (0, 2, 3)): 3,
        ((0, 2, 1), (0, 2, 3)): 0,
        ((2, 0, 1), (0, 2, 3)): 2,
        ((2, 1, 0), (0, 2, 3)): 3,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (0, 2, 4)): 18,
        # 2-way ties for low
        ((0, 0, 2), (0, 2, 4)): 10,
        ((0, 2, 0), (0, 2, 4)): 12,
        ((2, 0, 0), (0, 2, 4)): 8,
        # 2-way ties for high
        ((0, 1, 1), (0, 2, 4)): 0,
        ((1, 0, 1), (0, 2, 4)): 2,
        ((1, 1, 0), (0, 2, 4)): 4,
        # no ties
        ((0, 1, 2), (0, 2, 4)): 0,
        ((1, 0, 2), (0, 2, 4)): 2,
        ((1, 2, 0), (0, 2, 4)): 4,
        ((0, 2, 1), (0, 2, 4)): 0,
        ((2, 0, 1), (0, 2, 4)): 2,
        ((2, 1, 0), (0, 2, 4)): 4,
        # from_diag_best_path_type = 19
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (19, 1, 3)): 24,
        # 2-way ties for low
        ((0, 0, 2), (19, 1, 3)): 20,
        ((0, 2, 0), (19, 1, 3)): 22,
        ((2, 0, 0), (19, 1, 3)): 5,
        # 2-way ties for high
        ((0, 1, 1), (19, 1, 3)): 19,
        ((1, 0, 1), (19, 1, 3)): 1,
        ((1, 1, 0), (19, 1, 3)): 3,
        # no ties
        ((0, 1, 2), (19, 1, 3)): 19,
        ((1, 0, 2), (19, 1, 3)): 1,
        ((1, 2, 0), (19, 1, 3)): 3,
        ((0, 2, 1), (19, 1, 3)): 19,
        ((2, 0, 1), (19, 1, 3)): 1,
        ((2, 1, 0), (19, 1, 3)): 3,
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (19, 1, 4)): 25,
        # 2-way ties for low
        ((0, 0, 2), (19, 1, 4)): 20,
        ((0, 2, 0), (19, 1, 4)): 23,
        ((2, 0, 0), (19, 1, 4)): 6,
        # 2-way ties for high
        ((0, 1, 1), (19, 1, 4)): 19,
        ((1, 0, 1), (19, 1, 4)): 1,
        ((1, 1, 0), (19, 1, 4)): 4,
        # no ties
        ((0, 1, 2), (19, 1, 4)): 19,
        ((1, 0, 2), (19, 1, 4)): 1,
        ((1, 2, 0), (19, 1, 4)): 4,
        ((0, 2, 1), (19, 1, 4)): 19,
        ((2, 0, 1), (19, 1, 4)): 1,
        ((2, 1, 0), (19, 1, 4)): 4,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (19, 2, 3)): 26,
        # 2-way ties for low
        ((0, 0, 2), (19, 2, 3)): 21,
        ((0, 2, 0), (19, 2, 3)): 22,
        ((2, 0, 0), (19, 2, 3)): 7,
        # 2-way ties for high
        ((0, 1, 1), (19, 2, 3)): 19,
        ((1, 0, 1), (19, 2, 3)): 2,
        ((1, 1, 0), (19, 2, 3)): 3,
        # no ties
        ((0, 1, 2), (19, 2, 3)): 19,
        ((1, 0, 2), (19, 2, 3)): 2,
        ((1, 2, 0), (19, 2, 3)): 3,
        ((0, 2, 1), (19, 2, 3)): 19,
        ((2, 0, 1), (19, 2, 3)): 2,
        ((2, 1, 0), (19, 2, 3)): 3,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (19, 2, 4)): 27,
        # 2-way ties for low
        ((0, 0, 2), (19, 2, 4)): 21,
        ((0, 2, 0), (19, 2, 4)): 23,
        ((2, 0, 0), (19, 2, 4)): 8,
        # 2-way ties for high
        ((0, 1, 1), (19, 2, 4)): 19,
        ((1, 0, 1), (19, 2, 4)): 2,
        ((1, 1, 0), (19, 2, 4)): 4,
        # no ties
        ((0, 1, 2), (19, 2, 4)): 19,
        ((1, 0, 2), (19, 2, 4)): 2,
        ((1, 2, 0), (19, 2, 4)): 4,
        ((0, 2, 1), (19, 2, 4)): 19,
        ((2, 0, 1), (19, 2, 4)): 2,
        ((2, 1, 0), (19, 2, 4)): 4
    }
    
    partial_dp_mat, cur_cell_best_cum_cost, best_paths_mat = warmup_align_2(
        seq_1=seq_1,
        seq_2=seq_2,
        best_paths_mat=best_paths_mat,
        partial_dp_mat=partial_dp_mat,
        gap_open_cost=gap_open_cost,
        cost_mat=cost_mat,
        moves_for_gap_open_penalty_from_left=moves_for_gap_open_penalty_from_left,
        moves_for_gap_open_penalty_from_up=moves_for_gap_open_penalty_from_up,
        situation_mapper=situation_mapper
    )

    partial_dp_mat, cur_cell_best_cum_cost, best_paths_mat = do_core_align_2(
        seq_1=seq_1,
        seq_2=seq_2,
        best_paths_mat=best_paths_mat,
        partial_dp_mat=partial_dp_mat,
        gap_open_cost=gap_open_cost,
        cost_mat=cost_mat,
        moves_for_gap_open_penalty_from_left=moves_for_gap_open_penalty_from_left,
        moves_for_gap_open_penalty_from_up=moves_for_gap_open_penalty_from_up,
        situation_mapper=situation_mapper
    )

    print("best_paths_mat")
    for i in range(len(best_paths_mat)):
        print(best_paths_mat[i])

    seq_1_aligned_out, middle_part_out, seq_2_aligned_out = traceback_2(
        best_paths_mat=best_paths_mat,
        seq_1=seq_1,
        seq_2=seq_2
    )

    return (
        seq_1_aligned_out,
        middle_part_out,
        seq_2_aligned_out,
        cur_cell_best_cum_cost
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
        # print(path_indicator)
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
            # print("seq_1_index")
            # print(seq_1_index)
            # print("seq_2_index")
            # print(seq_2_index)
            break

    # print("seq_1_index")
    # print(seq_1_index)
    # print("seq_2_index")
    # print(seq_2_index)
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


def take_match(
    seq_1:str, 
    seq_2:str, 
    seq_1_index:int, 
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append("|")
    seq_2_aligned.append(seq_2[seq_2_index])

    return (
        seq_1_aligned,
        middle_part,
        seq_2_aligned
    )
    
def take_mismatch(
    seq_1:str, 
    seq_2:str, 
    seq_1_index:int, 
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append("*")
    seq_2_aligned.append(seq_2[seq_2_index])

    return (
        seq_1_aligned,
        middle_part,
        seq_2_aligned
    )

def take_gap_in_seq_1( 
    seq_1:str,
    seq_2:str, 
    seq_1_index:int,
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    seq_1_aligned.append("-")
    middle_part.append(" ")
    seq_2_aligned.append(seq_2[seq_2_index])

    return (
        seq_1_aligned,
        middle_part,
        seq_2_aligned
    )

def take_gap_in_seq_2( 
    seq_1:str,
    seq_2:str, 
    seq_1_index:int,
    seq_2_index:int, 
    seq_1_aligned:list, 
    middle_part:list,
    seq_2_aligned:list
):
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append(" ")
    seq_2_aligned.append("-")

    return (
        seq_1_aligned,
        middle_part,
        seq_2_aligned
    )


def traceback_2(
    best_paths_mat:list[list], 
    seq_1:str, 
    seq_2:str,
    # paths_to_moves_mapper gets the amounts by which 
    # the row and column indices 
    # in the edit graph need to change relative to the 
    # coordinates of the current cell.
    paths_to_moves_mapper={
        0: (-1, -1, take_match),
        1: (0, -1, take_gap_in_seq_1),
        2: (0, -1, take_gap_in_seq_1),
        3: (-1, 0, take_gap_in_seq_2),
        4: (-1, 0, take_gap_in_seq_2),
        5: random.choice(((0, -1, take_gap_in_seq_1), (-1, 0, take_gap_in_seq_2))),
        6: random.choice(((0, -1, take_gap_in_seq_1), (-1, 0, take_gap_in_seq_2))),
        7: random.choice(((0, -1, take_gap_in_seq_1), (-1, 0, take_gap_in_seq_2))),
        8: random.choice(((0, -1, take_gap_in_seq_1), (-1, 0, take_gap_in_seq_2))),
        9: (-1, -1, take_match),
        10: (-1, -1, take_match),
        11: (-1, -1, take_match),
        12: (-1, -1, take_match),
        13: random.choice(((0, -1, take_gap_in_seq_1), (0, -1, take_gap_in_seq_1))),
        14: random.choice(((-1, 0, take_gap_in_seq_2), (-1, 0, take_gap_in_seq_2))),
        15: (-1, -1, take_match),
        16: (-1, -1, take_match),
        17: (-1, -1, take_match),
        18: (-1, -1, take_match),
        19: (-1, -1, take_mismatch),
        20: (-1, -1, take_mismatch),
        21: (-1, -1, take_mismatch),
        22: (-1, -1, take_mismatch),
        23: (-1, -1, take_mismatch),
        24: (-1, -1, take_mismatch),
        25: (-1, -1, take_mismatch),
        26: (-1, -1, take_mismatch),
        27: (-1, -1, take_mismatch),
    }
) -> tuple[str, str, str]:
    """Perform traceback through best_paths_mat
    
    to find the alignment.

    Args: 
        best_paths_mat: list of length len(seq_1) + 1
            where each element is a list of length 
            len(seq_2) + 1

    Returns:
        (
            seq_1_aligned_out,
            middle_part_out,
            seq_2_aligned_out
        )
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
    best_paths_mat_row_index = m 
    best_paths_mat_col_index = n 
    # Because of the initial row and column in
    # best_paths_mat that doesn't align with
    # any parts of the two sequences, the indices
    # are off by one.
    seq_1_index = best_paths_mat_row_index - 1
    seq_2_index = best_paths_mat_col_index - 1

    # When we start at the bottom right of the 
    # best_paths_mat, there is no previous cell.
    # So, we just put path_indicator_prev = 0 
    # to initialize for the loop.
    path_indicator_prev = 0

    for w in range(max_num_alignment_moves):
        path_indicator_before_tie_break = best_paths_mat[best_paths_mat_row_index][best_paths_mat_col_index]
        # Sometimes the path_indicator may be for a tie
        # that we need to break.  
        path_indicator_after_tie_break, best_paths_mat = break_ties(
            best_paths_mat_row_index=best_paths_mat_row_index,
            best_paths_mat_col_index=best_paths_mat_col_index,
            best_paths_mat=best_paths_mat,
            path_indicator_prev=path_indicator_prev,
            path_indicator_curr=path_indicator_before_tie_break
        )
        best_paths_mat_row_index_delta, best_paths_mat_col_index_delta, move = paths_to_moves_mapper[path_indicator_after_tie_break]
        
        # Use the right move function.
        seq_1_aligned, middle_part, seq_2_aligned = move(
            seq_1=seq_1,
            seq_2=seq_2,
            seq_1_index=seq_1_index,
            seq_2_index=seq_2_index,
            seq_1_aligned=seq_1_aligned,
            middle_part=middle_part,
            seq_2_aligned=seq_2_aligned
        )

        # print("seq_1_aligned")
        # print(seq_1_aligned)
        # print("seq_2_aligned")
        # print(seq_2_aligned)
        
        # Update for next iteration.
        best_paths_mat_row_index += best_paths_mat_row_index_delta
        best_paths_mat_col_index += best_paths_mat_col_index_delta
        
        # Determine whether the loop should continue.
        if best_paths_mat_row_index < 1 and best_paths_mat_col_index < 1:
            # print("seq_1_index")
            # print(seq_1_index)
            # print("seq_2_index")
            # print(seq_2_index)
            break

        # Continue updating for next iteration.
        # Because of the initial row and column in
        # best_paths_mat which do not align with
        # any parts of the two sequences, the indices
        # are off by one.
        seq_1_index = best_paths_mat_row_index - 1
        seq_2_index = best_paths_mat_col_index - 1

        path_indicator_prev = path_indicator_after_tie_break

        
        # if path_indicator == 0:
        #     # match/mismatch is the best move
        #     seq_1_letter = seq_1[seq_1_index]
        #     seq_2_letter = seq_2[seq_2_index]
        #     if seq_1_letter == seq_2_letter:
        #         # There was a match.
        #         middle_part.append("|")
        #     else:
        #         # There was not a match.
        #         middle_part.append("*")

        #     seq_1_aligned.append(seq_1[seq_1_index])
        #     seq_1_index -= 1
        #     seq_2_aligned.append(seq_2[seq_2_index])
        #     seq_2_index -= 1
        # elif path_indicator == 1:
        #     # gap in seq_1 is the best move
        #     middle_part.append(" ")
        #     seq_1_aligned.append("-")
        #     seq_2_aligned.append(seq_2[seq_2_index])
        #     seq_2_index -= 1
        # else:
        #     # gap in seq_2 is the best move
        #     middle_part.append(" ")
        #     seq_1_aligned.append(seq_1[seq_1_index])
        #     seq_1_index -= 1
        #     seq_2_aligned.append("-")

        

    # print("seq_1_index")
    # print(seq_1_index)
    # print("seq_2_index")
    # print(seq_2_index)
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
    random.seed()
    # Randomly decide on how long the sequence should be.
    seq_len = random.randint(a=min_len, b=max_len)
    # Draw the desired number of letters from the alphabet.
    random_seq_list = random.choices(population=alphabet, k=seq_len)
    # Return the sequence as a string.
    return "".join(random_seq_list)

def draw_two_random_seqs(
    alphabet:list, 
    min_len_seq_1:int, 
    max_len_seq_1:int,
    min_len_seq_2:int, 
    max_len_seq_2:int,
    divergence:float
) -> list[str]:
    """
    Args:
        divergence: a number between 0 and 1, inclusive.
            Higher values for divergence will tend
            to make the two sequences more different
            from each other.
    """
    random.seed()
    seq_1 = draw_random_seq(
        alphabet=alphabet,
        min_len=min_len_seq_1,
        max_len=max_len_seq_1
    )

    len_seq_1 = len(seq_1)

    # seq_2 will just be a copy of seq_1 at first.
    seq_2_list = list(seq_1)
    
    # len_seq_2 is the length after all of the edits.
    len_seq_2 = random.randint(a=min_len_seq_2, b=max_len_seq_2)
    len_delta = len_seq_2 - len_seq_1
    initial_num_insertions = max(0, len_delta)
    initial_num_deletions = max(0, -len_delta)
    initial_num_substitutions = 0

    # Depending on divergence, we may want to do 
    # some additional edits to increase the 
    # distance between the two strings.
    additional_edit_ops = math.ceil(divergence * len_seq_2 / 3)

    num_insertions = initial_num_insertions + additional_edit_ops
    num_deletions = initial_num_deletions + additional_edit_ops
    num_substitutions = initial_num_substitutions + additional_edit_ops

    # With lower divergence, make it more likely
    # that we edit at the end of the sequence
    # so that the sequence is preserved as a 
    # sub-sequence.

    # Perform insertions.
    if num_insertions > 0:
        letters_to_insert = draw_random_seq(
            alphabet=alphabet, 
            min_len=num_insertions, 
            max_len=num_insertions
        )
        prob_insert_ends_only_on_insert = (1 - divergence)**(1/num_insertions)
    
    for i in range(num_insertions):
        # Prepare for iteration.
        len_seq_2_list = len(seq_2_list) 
        # Loop body
        
        rand = random.random()
        if rand < prob_insert_ends_only_on_insert/2:
            # Edit at left end.
            seq_2_index_for_insertion = 0
        elif rand < prob_insert_ends_only_on_insert:
            # Edit at right end.
            seq_2_index_for_insertion = len_seq_2_list
        else:
            # Edit in middle.
            middle_start = min(1, len_seq_2_list - 1)
            middle_end = max(1, len_seq_2_list - 1)
            seq_2_index_for_insertion = random.randint(
                a=middle_start, 
                b=middle_end
            )
        
        random_letter = letters_to_insert[i]
        seq_2_list.insert(seq_2_index_for_insertion, random_letter)

    # Perform deletions.
    if num_deletions > 0:
        prob_delete_ends_only_on_delete = (1 - divergence)**(1/num_deletions)
    for d in range(num_deletions):
        # Prepare for iteration.
        len_seq_2_list = len(seq_2_list) 
        # Loop body
        rand = random.random()
        if rand < prob_delete_ends_only_on_delete/2:
            # Edit at left end.
            seq_2_index_for_deletion = 0
        elif rand < prob_delete_ends_only_on_delete:
            # Edit at right end.
            seq_2_index_for_deletion = len_seq_2_list - 1
        else:
            # Edit in middle.
            middle_start = min(1, len_seq_2_list - 1)
            middle_end = max(middle_start, len_seq_2_list - 2)
            seq_2_index_for_deletion = random.randint(
                a=middle_start, 
                b=middle_end
            )

        seq_2_list.pop(seq_2_index_for_deletion)

    # Perform substitutions.
    if num_substitutions > 0:
        letters_to_sub = draw_random_seq(
            alphabet=alphabet, 
            min_len=num_substitutions, 
            max_len=num_substitutions
        )
        prob_sub_ends_only_on_sub = (1 - divergence)**(1/num_substitutions)

    for s in range(num_substitutions):
        # Prepare for iteration.
        len_seq_2_list = len(seq_2_list) 
        # Loop body
        rand = random.random()
        if rand < prob_sub_ends_only_on_sub/2:
            # Edit at left end.
            seq_2_index_for_sub = 0
        elif rand < prob_sub_ends_only_on_sub:
            # Edit at right end.
            seq_2_index_for_sub = len_seq_2_list - 1
        else:
            # Edit in middle.
            middle_start = min(1, len_seq_2_list - 1)
            middle_end = max(middle_start, len_seq_2_list - 2)
            seq_2_index_for_sub = random.randint(
                a=middle_start, 
                b=middle_end
            )

        seq_2_list[seq_2_index_for_sub] = letters_to_sub[s]

    seq_2 = "".join(seq_2_list)
    return [seq_1, seq_2]
    

def make_matrix(num_rows:int, num_cols:int, fill_val:int|float|str) -> list[list]:
    """Make a matrix as a nested list.
    
    See: https://www.freecodecamp.org/news/list-within-a-list-in-python-initialize-a-nested-list/
    """
    return [
        [fill_val]*(num_cols) for i in range(num_rows)
    ]


def break_ties(
    best_paths_mat_row_index:int,
    best_paths_mat_col_index:int,
    path_indicator_prev:int,
    path_indicator_curr:int,
    # best_paths_mat_row_index_delta:int,
    # best_paths_mat_col_index_delta:int,
    best_paths_mat:list[list],
    # seq_1:str,
    # seq_2:str,
    # gap_open_cost:int|float,
    # cost_mat:dict[dict],
    moves_for_gap_open_penalty_from_left:set={0, 3, 4, 11, 12, 14, 19, 22, 23},
    moves_for_gap_open_penalty_from_up:set={0, 1, 2, 9, 10, 13, 19, 20, 21},
    tie_fix_mapper:dict={
        # Keys are 2-tuples with elements of
        # path_indicator_prev,
        # path_indicator_curr.
        #
        # Values are integers (or None) indicating the best_path_type to 
        # the current cell in best_paths_mat.
        # If the value is None, then this indicates
        # that there is no way of getting such a combination
        # of path_indicator_prev and path_indicator_curr.
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (0, 4): 4,
        (0, 5): 5,
        (0, 6): 6,
        (0, 7): 7,
        (0, 8): 8,
        (0, 9): 9,
        (0, 10): 10,
        (0, 11): 11,
        (0, 12): 12,
        (0, 13): 13,
        (0, 14): 14,
        (0, 15): 15,
        (0, 16): 16,
        (0, 17): 17,
        (0, 18): 18,
        (0, 19): 19,
        (0, 20): 20,
        (0, 21): 21,
        (0, 22): 22,
        (0, 23): 23,
        (0, 24): 24,
        (0, 25): 25,
        (0, 26): 26,
        (0, 27): 27,
        (1, 0): 0,
        (1, 1): 1,
        (1, 2): 2,
        (1, 3): 3,
        (1, 4): 4,
        (1, 5): 3,
        (1, 6): 4,
        (1, 7): 3,
        (1, 8): 4,
        (1, 9): 0,
        (1, 10): 0,
        (1, 11): 11,
        (1, 12): 12,
        (1, 13): None,
        (1, 14): 14,
        (1, 15): 11,
        (1, 16): 12,
        (1, 17): 11,
        (1, 18): 12,
        (1, 19): 19,
        (1, 20): 19,
        (1, 21): 19,
        (1, 22): 22,
        (1, 23): 23,
        (1, 24): 22,
        (1, 25): 23,
        (1, 26): 22,
        (1, 27): 23,
        (2, 0): None,
        (2, 1): 1,
        (2, 2): 2,
        (2, 3): None,
        (2, 4): None,
        (2, 5): 1,
        (2, 6): 1,
        (2, 7): 2,
        (2, 8): 2,
        (2, 9): 1,
        (2, 10): 2,
        (2, 11): None,
        (2, 12): None,
        (2, 13): 13,
        (2, 14): None,
        (2, 15): 1,
        (2, 16): 1,
        (2, 17): 2,
        (2, 18): 2,
        (2, 19): None,
        (2, 20): 1,
        (2, 21): 2,
        (2, 22): None,
        (2, 23): None,
        (2, 24): 1,
        (2, 25): 1,
        (2, 26): 2,
        (2, 27): 2,
    }
) -> tuple[int, list[list]]:
    """Break ties in the best_paths_mat.
    
    This function is meant to be called during
    traceback of the best_paths_mat.
    As nodes in the edit graph (cells in best_paths_mat)
    are traced back along a minimum cost path,
    we know the direction in which each cell is entered,
    the position of each cell in best_paths_mat, path_indicator_prev, 
    and path_indicator_curr.
    Thus, (in some cases) we can infer what the best_path_type to the 
    current node should have been. So, in some
    cases, we can break ties.

    For example, if we trace back into a cell from the right
    that has a tie between taking a match and
    starting a gap in seq_1, (and the cell on the right
    says to continue a gap in seq_1), then the cell
    that we traced back into should not be a tie,
    but rather it should say to start a gap in seq_1.

    Args:
        path_indicator_prev: for the previous cell in the 
            traceback from end to start
        path_indicator_curr: for the current cell in the
            traceback from end to start

    Returns:
        (path_indicator_curr_new, best_paths_mat)
    """
    path_indicator_curr_new = tie_fix_mapper[(path_indicator_prev, path_indicator_curr)]
    best_paths_mat[best_paths_mat_row_index][best_paths_mat_col_index] = path_indicator_curr_new
    return (path_indicator_curr_new, best_paths_mat)

def find_best_path(
    i:int,
    j:int,
    best_paths_mat:list[list],
    partial_dp_mat:list[list],
    partial_dp_mat_prev_row_id:int,
    partial_dp_mat_cur_row_id:int,
    seq_1:str,
    seq_2:str,
    gap_open_cost:int|float,
    cost_mat:dict[dict],
    moves_for_gap_open_penalty_from_left:set={0, 3, 4, 11, 12, 14, 19, 22, 23},
    moves_for_gap_open_penalty_from_up:set={0, 1, 2, 9, 10, 13, 19, 20, 21},
    situation_mapper={
        # from_diag_best_path_type = 0
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (0, 1, 3)): 15,
        # 2-way ties for low
        ((0, 0, 2), (0, 1, 3)): 9,
        ((0, 2, 0), (0, 1, 3)): 11,
        ((2, 0, 0), (0, 1, 3)): 5,
        # 2-way ties for high
        ((0, 1, 1), (0, 1, 3)): 0,
        ((1, 0, 1), (0, 1, 3)): 1,
        ((1, 1, 0), (0, 1, 3)): 3,
        # no ties
        ((0, 1, 2), (0, 1, 3)): 0,
        ((1, 0, 2), (0, 1, 3)): 1,
        ((1, 2, 0), (0, 1, 3)): 3,
        ((0, 2, 1), (0, 1, 3)): 0,
        ((2, 0, 1), (0, 1, 3)): 1,
        ((2, 1, 0), (0, 1, 3)): 3,
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (0, 1, 4)): 16,
        # 2-way ties for low
        ((0, 0, 2), (0, 1, 4)): 9,
        ((0, 2, 0), (0, 1, 4)): 12,
        ((2, 0, 0), (0, 1, 4)): 6,
        # 2-way ties for high
        ((0, 1, 1), (0, 1, 4)): 0,
        ((1, 0, 1), (0, 1, 4)): 1,
        ((1, 1, 0), (0, 1, 4)): 4,
        # no ties
        ((0, 1, 2), (0, 1, 4)): 0,
        ((1, 0, 2), (0, 1, 4)): 1,
        ((1, 2, 0), (0, 1, 4)): 4,
        ((0, 2, 1), (0, 1, 4)): 0,
        ((2, 0, 1), (0, 1, 4)): 1,
        ((2, 1, 0), (0, 1, 4)): 4,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (0, 2, 3)): 17,
        # 2-way ties for low
        ((0, 0, 2), (0, 2, 3)): 10,
        ((0, 2, 0), (0, 2, 3)): 11,
        ((2, 0, 0), (0, 2, 3)): 7,
        # 2-way ties for high
        ((0, 1, 1), (0, 2, 3)): 0,
        ((1, 0, 1), (0, 2, 3)): 2,
        ((1, 1, 0), (0, 2, 3)): 3,
        # no ties
        ((0, 1, 2), (0, 2, 3)): 0,
        ((1, 0, 2), (0, 2, 3)): 2,
        ((1, 2, 0), (0, 2, 3)): 3,
        ((0, 2, 1), (0, 2, 3)): 0,
        ((2, 0, 1), (0, 2, 3)): 2,
        ((2, 1, 0), (0, 2, 3)): 3,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (0, 2, 4)): 18,
        # 2-way ties for low
        ((0, 0, 2), (0, 2, 4)): 10,
        ((0, 2, 0), (0, 2, 4)): 12,
        ((2, 0, 0), (0, 2, 4)): 8,
        # 2-way ties for high
        ((0, 1, 1), (0, 2, 4)): 0,
        ((1, 0, 1), (0, 2, 4)): 2,
        ((1, 1, 0), (0, 2, 4)): 4,
        # no ties
        ((0, 1, 2), (0, 2, 4)): 0,
        ((1, 0, 2), (0, 2, 4)): 2,
        ((1, 2, 0), (0, 2, 4)): 4,
        ((0, 2, 1), (0, 2, 4)): 0,
        ((2, 0, 1), (0, 2, 4)): 2,
        ((2, 1, 0), (0, 2, 4)): 4,
        # from_diag_best_path_type = 19
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (19, 1, 3)): 24,
        # 2-way ties for low
        ((0, 0, 2), (19, 1, 3)): 20,
        ((0, 2, 0), (19, 1, 3)): 22,
        ((2, 0, 0), (19, 1, 3)): 5,
        # 2-way ties for high
        ((0, 1, 1), (19, 1, 3)): 19,
        ((1, 0, 1), (19, 1, 3)): 1,
        ((1, 1, 0), (19, 1, 3)): 3,
        # no ties
        ((0, 1, 2), (19, 1, 3)): 19,
        ((1, 0, 2), (19, 1, 3)): 1,
        ((1, 2, 0), (19, 1, 3)): 3,
        ((0, 2, 1), (19, 1, 3)): 19,
        ((2, 0, 1), (19, 1, 3)): 1,
        ((2, 1, 0), (19, 1, 3)): 3,
        # from_left_best_path_type == 1
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (19, 1, 4)): 25,
        # 2-way ties for low
        ((0, 0, 2), (19, 1, 4)): 20,
        ((0, 2, 0), (19, 1, 4)): 23,
        ((2, 0, 0), (19, 1, 4)): 6,
        # 2-way ties for high
        ((0, 1, 1), (19, 1, 4)): 19,
        ((1, 0, 1), (19, 1, 4)): 1,
        ((1, 1, 0), (19, 1, 4)): 4,
        # no ties
        ((0, 1, 2), (19, 1, 4)): 19,
        ((1, 0, 2), (19, 1, 4)): 1,
        ((1, 2, 0), (19, 1, 4)): 4,
        ((0, 2, 1), (19, 1, 4)): 19,
        ((2, 0, 1), (19, 1, 4)): 1,
        ((2, 1, 0), (19, 1, 4)): 4,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 3
        # 3-way ties
        ((0, 0, 0), (19, 2, 3)): 26,
        # 2-way ties for low
        ((0, 0, 2), (19, 2, 3)): 21,
        ((0, 2, 0), (19, 2, 3)): 22,
        ((2, 0, 0), (19, 2, 3)): 7,
        # 2-way ties for high
        ((0, 1, 1), (19, 2, 3)): 19,
        ((1, 0, 1), (19, 2, 3)): 2,
        ((1, 1, 0), (19, 2, 3)): 3,
        # no ties
        ((0, 1, 2), (19, 2, 3)): 19,
        ((1, 0, 2), (19, 2, 3)): 2,
        ((1, 2, 0), (19, 2, 3)): 3,
        ((0, 2, 1), (19, 2, 3)): 19,
        ((2, 0, 1), (19, 2, 3)): 2,
        ((2, 1, 0), (19, 2, 3)): 3,
        # from_left_best_path_type == 2
        # and from_up_best_path_type == 4
        # 3-way ties
        ((0, 0, 0), (19, 2, 4)): 27,
        # 2-way ties for low
        ((0, 0, 2), (19, 2, 4)): 21,
        ((0, 2, 0), (19, 2, 4)): 23,
        ((2, 0, 0), (19, 2, 4)): 8,
        # 2-way ties for high
        ((0, 1, 1), (19, 2, 4)): 19,
        ((1, 0, 1), (19, 2, 4)): 2,
        ((1, 1, 0), (19, 2, 4)): 4,
        # no ties
        ((0, 1, 2), (19, 2, 4)): 19,
        ((1, 0, 2), (19, 2, 4)): 2,
        ((1, 2, 0), (19, 2, 4)): 4,
        ((0, 2, 1), (19, 2, 4)): 19,
        ((2, 0, 1), (19, 2, 4)): 2,
        ((2, 1, 0), (19, 2, 4)): 4
    }
) -> tuple[int, int|float]:
    """Find the best path to align two prefixes
    
    given the best paths in the alignments
    of the prefixes that come before them.
    That is, given the best alignments of
    the following three prefixes:
        seq_1[0:i-1] and seq_2[0:j-1]
        seq_1[0:i] and seq_2[0:j-1]
        seq_1[0:i-1] and seq_2[0:j]
    find the best alignment of
        seq_1[0:i] and seq_2[0:j].
    
    Args:
        situation_mapper: dict. Keys are tuples
            of the form (possible_cum_cost_ranks, 
            (from_diag_best_path_type, 
            from_left_best_path_type, 
            from_up_best_path_type)).

    Returns:
        The tuple (best_path_type, best_cum_cost),
        where best_path_type is one of the following:
            0: match
            1: starting gap in seq_1
            2: continuing gap in seq_1
            3: starting gap in seq_2
            4: continuing gap in seq_2
            5: tie between 1 and 3
            6: tie between 1 and 4
            7: tie between 2 and 3
            8: tie between 2 and 4
            9: tie between 0 and 1
            10: tie between 0 and 2
            11: tie between 0 and 3
            12: tie between 0 and 4
            13: tie between 1 and 2
            14: tie between 3 and 4
            15: tie between 0, 1, and 3
            16: tie between 0, 1, and 4
            17: tie between 0, 2, and 3
            18: tie between 0, 2, and 4
            19: mismatch
            20: tie between 19 and 1
            21: tie between 19 and 2
            22: tie between 19 and 3
            23: tie between 19 and 4
            24: tie between 19, 1, and 3
            25: tie between 19, 1, and 4
            26: tie between 19, 2, and 3
            27: tie between 19, 2, and 4
        and best_cum_cost is the cumulative cost in the 
        optimal alignment of seq_1[0:i] and 
        seq_2[0:j]
    """
    partial_dp_mat_prev_col_id = j - 1
    partial_dp_mat_cur_col_id = j

    # The indices into our sequences are always 1 behind
    # the indices into our dynamic programming matrices.
    seq_1_index = i - 1
    seq_2_index = j - 1
    
    best_paths_mat_prev_row_id = i - 1
    best_paths_mat_cur_row_id = i
    best_paths_mat_prev_col_id = partial_dp_mat_prev_col_id
    best_paths_mat_cur_col_id = j
    
    diag_best_path_type = best_paths_mat[best_paths_mat_prev_row_id][best_paths_mat_prev_col_id]
    diag_best_cost = partial_dp_mat[partial_dp_mat_prev_row_id][partial_dp_mat_prev_col_id]
    from_diag_best_cost = diag_best_cost + cost_mat[seq_1[seq_1_index]][seq_2[seq_2_index]]
    # from_diag_best_path_type is the path_type that we would
    # have for the current cell if we accepted a match/mismatch
    # (and there were no ties).
    if seq_1[seq_1_index] == seq_2[seq_2_index]:
        from_diag_best_path_type = 0
    else:
        from_diag_best_path_type = 19

    left_best_path_type = best_paths_mat[best_paths_mat_cur_row_id][best_paths_mat_prev_col_id]
    left_best_cost = partial_dp_mat[partial_dp_mat_cur_row_id][partial_dp_mat_prev_col_id]

    # In calculating the best path to the current
    # cell, we must worry about ties in 
    # left_best_path_type and up_best_path_type.
    if left_best_path_type in moves_for_gap_open_penalty_from_left:
        # Pay for gap opening.
        gap_extension_cost = cost_mat["-"][seq_2[seq_2_index]]
        from_left_best_cost = left_best_cost + gap_open_cost + gap_extension_cost
        # 1: starting gap in seq_1
        from_left_best_path_type = 1
    else:
        # It is not required to open a gap
        # to get to the current cell.
        # Do not pay for gap opening.
        gap_extension_cost = cost_mat["-"][seq_2[seq_2_index]]
        from_left_best_cost = left_best_cost + gap_extension_cost
        # 2: continuing gap in seq_1
        from_left_best_path_type = 2

    up_best_path_type = best_paths_mat[best_paths_mat_prev_row_id][best_paths_mat_cur_col_id]
    up_best_cost = partial_dp_mat[partial_dp_mat_prev_row_id][partial_dp_mat_cur_col_id]
    if up_best_path_type in moves_for_gap_open_penalty_from_up:
        # Pay for gap opening.
        gap_extension_cost = cost_mat[seq_1[seq_1_index]]["-"]
        from_up_best_cost = up_best_cost + gap_open_cost + gap_extension_cost
        from_up_best_path_type = 3    
    else:
        # Do not pay for gap opening.
        gap_extension_cost = cost_mat[seq_1[seq_1_index]]["-"]
        from_up_best_cost = up_best_cost + gap_extension_cost
        from_up_best_path_type = 4

    # possible_cum_cost is for the current cell.
    possible_cum_cost = [
        from_diag_best_cost,
        from_left_best_cost,
        from_up_best_cost
    ]
    # Note that at this point, we have:
    # from_diag_best_path_type is 0 or 19
    # from_left_best_path_type is 1 or 2
    # from_up_best_path_type is 3 or 4
   
    # Get the ranks of the values in possible_cum_cost.
    # A minimum value in possible_cum_cost will be assigned
    # a rank of 0.  Larger values will be assigned higher ranks.
    # In the case of ties, ranks are repeated.  For example,
    # for [3, 3, 9], the ranks are (0, 0, 2).
    possible_cum_cost_sorted = sorted(possible_cum_cost)
    # Rank trick: https://stackoverflow.com/a/53661474/8423001
    possible_cum_cost_ranks = tuple(
        possible_cum_cost_sorted.index(k) for k in possible_cum_cost
    )

    # Map the ranks and what would be the 3 best paths
    # to possibly put in the current cell into
    # a path type to put in the current cell.
    best_path_type = situation_mapper[(
        possible_cum_cost_ranks, 
        (from_diag_best_path_type, from_left_best_path_type, from_up_best_path_type)
    )]

    # It's sorted in ascending order,
    # so a minimum value is at the beginning.
    best_cum_cost = possible_cum_cost_sorted[0]

    return (best_path_type, best_cum_cost)
    
    

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
    max_possible_new_score_indices = [i for i, s 
        in enumerate(possible_new_scores)
        if s == max_possible_new_score]
    best_type_of_path = random.choice(max_possible_new_score_indices)
    #######

    # # possible_new_scores_sorted is in ascending order. 
    # possible_new_scores_sorted = sorted(possible_new_scores)

    # # Because possible_new_scores_sorted is in ascending order,
    # # the last entry is a maximum.
    # # max_possible_new_score = possible_new_scores_sorted[-1]
    # # Check for ties
    # if max_possible_new_score != possible_new_scores_sorted[-2]:
    #     # The maximum is unique.
    #     # Find the index corresponding to where the maximum
    #     # is achieved.
    #     best_type_of_path = possible_new_scores.index(max_possible_new_score)
    # elif max_possible_new_score != possible_new_scores_sorted[-3]:
    #     # The max was achieved 2 times.
    #     # Break the 2-way tie randomly.
    #     possible_new_scores.index(max_possible_new_score)
    # else:
    #     # The max was achieved 3 times.
    #     # Break the 3-way tie randomly.
    #     best_type_of_path = random.randint(0, 2)
    # # if max_possible_new_score == possible_new_scores_sorted[-2]:
    # #     # The max was achieved at least twice.
    # #     # Check if it was achieved 3 times.
    # #     if max_possible_new_score == possible_new_scores_sorted[-3]:
            
    
    
    
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
    D_mat:list[list],
    best_paths_mat:list[list],
    score:int|float
) -> tuple[list[list], list[list], list[list], list[list], list[list], int]:
    """
    Find a global alignment of the subsequences
    
    seq_1[1:] (assuming len(seq_1) > 1) and seq_2.  
    If len(seq_1) == 1, then no additional aligning
    is done.

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
        D_mat:
        best_paths_mat: already filled for the first two
            rows from an initial run of the algorithm.

    Returns:
        (
            partial_A_mat,
            partial_B_mat,
            partial_C_mat,
            D_mat,
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
        D_mat[i][j] = max(
            partial_A_mat[partial_mat_cur_row_id][j],
            partial_B_mat[partial_mat_cur_row_id][j],
            partial_C_mat[partial_mat_cur_row_id][j],
        )
        # Update the 1-index columns based on how gaps are penalized.
        j = 1
        seq_2_index = j - 1
        
        # There's no need for the usual max operation
        # for partial_A_mat because partial_A_mat,
        # partial_B_mat, and partial_C_mat all have the same 0-index
        # column.
        prev_best = partial_A_mat[partial_mat_prev_row_id][j - 1]
        partial_A_mat[partial_mat_cur_row_id][j] = scoring_mat[seq_1[seq_1_index]][seq_2[seq_2_index]] + prev_best
        
        # The gap existence cost is always paid for partial_B_mat
        # because j == 1.
        # There couldn't have been a pre-existing gap in seq_1.
        partial_B_mat[partial_mat_cur_row_id][j] = partial_B_mat[partial_mat_cur_row_id][j - 1] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost
        
        # Consider partial_C_mat
        partial_C_mat[partial_mat_cur_row_id][j] = max(
            partial_A_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_B_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] - gap_existence_cost,
            partial_C_mat[partial_mat_prev_row_id][j] + scoring_mat["-"][seq_2[seq_2_index]] 
        )
        D_mat[i][j] = max(
            partial_A_mat[partial_mat_cur_row_id][j],
            partial_B_mat[partial_mat_cur_row_id][j],
            partial_C_mat[partial_mat_cur_row_id][j],
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
            D_mat[i][j] = max(
                partial_A_mat[partial_mat_cur_row_id][j],
                partial_B_mat[partial_mat_cur_row_id][j],
                partial_C_mat[partial_mat_cur_row_id][j],
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
        D_mat,
        best_paths_mat,
        score
    )


def do_core_align_2(
    seq_1:str, 
    seq_2:str, 
    best_paths_mat:list[list],
    partial_dp_mat:list[list],
    gap_open_cost,
    cost_mat:dict[dict],
    moves_for_gap_open_penalty_from_left,
    moves_for_gap_open_penalty_from_up,
    situation_mapper
) -> tuple[list[list], int|float, list[list]]:
    """
    Find a global alignment of the sequences
    
    `seq_1` and `seq_2`, assuming that the 
    lengths of both sequences are greater than 1.  

    Args:
        gap_open_cost: The cost for a gap just to exist.
            This cost should be non-negative.
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment.

    Returns:
        (
            partial_dp_mat,
            cur_cell_best_cum_cost,
            best_paths_mat
        )
    """
    # Prep for loop
    row_range = range(2, len(seq_1) + 1)
    core_col_range = range(1, len(seq_2) + 1)

    partial_dp_mat_prev_row_id = 1
    partial_dp_mat_cur_row_id = 0

    for i in row_range:
        # Prep for this row iteration.
        best_paths_mat_cur_row_id = i
        # The indices into our sequences are always 1 behind
        # the indices into our dynamic programming matrices.
        seq_1_index = i - 1
        # Figure out the cost of extending 
        # a gap in seq_2 based on where we 
        # are in seq_1.
        # Unless the cost_mat is wacky, this
        # will always be the same number.
        gap_extension_cost_seq_2 = cost_mat[seq_1[seq_1_index]]["-"]

        ###############################################
        # Handle the 0-index column first.
        ###############################################
        j = 0
        # Prep for this column iteration.
        partial_dp_mat_cur_col_id = j
        best_paths_mat_cur_col_id = j
        # The best (only) thing to do is to
        # continue a gap in seq_2.
        cur_cell_best_path_type = 4
        cur_cell_best_cum_cost = partial_dp_mat[partial_dp_mat_prev_row_id][partial_dp_mat_cur_col_id] + gap_extension_cost_seq_2
        # Update best_paths_mat.
        best_paths_mat[best_paths_mat_cur_row_id][best_paths_mat_cur_col_id] = cur_cell_best_path_type
        # Update partial_dp_mat.
        partial_dp_mat[partial_dp_mat_cur_row_id][partial_dp_mat_cur_col_id] = cur_cell_best_cum_cost
        ###############################################
        # Handle the other columns.
        ###############################################
        for j in core_col_range:
            # Prep for this column iteration.
            partial_dp_mat_cur_col_id = j
            best_paths_mat_cur_col_id = j
            # Loop body
            cur_cell_best_path_type, cur_cell_best_cum_cost = find_best_path(
                i=i,
                j=j,
                best_paths_mat=best_paths_mat,
                partial_dp_mat=partial_dp_mat,
                partial_dp_mat_prev_row_id=partial_dp_mat_prev_row_id,
                partial_dp_mat_cur_row_id=partial_dp_mat_cur_row_id,
                seq_1=seq_1,
                seq_2=seq_2,
                gap_open_cost=gap_open_cost,
                cost_mat=cost_mat,
                moves_for_gap_open_penalty_from_left=moves_for_gap_open_penalty_from_left,
                moves_for_gap_open_penalty_from_up=moves_for_gap_open_penalty_from_up,
                situation_mapper=situation_mapper
            )

            # Update best_paths_mat.
            best_paths_mat[best_paths_mat_cur_row_id][best_paths_mat_cur_col_id] = cur_cell_best_path_type

            # Update partial_dp_mat.
            partial_dp_mat[partial_dp_mat_cur_row_id][partial_dp_mat_cur_col_id] = cur_cell_best_cum_cost

        # Prep for next row iteration.
        # Do some swapping.
        print(f"partial_dp_mat with cur_row: {partial_dp_mat_cur_row_id}")
        print(partial_dp_mat[partial_dp_mat_cur_row_id])
        partial_dp_mat_prev_row_id, partial_dp_mat_cur_row_id = partial_dp_mat_cur_row_id, partial_dp_mat_prev_row_id

    return (
        partial_dp_mat,
        cur_cell_best_cum_cost,
        best_paths_mat
    )

def warmup_align_2(
    seq_1:str, 
    seq_2:str, 
    best_paths_mat:list[list],
    partial_dp_mat:list[list],
    gap_open_cost,
    cost_mat:dict[dict],
    moves_for_gap_open_penalty_from_left,
    moves_for_gap_open_penalty_from_up,
    situation_mapper
) -> tuple[list[list], int|float, list[list]]:
    """Find a global alignment of the subsequences
    
    `seq_1[0]` and `seq_2`, assuming that the 
    lengths of both sequences are greater than 1.  

    Args:
        gap_open_cost: The cost for a gap just to exist.
            This cost should be non-negative.
            It can be incurred multiple times
            if there are multiple runs of gaps in the
            alignment.

    Returns:
        (
            partial_dp_mat,
            cur_cell_best_cum_cost,
            best_paths_mat
        )
    """
    # Prep for loop
    col_range = range(1, len(seq_2) + 1)

    partial_dp_mat_prev_row_id = 0
    partial_dp_mat_cur_row_id = 1
    i = 1
    # Prep for this row iteration.
    best_paths_mat_cur_row_id = i

    for j in col_range:
        # Prep for this column iteration.
        partial_dp_mat_cur_col_id = j
        best_paths_mat_cur_col_id = j
        # Loop body
        cur_cell_best_path_type, cur_cell_best_cum_cost = find_best_path(
            i=i,
            j=j,
            best_paths_mat=best_paths_mat,
            partial_dp_mat=partial_dp_mat,
            partial_dp_mat_prev_row_id=partial_dp_mat_prev_row_id,
            partial_dp_mat_cur_row_id=partial_dp_mat_cur_row_id,
            seq_1=seq_1,
            seq_2=seq_2,
            gap_open_cost=gap_open_cost,
            cost_mat=cost_mat,
            moves_for_gap_open_penalty_from_left=moves_for_gap_open_penalty_from_left,
            moves_for_gap_open_penalty_from_up=moves_for_gap_open_penalty_from_up,
            situation_mapper=situation_mapper
        )

        # Update best_paths_mat.
        best_paths_mat[best_paths_mat_cur_row_id][best_paths_mat_cur_col_id] = cur_cell_best_path_type

        # Update partial_dp_mat.
        partial_dp_mat[partial_dp_mat_cur_row_id][partial_dp_mat_cur_col_id] = cur_cell_best_cum_cost

    # Prep for next row iteration.
    # Do some swapping.
    partial_dp_mat_prev_row_id, partial_dp_mat_cur_row_id = partial_dp_mat_cur_row_id, partial_dp_mat_prev_row_id

    return (
        partial_dp_mat,
        cur_cell_best_cum_cost,
        best_paths_mat
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
    D_mat:list[list],
    best_paths_mat:list[list]
) -> tuple[list[list], list[list], list[list], list[list], list[list], int|float]:
    """
    Find a global alignment of the subsequences
    
    seq_1[0] and seq_2.  In other words, solve some of the 
    subproblmes relevant to the global alignment of seq_1
    and seq_2.

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
        D_mat: 
        best_paths_mat: already initialized for 0-index row and 
            0-index column.

    Returns:
        (
            partial_A_mat,
            partial_B_mat,
            partial_C_mat,
            D_mat,
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
    
    D_mat[i][j] = max(
        partial_A_mat[partial_mat_cur_row_id][j],
        partial_B_mat[partial_mat_cur_row_id][j],
        partial_C_mat[partial_mat_cur_row_id][j],
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

        D_mat[i][j] = max(
            partial_A_mat[partial_mat_cur_row_id][j],
            partial_B_mat[partial_mat_cur_row_id][j],
            partial_C_mat[partial_mat_cur_row_id][j],
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
        D_mat,
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
    """Initialze matrices for the NW algorithm. 
    
    It initializes correctly for the 0-index rows and the 0-index 
    columns.  However, it leaves complete initialization of 
    partial_A_mat, partial_B_mat, and partial_C_mat to other 
    functions.
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


def init_partial_dynamic_prog_matrix_2(
    seq_1:str,
    seq_2:str,
    cost_mat:dict[dict], 
    gap_open_cost:int|float,
    dynamic_prog_num_cols:int
) -> list[list]:
    """Initialize a partial dynamic programming

    matrix.  It's partial because it only has 2 rows.
    It initializes correctly for the 0-index row and the 
    entry in the 1-index row and 0-index column.
    """
    mat = make_matrix(
        num_rows=2,
        num_cols=dynamic_prog_num_cols,
        fill_val=0
    )

    # Take care of initialization with gap costs.
  
    # Loop prep
    # Start in column 1
    j = 1

    # The indices into our sequences are always 1 behind
    # the indices into our dynamic programming matrices.
    seq_2_index = j - 1

    # Pay gap_open_cost for the entry in the 
    # 0th row and 1st column.
    cur_gap_score = gap_open_cost + cost_mat["-"][seq_2[seq_2_index]]
    mat[0][j] = cur_gap_score

    # We do not have to pay the gap_open_cost for 
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
        cur_gap_score = cur_gap_score + cost_mat["-"][seq_2[seq_2_index]]
        mat[0][j] = cur_gap_score

    # We also have to pay the gap_open_cost for 
    # the entry in the 1-index row and 0-index column.
    mat[1][0] = gap_open_cost + cost_mat[seq_1[0]]["-"]
    return mat


def init_D_mat(
    dynamic_prog_num_rows:int,
    dynamic_prog_num_cols:int,
    partial_A_mat:list[list],
    partial_B_mat:list[list],
    partial_C_mat:list[list]    
) -> list[list]:
    """Initialize a matrix where the entry in row i and column j
    
    indicates the best final score for the alignment of the 
    subsequences seq_1[0:(i - 1)] and seq_2[0:(j - 1)].
    The matrix representations of partial_A_mat, partial_B_mat,
    and partial_C_mat should all have the same dimensions.
    The length of each of partial_A_mat, partial_B_mat,
    and partial_C_mat should be 2.
    
    Args:
        dynamic_prog_num_rows: len(seq_1) + 1
        dynamic_prog_num_cols: len(seq_2) + 1
    Returns:
        D_mat as a nested list where each entry in the list
            is a list representing a row of D_mat.
    """
    D_mat = make_matrix(
        num_rows=dynamic_prog_num_rows,
        num_cols=dynamic_prog_num_cols,
        fill_val=0
    )

    # D_mat[i][j] is always max(
    #     partial_A_mat[i][j],
    #     partial_B_mat[i][j],
    #     partial_C_mat[i][j],
    # )
    # but because the other matrices only
    # have 2 rows, and because the other matrices
    # are assumed to only be initialized properly
    # for the 0-index rows and 0-index columns,
    # we only fill in some of the entries 
    # of D_mat.
    i = 0
    for j in range(dynamic_prog_num_cols):
        D_mat[i][j] = max(
            partial_A_mat[i][j],
            partial_B_mat[i][j],
            partial_C_mat[i][j],
        )
    i = 1
    j = 0
    D_mat[i][j] = max(
            partial_A_mat[i][j],
            partial_B_mat[i][j],
            partial_C_mat[i][j],
        )

    return D_mat


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
        There are multiple possible values for each entry in the best_paths_mat
        to indicate an alignment "move" such as:
            0: match/mismatch
            1: starting gap in seq_1
            2: continuing gap in seq_1
            3: starting gap in seq_2
            4: continuing gap in seq_2
    """
    best_paths_mat = make_matrix(
        num_rows=dynamic_prog_num_rows,
        num_cols=dynamic_prog_num_cols,
        fill_val=2
    )

    # Update some entries specifically.
    best_paths_mat[0][1] = 1
    best_paths_mat[1][0] = 3

    for i in range(2, dynamic_prog_num_rows):
        best_paths_mat[i][0] = 4

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