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
from copy import deepcopy

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
    
    # Get the cost_mat.
    max_score = get_max_similarity_score(scoring_mat=scoring_mat)

    cost_mat = get_cost_mat(
        scoring_mat=scoring_mat,
        max_score=max_score
    )

    # Perform the alignment, insert gaps, and compute the score.
    alignment = find_global_alignment(
        seq_1=seq_1,
        seq_2=seq_2,
        cost_mat=cost_mat,
        gap_open_cost=gap_existence_cost
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


def get_cost_mat(
    scoring_mat:dict[dict], 
    max_score:int|float,
    delta_d:int|float=None,
    delta_i:int|float=None
) -> dict[dict]:
    """Get a valid cost matrix from a scoring matrix.

    The cost matrix will be a valid distance matrix.
    
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
    # Make sure we don't mutate the scoring_mat
    cost_mat = deepcopy(scoring_mat)
    b = max_score
    if delta_d is None:
        delta_d = math.floor(b/2)
    if delta_i is None:
        delta_i = math.ceil(b/2)
    
    for seq_1_letter, seq_2_scores in cost_mat.items():
        # seq_1_letter is a key for cost_mat.
        # seq_2_scores is the inner dict for the outer
        # key of seq_1_letter.
        for seq_2_letter, score in seq_2_scores.items():
            # The scores are transformed differently
            # for insertions and deletions, than they
            # are for matches and mismatches.
            if seq_1_letter == "-" and seq_2_letter != "-":
                # Update deletions (horizontal steps)
                seq_2_scores[seq_2_letter] = -score + delta_d
            elif seq_2_letter == "-" and seq_1_letter != "-":
                # Update insertions (vertical steps)
                seq_2_scores[seq_2_letter] = -score + delta_i 
            else:
                # Update matches and mismatches.
                seq_2_scores[seq_2_letter] = -score + delta_d + delta_i
   
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


def find_global_alignment(
    seq_1:str,
    seq_2:str,
    cost_mat:dict[dict],
    gap_open_cost:int|float
) -> tuple[str, str, str, int|float]:
    """
    Args:
        cost_mat: keys are symbols representing nucleotides
            or amino acid residues. A symbol of '-' is used
            for a gap. The inner dict contains the same 
            keys as the outer dict and contains values
            that are numbers representing edit costs.
            For example, cost_mat["-"]["A"] is the cost
            for inserting a gap in seq_1 while accepting
            an "A" from seq_2 and cost_mat["T"]["C"]
            is the cost of a mismatch of a "T" in seq_1
            and a "C" in seq_2.
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
    # Imagine a 3-d parking garage like in Mario.
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
    dp_array = make_dp_array(
        seq_1=seq_1,
        seq_2=seq_2,
        cost_mat=cost_mat,
        gap_open_cost=gap_open_cost
    )

    # Loop through the dp_array and write the
    # best costs to get to each position.
    dp_array_forward(
        dp_array=dp_array,
        seq_1=seq_1,
        seq_2=seq_2,
        cost_mat=cost_mat,
        gap_open_cost=gap_open_cost
    )

    # Traceback the dp_array to determine
    # the sequence of moves in reverse
    # order needed to produce an optimal alignment.


    return (
        seq_1_aligned_out,
        middle_part_out,
        seq_2_aligned_out,
        cur_cell_best_cum_cost
    )


def dp_array_forward(
    dp_array:list[list[list]],
    seq_1:str,
    seq_2:str,
    cost_mat:dict[dict],
    gap_open_cost:int|float
):
    """
    Operates in place on the dp_array.
    """
    # Prepare for loop.
    dim_1 = len(seq_1) + 1
    dim_2 = len(seq_2) + 1
    for i in range(1, dim_1):
        seq_1_index = i - 1
        for j in range(1, dim_2):
            seq_2_index = j - 1
            ######################################
            level = 0
            previous_costs = (
                dp_array[i - 1][j - 1][0],
                dp_array[i - 1][j - 1][1],
                dp_array[i - 1][j - 1][2]
            )
            new_cost = cost_mat[seq_1[seq_1_index]][seq_2[seq_2_index]]
            dp_array[i][j][level] = min(previous_costs) + new_cost
            ######################################
            level = 1
            previous_costs = (
                dp_array[i][j - 1][0] + gap_open_cost,
                dp_array[i][j - 1][1],
                dp_array[i][j - 1][2] + gap_open_cost
            )
            new_cost = cost_mat["-"][seq_2[seq_2_index]]
            dp_array[i][j][level] = min(previous_costs) + new_cost
            ######################################
            level = 2
            previous_costs = (
                dp_array[i - 1][j][0] + gap_open_cost,
                dp_array[i - 1][j][1] + gap_open_cost,
                dp_array[i - 1][j][2]
            )
            new_cost = cost_mat[seq_1[seq_1_index]]["-"]
            dp_array[i][j][level] = min(previous_costs) + new_cost
            ######################################

    return None

def dp_array_backward(
    dp_array: list[list[list]],
    seq_1: str,
    seq_2: str
):
    """
    Traces backward through the dp_array

    to determine which alignment moves are best.
    """
    seq_1_aligned = []
    seq_2_aligned = []
    middle_part = []    
    
    # Prepare for loop.
    dim_1 = len(seq_1) + 1
    dim_2 = len(seq_2) + 1
    max_num_alignment_moves = dim_1 + dim_2 - 1
    i = dim_1 - 1
    j = dim_2 - 1
    for h in range(max_num_alignment_moves):
        # Find the dp_array values to compare.
        # This depends on which cell was selected 
        # as the best last time. 
        costs_to_compare = dp_array[i][j]
        seq_1_index = i - 1
        seq_2_index = j - 1
        # Find a minimum of the dp_array values compared.
        # Randomly break ties.
        # https://stackoverflow.com/a/53661474/8423001
        cost_ranks = [sorted(costs_to_compare).index(x) for x in costs_to_compare]
        is_match = (seq_1[seq_1_index] == seq_2[seq_2_index])
        # Figure out the move to make in the alignment graph.
        move, delta_i, delta_j = cost_ranks_dispatcher(
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
        ((2, 1, 0), True): take_gap_in_seq_2,
        ((0, 2, 2), True): take_match,
        ((2, 0, 2), True): take_gap_in_seq_1,
        ((2, 2, 0), True): take_gap_in_seq_2,
        ((1, 1, 1), True): random.choice((take_match, take_gap_in_seq_1, take_gap_in_seq_2)),
        ((1, 1, 2), True): random.choice((take_match, take_gap_in_seq_1)),
        ((1, 2, 1), True): random.choice((take_match, take_gap_in_seq_2)),
        ((2, 1, 1), True): random.choice((take_gap_in_seq_1, take_gap_in_seq_2)),
        ((1, 2, 2), True): take_match,
        ((2, 1, 2), True): take_gap_in_seq_1,
        ((2, 2, 1), True): take_gap_in_seq_2,
        ((2, 2, 2), True): random.choice((take_match, take_gap_in_seq_1, take_gap_in_seq_2)),
        ((0, 0, 0), True): random.choice((take_mismatch, take_gap_in_seq_1, take_gap_in_seq_2)),
        ((0, 0, 1), True): take_gap_in_seq_2,
        ((0, 1, 0), True): take_gap_in_seq_1,
        ((1, 0, 0), True): take_mismatch,
        ((0, 0, 2), True): random.choice((take_mismatch, take_gap_in_seq_1)),
        ((0, 2, 0), True): random.choice((take_mismatch, take_gap_in_seq_2)),
        ((2, 1, 0), True): take_gap_in_seq_2,
        ((0, 2, 2), True): take_mismatch,
        ((2, 0, 2), True): take_gap_in_seq_1,
        ((2, 2, 0), True): take_gap_in_seq_2,
        ((1, 1, 1), True): random.choice((take_mismatch, take_gap_in_seq_1, take_gap_in_seq_2)),
        ((1, 1, 2), True): random.choice((take_mismatch, take_gap_in_seq_1)),
        ((1, 2, 1), True): random.choice((take_mismatch, take_gap_in_seq_2)),
        ((2, 1, 1), True): random.choice((take_gap_in_seq_1, take_gap_in_seq_2)),
        ((1, 2, 2), True): take_mismatch,
        ((2, 1, 2), True): take_gap_in_seq_1,
        ((2, 2, 1), True): take_gap_in_seq_2,
        ((2, 2, 2), True): random.choice((take_mismatch, take_gap_in_seq_1, take_gap_in_seq_2)),
    }

    delta_dispatch_dict = {
        take_match: (-1, -1),
        take_mismatch: (-1, -1),
        take_gap_in_seq_1: (0, -1),
        take_gap_in_seq_2: (-1, 0)
    }

    move = move_dipatch_dict[cost_ranks_with_is_match]
    delta_i, delta_j = delta_dispatch_dict[move]
    return (move, delta_i, delta_j)


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
    """Modifies the lists in-place."""
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
    """Modifies the lists in-place."""
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
    """Modifies the lists in-place."""
    seq_1_aligned.append(seq_1[seq_1_index])
    middle_part.append(" ")
    seq_2_aligned.append("-")

    return (
        seq_1_aligned,
        middle_part,
        seq_2_aligned
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


def make_dp_array(
    seq_1:str, 
    seq_2:str,
    cost_mat:dict[dict],
    gap_open_cost:int|float
    ) -> list[list[list]]:
    # Create the array.
    seq_1_len = len(seq_1)
    seq_2_len = len(seq_2)
    dim_1 = seq_1_len + 1
    dim_2 = seq_2_len + 1
    dp_array = make_3d_array(
        dim_1=dim_1,
        dim_2=dim_2,
        dim_3=3,
        fill_val=0
    )
    # Initialize its values.
    # Initialize the 0-level for paths
    # that end in a match/mismatch.
    level = 0
    for i in range(1, dim_1):
        # Initialize i-th row in 0-th column.
        dp_array[i][0][level] = math.inf
    
    for j in range(1, dim_2):
        # Init j-th column in 0-th row.
        dp_array[0][j][level] = math.inf
    
    # Initialize the 1-level for paths
    # that end in a gap in seq_1.
    level = 1

    for i in range(1, dim_1):
        # Initialize i-th row in 0-th column.
        dp_array[i][0][level] = math.inf

    seq_2_index = 0
    dp_array[0][1][level] = gap_open_cost + cost_mat["-"][seq_2[seq_2_index]]
    for j in range(2, dim_2):
        # Init j-th column in 0-th row.
        seq_2_index += 1
        dp_array[0][j][level] = dp_array[0][j - 1][level] + cost_mat["-"][seq_2[seq_2_index]]

    # Initialize the 2-level for paths
    # that end in a gap in seq_2.
    level = 2
    
    seq_1_index = 0
    dp_array[1][0][level] = gap_open_cost + cost_mat[seq_1[seq_1_index]]["-"]
    for i in range(2, dim_1):
        # Initialize i-th row in 0-th column.
        seq_1_index += 1
        dp_array[i][0][level] = dp_array[i - 1][0][level] + cost_mat[seq_1[seq_1_index]]["-"]

    for j in range(1, dim_2):
        # Init j-th column in 0-th row.
        dp_array[0][j][level] = math.inf

    return dp_array

def make_3d_array(dim_1:int, dim_2:int, dim_3:int, fill_val:int|float|str) -> list[list[list]]:
    """ See: https://www.freecodecamp.org/news/list-within-a-list-in-python-initialize-a-nested-list/"""
    return [[[fill_val]*(dim_3) for i in range(dim_2)] for i in range(dim_1)]


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