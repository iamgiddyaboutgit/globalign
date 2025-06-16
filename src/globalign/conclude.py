#!/usr/bin/env python3

import math
from pathlib import Path
from typing import NamedTuple

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

    def _generate_alignment_printout(
        self,
        desc_1: str="seq_1", 
        desc_2: str="seq_2", 
        chars_per_line: int=70
    ):
        seq_1_aligned = self.seq_1_aligned
        middle_part = self.middle_part
        seq_2_aligned = self.seq_2_aligned
        cost = self.cost
        score = self.score
        scoring_mat = self.scoring_mat
        costing_mat = self.costing_mat
        gap_open_score = self.gap_open_score
        gap_open_cost = self.gap_open_cost

        # Handle long alignments with proper line breaking.
        alignment_len = len(middle_part)
        num_sets_needed = math.ceil(alignment_len / chars_per_line)
        
        # Prep for loop
        lower = 0
        if num_sets_needed == 1:
            upper = alignment_len
        else:   
            upper = chars_per_line

        yield desc_1
        yield "\n"
        yield desc_2
        
        for u in range(num_sets_needed):
            yield "\n\n"
            # Loop body
            yield seq_1_aligned[lower:upper]
            yield "\n"
            yield middle_part[lower:upper]
            yield "\n"
            yield seq_2_aligned[lower:upper]
            # Prep for next iteration
            lower = upper
            upper = lower + chars_per_line
        
        yield "\n\n"

        yield f"score: {score}\n"
        yield f"cost: {cost}\n"
        yield f"###########################################\n# Settings\n###########################################\n"
        yield f"scoring_mat:\n"
        yield prettify_mat(scoring_mat)
        yield f"\n\ngap_open_score: {gap_open_score}\n"

        yield "\ncosting_mat:\n"
        yield prettify_mat(costing_mat)
        yield f"\n\ngap_open_cost: {gap_open_cost}\n"
        
        return None

    def __str__(
            self, 
            desc_1: str="seq_1", 
            desc_2: str="seq_2", 
            chars_per_line: int=70
        ):
        return "".join(
            self._generate_alignment_printout(
                desc_1=desc_1,
                desc_2=desc_2,
                chars_per_line=chars_per_line
            )
        )
    
    def print(
        self, 
        desc_1: str="seq_1", 
        desc_2: str="seq_2", 
        chars_per_line: int=70
    ) -> None:
        __str__ = self.__str__
        print(
            __str__(
                desc_1=desc_1,
                desc_2=desc_2,
                chars_per_line=chars_per_line
            )
        )
        return None

    def write(
        self, 
        file: Path|str=None,
        desc_1: str="seq_1", 
        desc_2: str="seq_2", 
        chars_per_line: int=70
    ):
        """Write the alignment results
        
        to a file or to sys.stdout.

        Args:
            file: The path to a file to which the results will be written.
                If file is specified, then it overwrites
                self.output; otherwise, self.output
                will be used.
                Use the string "stdout", to write to sys.stdout. 
                If self.output
                is None, then the results will be written to sys.stdout.
        """
        output = self.output
        _print = self.print
        __str__ = self.__str__

        if (file is None and output is None) or file == "stdout":
            _print(
                desc_1=desc_1,
                desc_2=desc_2,
                chars_per_line=chars_per_line
            )
            return None
        elif file is None and output is not None:
            file_2 = output
        else:
            file_2 = file 

        s = __str__(
            desc_1=desc_1,
            desc_2=desc_2,
            chars_per_line=chars_per_line
        )
        with open(file=file_2, mode="w+") as fh:
            fh.write(s)
        
        return None


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

def print_nested_list_aligned(nested_list: list[list[int|float|str]]) -> None:
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
    nested_list_representation = []
    for row in nested_list:
        row_formatted = []
        for j, cell in enumerate(row):
            # The format specification of :>{width}
            # right-aligns the string within the width.
            row_formatted.append(f"{str(cell):>{widths[j] + 1}}")
        row_formatted.append("\n")
        nested_list_representation.extend(row_formatted)
    
    print("".join(nested_list_representation))
    
    return None


def prettify_mat(mat: dict[dict]) -> str:
    """Makes a nested dictionary 
    
    representation of a matrix look better
    prior to printing.
    
    Args:
        mat: 
    """
    # Determine the column width in the pretty-printed result.
    # Consider how wide each cell is
    # as well as the column headers.
    widths = []
    # Because we assume that each inner dict in mat
    # is the same length, we can get the number of columns
    # from the number of entries in the 0-th "row"
    # of mat.
    try:
        col_headers = list(list(mat.values())[0].keys())
    except:
        print("mat does not appear to represent a matrix as a nested dictionary.")
        raise

    
    # Loop through the "columns" of the nested dict.
    for col_header in col_headers:
        # If each cell in the j-th "column" of the nested dict,
        # is given a string representation, what is the 
        # length of the longest representation?
        width_required = len(str(col_header))
        for row_header in mat.keys():
            # Is the length of the string representation of the 
            # current cell longer than the longest such 
            # representation found so far for the 
            # current col_header?
            # If it is, then save it and check the next cell
            # down with the same col_header.
            cell = mat[row_header][col_header]
            width_required = max(width_required, len(str(cell)))
        widths.append(width_required)

    # Print the numbers with formatting
    representation = []
    # Start with some extra spacing to get the columns
    # to line up.
    longest_row_header_len = max([len(str(x)) for x in col_headers])
    initial_space = "".join([" "]*(longest_row_header_len + 1))
    representation.append(initial_space)
    # Add the column headers.
    representation.extend([f"{str(col_header):>{w + 1}}" for col_header, w in zip(col_headers, widths)])
    # Loop to get the rest of representation.
    for row_header in mat.keys():
        representation.append("\n")
        representation.append(f"{str(row_header):<{longest_row_header_len + 1}}")
        for col_header, w in zip(col_headers, widths):
            cell = mat[row_header][col_header]
            representation.append(f"{str(cell):>{w + 1}}")
    
    return "".join(representation)

