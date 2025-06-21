#!/usr/bin/env python3

from pathlib import Path
import random
from importlib import resources
import math
from copy import deepcopy
from dataclasses import dataclass

@dataclass
class SimpleScoringSettings:
    """Keep things simple to avoid conflicts
    
    such as when a custom scoring matrix is used.
    """
    match_score: int = 2
    mismatch_score: int = -3
    gap_open_score: int = -4
    gap_extension_score: int = -2

    # https://stackoverflow.com/questions/60179799/python-dataclass-whats-a-pythonic-way-to-validate-initialization-arguments
    def __post_init__(self):
        # Create locally defined variables
        # for use within the __post_init__ scope.
        match_score = self.match_score
        mismatch_score = self.mismatch_score
        gap_open_score = self.gap_open_score
        gap_extension_score = self.gap_extension_score
        
        if match_score is None:
            match_score_2 = 2
        else:
            match_score_2 = match_score
        
        if mismatch_score is None:
            mismatch_score_2 = -3
        else:
            mismatch_score_2 = mismatch_score

        if gap_open_score is None:
            gap_open_score_2 = -4
        else:
            gap_open_score_2 = gap_open_score

        if gap_extension_score is None:
            gap_extension_score_2 = -2
        else:
            gap_extension_score_2 = gap_extension_score

        try:
            match_score_3 = int(match_score_2)
        except (TypeError, ValueError) as e:
            print("match_score must be convertible to an integer.")
            raise e
        
        try:
            mismatch_score_3 = int(mismatch_score_2)
        except (TypeError, ValueError) as e:
            print("mismatch_score must be convertible to an integer.")
            raise e
        
        try:
            gap_open_score_3 = int(gap_open_score_2)
        except (TypeError, ValueError) as e:
            print("gap_open_score must be convertible to an integer.")
            raise e

        try:
            gap_extension_score_3 = int(gap_extension_score_2)
        except (TypeError, ValueError) as e:
            print("gap_extension_score must be convertible to an integer.")
            raise e
        
        if match_score_3 <= 0:
            raise ValueError
        
        if mismatch_score_3 >= 0:
            raise ValueError
        
        if gap_open_score_3 > 0:
            raise ValueError
        
        if gap_extension_score_3 >= 0:
            raise ValueError
        
        self.match_score = match_score_3
        self.mismatch_score = mismatch_score_3
        self.gap_open_score = gap_open_score_3
        self.gap_extension_score = gap_extension_score_3

        return None
    
@dataclass
class SimpleCostingSettings:
    mismatch_cost: int = 5
    gap_open_cost: int = 4
    gap_extension_cost: int = 3

    # https://stackoverflow.com/questions/60179799/python-dataclass-whats-a-pythonic-way-to-validate-initialization-arguments
    def __post_init__(self):
        mismatch_cost = self.mismatch_cost
        gap_open_cost = self.gap_open_cost
        gap_extension_cost = self.gap_extension_cost

        if mismatch_cost is None:
            mismatch_cost_2 = 5
        else:
            mismatch_cost_2 = mismatch_cost

        if gap_open_cost is None:
            gap_open_cost_2 = 4
        else:
            gap_open_cost_2 = gap_open_cost

        if gap_extension_cost is None:
            gap_extension_cost_2 = 3
        else:
            gap_extension_cost_2 = gap_extension_cost

        try:
            self.mismatch_cost = int(mismatch_cost_2)
        except (TypeError, ValueError) as e:
            print("mismatch_cost must be convertible to an integer.")
            raise e
        
        try:
            self.gap_open_cost = int(gap_open_cost_2)
        except (TypeError, ValueError) as e:
            print("gap_open_cost must be convertible to an integer.")
            raise e
        
        try:
            self.gap_extension_cost = int(gap_extension_cost_2)
        except (TypeError, ValueError) as e:
            print("gap_extension_cost must be convertible to an integer.")
            raise e
        
        if mismatch_cost_2 <= 0:
            raise ValueError
        
        if gap_open_cost_2 < 0:
            raise ValueError
        
        if gap_extension_cost_2 <= 0:
            raise ValueError

        return None


def validate_and_transform_args(
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
) -> tuple:
    """Validates the command line arguments

    or the arguments that are passed when
    the module is imported and its functionality
    used that way.

    Returns:
        tuple with entries of
            seq_1_validated,
            seq_2_validated,
            scoring_mat_validated,
            costing_mat_validated,
            gap_open_score_validated,
            gap_open_cost_validated,
            output_validated
    """
    ##################################################################
    # Validate and transform output argument.
    ##################################################################
    if output is not None:
        output_b = Path(output)
        if output_b.is_file():
            output_validated = output_b
            raise RuntimeWarning(f"Overwriting {output_b}")
        elif not output_b.is_file() and output_b.parent.exists():
            output_validated = output_b
        else:
            raise FileNotFoundError("The parent directory of output does not exist.")
    else:
        output_validated = None
    ##################################################################
    # Validate and transform: 
    # input_fasta
    # seq_1
    # seq_2
    ##################################################################
    if input_fasta is not None and seq_1 is None and seq_2 is None:
        input_fasta_b = Path(input_fasta)
        try:
            seq_1, seq_2 = read_first_2_seqs_from_fasta(input_fasta_b)
        except FileNotFoundError:
            print("input_fasta does not point to a valid file.  Please make sure it is in the correct FASTA format.  Note that reading from standard input is not supported at this time.")
            raise 
    elif (input_fasta is None and seq_2 is None) or (input_fasta is not None and seq_1 is not None) or (seq_1 is None and seq_2 is not None):
        raise RuntimeError("The combination of arguments for input_fasta, seq_1, and seq_2 does not make sense.")
    
    # Check that the product of the lengths of the sequences is
    # positive and less than 20_000_000.     
    check_seq_lengths(seq_1, seq_2, 20_000_000)
    seq_1_validated = seq_1.upper()
    seq_2_validated = seq_2.upper()
    del seq_1
    del seq_2
    ##################################################################
    # Validate and transform scoring and costing settings.
    ##################################################################
    # Check that there are no unwanted combinations.
    if scoring_mat_name is not None and any([x is not None for x in (scoring_mat_path, match_score, mismatch_score, mismatch_cost, gap_extension_score, gap_extension_cost)]):
        raise RuntimeError("The scoring_mat_name should not be specified if any of the other options with scores or costs are specified, except for the gap_open options.")
    elif scoring_mat_path is not None and any([x is not None for x in (scoring_mat_name, match_score, mismatch_score, mismatch_cost, gap_extension_score, gap_extension_cost)]):
        raise RuntimeError("The scoring_mat_path should not be specified if any of the other options with scores or costs are specified, except for the gap_open options.")
    elif any([x is not None for x in (match_score, mismatch_score, gap_open_score, gap_extension_score)]) and any([x is not None for x in [mismatch_cost, gap_open_cost, gap_extension_cost]]):
        raise RuntimeError("Scoring and costing options should not both be set.")
    
    # Now, that there are no unwanted combinations, we can set
    # things to their values or to suitable defaults.
    simple_scoring_settings = SimpleScoringSettings(
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_open_score=gap_open_score,
        gap_extension_score=gap_extension_score
    )
    
    simple_costing_settings = SimpleCostingSettings(
        mismatch_cost=mismatch_cost,
        gap_open_cost=gap_open_cost,
        gap_extension_cost=gap_extension_cost
    )

    # gap_open_score and gap_open_cost should always
    # be opposites of each other.  
    if gap_open_score is not None:
        # gap_open_score was specified but
        # gap_open_cost was not.  Therefore,
        # gap_open_cost was set to a default
        # that we do not care about.
        simple_costing_settings.gap_open_cost = - simple_scoring_settings.gap_open_score
    else:
        # gap_open_score was set to a default value.
        # gap_open_cost was either specified
        # or set to a default value.  Either
        # way, we want to use gap_open_cost.
        simple_scoring_settings.gap_open_score = - simple_costing_settings.gap_open_cost


    if scoring_mat_name is not None:
        # Do fancy stuff with the importlib
        # library so that files are accessible
        # on other people's machines.
        data_traversable = resources.files("globalign.data")
        blosum_file_name = "".join([scoring_mat_name, ".mtx"])
        blo = data_traversable.joinpath("scoring_matrices", blosum_file_name)
        with resources.as_file(blo) as f:
            scoring_mat_b = read_scoring_mat(f)

        # Check that the sequences
        # only contain letters present in the scoring matrix.
        common_alphabet = get_common_alphabet(seq_1_validated, seq_2_validated)
        validate_scoring_mat_keys(
            scoring_mat_keys=scoring_mat_b.keys(),
            common_alphabet=common_alphabet
        )
        scoring_mat_validated = scoring_mat_b
    
        # https://curiouscoding.nl/posts/alignment-scores-transform/
        max_score = get_max_val(scoring_mat_validated)
        
        costing_mat_validated = scoring_mat_to_costing_mat(
            scoring_mat=scoring_mat_validated,
            max_score=max_score
        )
    elif scoring_mat_path is not None:
        scoring_mat_path_2 = Path(scoring_mat_path)
        scoring_mat = read_scoring_mat(scoring_mat_path_2)
        
        # Check that the scoring matrix is symmetric.
        if not check_symmetric(mat=scoring_mat):
            raise RuntimeError("The scoring matrix is not symmetric.")
        
        # For each row, the entry on the main diagonal
        # should be greater than or equal to the other entries in the row.
        if not check_big_main_diag(mat=scoring_mat):
            raise RuntimeError("The scoring matrix does not make sense because the maximum for each row does not occur on the main diagonal.")
        # Check that the sequences
        # only contain letters present in the scoring matrix.
        common_alphabet = get_common_alphabet(seq_1_validated, seq_2_validated)
        validate_scoring_mat_keys(
            scoring_mat_keys=scoring_mat.keys(),
            common_alphabet=common_alphabet
        )
        scoring_mat_validated = scoring_mat
    
        # https://curiouscoding.nl/posts/alignment-scores-transform/
        max_score = get_max_val(scoring_mat_validated)
        
        costing_mat_validated = scoring_mat_to_costing_mat(
            scoring_mat=scoring_mat_validated,
            max_score=max_score
        )
    elif any([x is not None for x in [mismatch_cost, gap_open_cost, gap_extension_cost]]):
        common_alphabet = get_common_alphabet(seq_1_validated, seq_2_validated)
        costing_mat_validated = create_costing_mat(
            common_alphabet=common_alphabet,
            mismatch_cost=simple_costing_settings.mismatch_cost,
            gap_open_cost=simple_costing_settings.gap_open_cost,
            gap_extension_cost=simple_costing_settings.gap_extension_cost
        )
        scoring_mat_validated = costing_mat_to_scoring_mat(
            costing_mat=costing_mat_validated,
            max_score=simple_scoring_settings.match_score
        )
    else:
        common_alphabet = get_common_alphabet(seq_1_validated, seq_2_validated)

        scoring_mat_validated = create_scoring_mat(
            common_alphabet=common_alphabet,
            match_score=simple_scoring_settings.match_score,
            mismatch_score=simple_scoring_settings.mismatch_score,
            gap_extension_score=simple_scoring_settings.gap_extension_score
        )
        
        costing_mat_validated = scoring_mat_to_costing_mat(
            scoring_mat=scoring_mat_validated,
            max_score=simple_scoring_settings.match_score
        )

    return (
        seq_1_validated,
        seq_2_validated,
        scoring_mat_validated,
        costing_mat_validated,
        simple_scoring_settings.gap_open_score,
        simple_costing_settings.gap_open_cost,
        output_validated
    )

def get_common_alphabet(seq_1, seq_2):
    common_alphabet = list(set(seq_1).union(set(seq_2)))
    common_alphabet.sort()
    return common_alphabet


def check_seq_lengths(seq_1, seq_2, max_seq_len_prod):
    """Check that the product of the lengths of the sequences is
    
    reasonable, i.e. positive and less than max_seq_len_prod.

    Raises:
        RuntimeError: For a variety of a reasons.
    """
    m = len(seq_1)
    n = len(seq_2)
    seq_len_prod = m*n
    if not seq_len_prod < max_seq_len_prod:
        raise RuntimeError(f"Your sequences are too long.  The product of their lengths should be less than {max_seq_len_prod}.  They have lengths of {m} and {n}")
    elif seq_len_prod == 0:
        raise RuntimeError(f"Detected a sequence of length 0.")
    return None

def read_scoring_mat(scoring_mat_path:Path) -> dict[dict]:
    """Read in scoring matrix.
    
    Raises:
        FileNotFoundError: if not scoring_mat_path.is_file().
        RunTimeError: if the header row did not have single letters spaced apart.
        RunTimeError: if row headers do not match column headers.
    """
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


def create_scoring_mat(
    common_alphabet: list, 
    match_score: int, 
    mismatch_score: int, 
    gap_extension_score: int
) -> dict[dict]:
    common_alphabet.append("-")
    scoring_mat = dict()
    for common_key_outer in common_alphabet:
        scoring_mat[common_key_outer] = dict()
        for common_key_inner in common_alphabet:
            if common_key_outer == common_key_inner:
                scoring_mat[common_key_outer][common_key_inner] = match_score
            elif common_key_outer == "-" or common_key_inner == "-":
                scoring_mat[common_key_outer][common_key_inner] = gap_extension_score
            else:
                scoring_mat[common_key_outer][common_key_inner] = mismatch_score

    return scoring_mat

def create_costing_mat(
    common_alphabet: list, 
    mismatch_cost: int, 
    gap_open_cost: int,
    gap_extension_cost: int
) -> dict[dict]:
    common_alphabet.append("-")
    costing_mat = dict()
    for common_key_outer in common_alphabet:
        costing_mat[common_key_outer] = dict()
        for common_key_inner in common_alphabet:
            if common_key_outer == common_key_inner:
                costing_mat[common_key_outer][common_key_inner] = 0
            elif common_key_outer == "-" or common_key_inner == "-":
                costing_mat[common_key_outer][common_key_inner] = gap_extension_cost
            else:
                costing_mat[common_key_outer][common_key_inner] = mismatch_cost

    return costing_mat


def validate_scoring_mat_keys(
    scoring_mat_keys: set,
    common_alphabet: list
) -> None:
    """Check that the scoring_mat_keys include
    
    a gap '-' and any other necessary letters
    (i.e. letters in common_alphabet).
    """
    common_alphabet.append("-")
    diff = set(common_alphabet).difference(scoring_mat_keys)
    if len(diff) == 0:
        return None
    else:
        raise RuntimeError(f"common_alphabet contains values not in scoring_mat_keys, e.g. {diff}.  Please check your sequences and your scoring matrix.")


def get_max_val(m:dict[dict]) -> int|float:
    """Get the max value inside a nested dictionary.
    """
    # prep for loop
    cur_max = - math.inf
    for key, inner_dict in m.items():
        new_possible_max = max(inner_dict.values())
        cur_max = max(cur_max, new_possible_max)

    return cur_max


def scoring_mat_to_costing_mat(
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
    costing_mat = deepcopy(scoring_mat)
    b = max_score
    if delta_d is None:
        delta_d = math.floor(b/2)
    if delta_i is None:
        delta_i = math.ceil(b/2)
    
    for seq_1_letter, seq_2_scores in costing_mat.items():
        # seq_1_letter is a key for costing_mat.
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
   
    return costing_mat

def costing_mat_to_scoring_mat(
    costing_mat:dict[dict], 
    max_score:int|float,
    delta_d:int|float=None,
    delta_i:int|float=None
) -> dict[dict]:
    """Get a scoring matrix from a costing matrix.

    Args: 
        costing_mat: Nested dict representation of 
            a distance matrix
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
        Nested dict representation of a scoring matrix

    Reference: https://curiouscoding.nl/posts/alignment-scores-transform/
    """
    # Make sure we don't mutate the costing_mat
    scoring_mat = deepcopy(costing_mat)
    b = max_score
    if delta_d is None:
        delta_d = math.floor(b/2)
    if delta_i is None:
        delta_i = math.ceil(b/2)
    
    for seq_1_letter, seq_2_costs in scoring_mat.items():
        # seq_1_letter is a key for costing_mat.
        # seq_2_costs is the inner dict for the outer
        # key of seq_1_letter.
        for seq_2_letter, cost in seq_2_costs.items():
            # The costs are transformed differently
            # for insertions and deletions, than they
            # are for matches and mismatches.
            if seq_1_letter == "-" and seq_2_letter != "-":
                # Update deletions (horizontal steps)
                seq_2_costs[seq_2_letter] = delta_d - cost
            elif seq_2_letter == "-" and seq_1_letter != "-":
                # Update insertions (vertical steps)
                seq_2_costs[seq_2_letter] = delta_i - cost
            else:
                # Update matches and mismatches.
                seq_2_costs[seq_2_letter] = delta_d + delta_i - cost
   
    return scoring_mat

def read_seq_from_fasta(fasta_path:Path):
    """Read in a FASTA file. 

    Raises:
        RuntimeError: If invalid FASTA format is detected.
    Yields:
        2-tuples where the 0th element is the description
        and the 1st element is the sequence
    
    See: NCBI FASTA specification
    https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
    """
    with fasta_path.open() as f:
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


def read_first_2_seqs_from_fasta(fasta_path: Path) -> tuple[str]:
    """
    Returns:
        (seq_1, seq_2)
    Raises:
        RuntimeError
    """
    counter = 0
    seq_1 = None
    seq_2 = None
    for desc_and_seq in read_seq_from_fasta(fasta_path=fasta_path):
        counter += 1
        if counter == 1:
            desc_1, seq_1 = desc_and_seq
        elif counter == 2:
            desc_2, seq_2 = desc_and_seq
        else:
            break

    if seq_1 is not None and seq_2 is not None:    
        return (seq_1, seq_2)
    else:
        raise RuntimeError("Two sequences could not be read from the FASTA file.")
    
    
def draw_random_seq(
    alphabet:list[str], 
    min_len:int, 
    max_len:int,
    seed:int=None
) -> str:
    """
    Raises:
        IndexError: If alphabet == [].
        TypeError: If alphabet is not a list with a len() method.
        ValueError: If min_len > max_len or if min_len < 0.
    """
    random.seed(seed)
    # Randomly decide on how long the sequence should be.
    if min_len < 0:
        print("min_len must be a non-negative integer.")
        raise ValueError
    
    try:
        seq_len = random.randint(a=min_len, b=max_len)
    except ValueError:
        print("min_len and max_len must be non-negative integers with max_len >= min_len.")
        raise
    # Draw the desired number of letters from the alphabet.
    try:
        random_seq_list = random.choices(population=alphabet, k=seq_len)
    except (IndexError, TypeError):
        print("alphabet must be a non-empty list of strings")
        raise
    # Return the sequence as a string.
    return "".join(random_seq_list)


def draw_two_random_seqs(
    alphabet:list, 
    min_len_seq_1:int, 
    max_len_seq_1:int,
    min_len_seq_2:int, 
    max_len_seq_2:int,
    divergence:float,
    seed_1:int=None,
    seed_2:int=None
) -> tuple[str]:
    """
    Args:
        divergence: a number between 0 and 1, inclusive.
            Higher values for divergence will tend
            to make the two sequences more different
            from each other.
    """
    seq_1 = draw_random_seq(
        alphabet=alphabet,
        min_len=min_len_seq_1,
        max_len=max_len_seq_1,
        seed=seed_1
    )

    len_seq_1 = len(seq_1)

    # seq_2 will just be a copy of seq_1 at first.
    seq_2_list = list(seq_1)
    
    # len_seq_2 is the length after all of the edits.
    # Change the global random state for making seq_2.
    random.seed(seed_2)
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
            max_len=num_insertions,
            seed=seed_2
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
    return (seq_1, seq_2)

def make_matrix(num_rows:int, num_cols:int, fill_val:int|float|str|None) -> list[list[int|float|str|None]]:
    """Make a matrix as a nested list.
    
    See: https://www.freecodecamp.org/news/list-within-a-list-in-python-initialize-a-nested-list/
    """
    return [
        [fill_val]*(num_cols) for i in range(num_rows)
    ]

def make_3d_array(dim_1:int, dim_2:int, dim_3:int, fill_val:int|float|str) -> list[list[list]]:
    """ See: https://www.freecodecamp.org/news/list-within-a-list-in-python-initialize-a-nested-list/"""
    return [[[fill_val]*(dim_3) for i in range(dim_2)] for i in range(dim_1)]


def check_symmetric(mat:dict[dict]) -> bool:
    """Check if a matrix is symmetric.
    
    Args:
        mat: nested dictionary representing a matrix
    
    Returns:
        True if mat is symmetric and False otherwise.

    Raises:
        AttributeError: if mat is not a nested dictionary.
    """
    # https://realpython.com/iterate-through-dictionary-python/#traversing-a-dictionary-directly
    # https://softwareengineering.stackexchange.com/questions/187715/validation-of-the-input-parameter-in-caller-code-duplication
    try:
        for outer_key in mat.keys():
            # Assume that the outer and inner
            # keys are the same.
            for inner_key in mat.keys():
                try:
                    has_eq_vals = (mat[outer_key][inner_key] == mat[inner_key][outer_key])
                except KeyError:
                    return False
                if not has_eq_vals:
                    return False
        
        return True
    except AttributeError:
        print("The check_symmetric function expected a nested dictionary.")
        raise 


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