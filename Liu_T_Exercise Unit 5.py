from IPython.core.display import HTML, display
import pandas as pd
from numpy import full


def pretty_table_from_array(data_array, row_labels, col_labels):
    """Show an HTML table from a 2d numpy array"""
    df = pd.DataFrame(data_array, index=row_labels, columns=col_labels)
    table_html = df.to_html()
    return HTML(table_html)


def score_match(nt1, nt2, scoring_matrix,
                scoring_matrix_indices={'A': 0, 'G': 1, 'C': 2, 'T': 3}):
    """Return the score for a substitution between nt1 and nt2 based on the scoring matrix
    nt1 -- a string representing the first nucleotide
    nt2 -- a string representing the second nucleotide
    scoring_matrix -- an N x N numpy array, where N is
      the number of nucleotides (so usually 4x4)
    scoring_matrix_indices -- a dict mapping rows and columns
      of the scoring array to nucleotides

    """
    return scoring_matrix[scoring_matrix_indices[nt1], scoring_matrix_indices[nt2]]


def needleman_wunsch(seq1, seq2, scoring_matrix,
                     scoring_matrix_indices={"A": 0, "G": 0, "G": 0, "C": 0},
                     scoring_function=score_match, gap_penalty=-1):
    """Perform Needleman Wunsch global alignment on two sequences
    seq1 -- a sequence as a string
    seq2 -- a sequence as a string
    gap_function -- a function that takes no parameters and returns the score for a gap
    scoring_function -- a function that takes two nucleotides and returns a score

    """
    # build an array of zeroes
    n_rows = len(seq1) + 1  # need an extra row up top
    n_columns = len(seq2) + 1  # need an extra column on the left
    scoring_array = full([n_rows, n_columns], 0)
    traceback_array = full([n_rows, n_columns], "-")

    # Define Unicode arrows we'll use in the traceback array
    up_arrow = "\u2191"
    right_arrow = "\u2192"
    down_arrow = "\u2193"
    left_arrow = "\u2190"
    down_right_arrow = "\u2198"
    up_left_arrow = "\u2196"

    arrow = "-"

    # iterate over columns first because we want to do
    # all the columns for row 1 before row 2
    for row in range(n_rows):
        for col in range(n_columns):
            if row == 0 and col == 0:
                # We're in the upper right corner
                score = 0
            elif row == 0:
                # We're on the first row
                # but NOT in the corner

                # Look up the score of the previous cell (to the left) in the score array\
                previous_score = scoring_array[row, col - 1]
                # add the gap penalty to it's score
                score = previous_score + gap_penalty
            elif col == 0:
                # We're on the first column but not in the first row
                previous_score = scoring_array[row - 1, col]
                score = previous_score + gap_penalty
            else:
                # We're in a 'middle' cell of the alignment

                # Calculate the scores for coming from above,
                # from the left, (representing an insertion into seq1)
                cell_to_the_left = scoring_array[row, col - 1]
                from_left_score = cell_to_the_left + gap_penalty

                # or from above (representing an insertion into seq2)
                above_cell = scoring_array[row - 1, col]
                from_above_score = above_cell + gap_penalty

                # diagonal cell, representing a substitution (e.g. A --> T)

                diagonal_left_cell = scoring_array[row - 1, col - 1]

                # Since the table has an extra row and column (the blank ones),
                # when indexing back to the sequence we want row -1 and col - 1.
                # since row 1 represents character 0 of the sequence.
                curr_nt_seq1 = seq1[row - 1]
                curr_nt_seq2 = seq2[col - 1]

                # the scoring matrix will tell us the score for matches,
                # transitions and transversions
                diagonal_left_cell_score = diagonal_left_cell + \
                                           score_match(curr_nt_seq1, curr_nt_seq2, scoring_matrix)
                score = max([from_left_score, from_above_score, diagonal_left_cell_score])
                # take the max
                # make note of which cell was the max in the traceback array
                # using Unicode arrows
                if score == from_left_score:
                    arrow = left_arrow
                elif score == from_above_score:
                    arrow = up_arrow
                elif score == diagonal_left_cell_score:
                    arrow = up_left_arrow

            traceback_array[row, col] = arrow
            scoring_array[row, col] = score
    return scoring_array, traceback_array


if __name__ == '__main__':
    # Build a dict to assign each nucleotide one row or column
    # index in the table
    nucleotides = "AGCT"

    # Step through each nucleotide and give it a row and column index
    # using a dictionary with keys = nucleotides and values = indices
    nucleotide_indices = {nucleotide: i for i, nucleotide in enumerate(nucleotides)}

    # Set up scores
    match_score = 1
    # We want separate scores for substitutions that are
    # transitions or transversions
    transversion_score = -2
    transition_score = -1

    seq1 = "GCATGCT"
    seq2 = "GATACCA"
    row_labels = [label for label in "-" + seq1]
    column_labels = [label for label in "-" + seq2]
    scoring_matrix = full([len(nucleotides), len(nucleotides)], transition_score)

    scoring_array, traceback_array = needleman_wunsch(seq1, seq2, scoring_matrix)
    display(pretty_table_from_array(scoring_array, row_labels, column_labels))
    display(pretty_table_from_array(traceback_array, row_labels, column_labels))
