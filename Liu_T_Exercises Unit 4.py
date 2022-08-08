import pandas as pd
import numpy as np
from urllib.request import urlretrieve


def normalize_table(_df):
    """
    analyzing_tabular_omics_data_in_pandas exercise #1.
    Normalizes a table by taking the sum of the column and dividing by # rows.
    Designed to not be used with apply function
    :param _df: DataFrame containing table to be normalized.
    :return: Normalized DataFrame.
    """

    # Mutable workaround for DataFrames, create actual copy of passed df.
    _df = pd.DataFrame(_df, copy=True)
    col_sum = _df.iloc[:, 1:].sum()  # Get the sum of each column
    _df.iloc[:, 1:] = _df.iloc[:, 1:].div(col_sum)  # Divide each column by col_sum

    return _df


def calc_shannon_div_index(_df):
    """
    analyzing_tabular_omics_data_in_pandas exercise #2.
    Given a table, calculate Shannon Diversity Index
    Assumes other functions do not exist (won't use normalize_table function).
    :param _df: DataFrame
    :return: DataFrame w/ Shannon Diversity Index
    """

    # Mutable workaround, create two actual copies.
    _df = pd.DataFrame(_df, copy=True)
    orig_df = pd.DataFrame(_df, copy=True)

    # Normalize table to get proportions.
    col_sum = _df.iloc[:, 1:].sum()
    _df.iloc[:, 1:] = _df.iloc[:, 1:].div(col_sum)

    # Could use apply, but would need to define other function so loop instead.
    for col in orig_df.columns[1:]:
        # Use numpy for log as math's log does not take in multidimensional parameters.
        # Mask to get rid of divide by zero 0 when taking natural log.
        orig_df[col] = -1 * (_df[[col]] * np.log(_df[[col]].mask(_df[[col]] <= 0)).fillna(0))
    orig_df = orig_df.iloc[:, 1:].sum()

    return orig_df


if __name__ == '__main__':
    genome_url = 'https://raw.githubusercontent.com/zaneveld/' \
                 'full_spectrum_bioinformatics/master/content/07_tabular_omics_data' \
                 '/resources/scenario1_otus_pandas.txt'
    genome_file_name = 'scenario_1_otus_pandas.tsv'
    urlretrieve(genome_url, genome_file_name)

    # Analyzing Tabular Omics Exercise #1:
    df = pd.read_csv('scenario_1_otus_pandas.tsv', sep="\t")
    print(f'Original dataframe: \n {df}')
    df_1 = normalize_table(df)
    print(f'Normalized dataframe: \n {df_1}')

    # Check original dataframe has not been changed:
    print(f'Original dataframe after normalization: \n {df}')

    # Analyzing Tabular Omics Exercise #2:
    df_2 = calc_shannon_div_index(df)
    print(f'Shannon Diversity Index: \n {df_2}')


