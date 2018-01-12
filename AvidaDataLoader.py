import pandas as pd

def load_generic(path, trailing_space=False, columns=None):
    """
    Load a generic Avida-style data file.

    :param path:  the path to the data file
    :param trailing_space:  is there a trailing space at the end of each line?
    :param columns: the names of the columns

    :return: a Pandas dataframe
    """
    data = pd.read_csv(path,
                           comment='#', skip_blank_lines=True,
                           delimiter=' ', header=None)
    if trailing_space:
        data.drop(data.columns[[-1]], axis=1, inplace=True)

    if columns:
        data.columns = columns

    return data.infer_objects()


def load_spatial_reactions(path):
    """
    Load a spatial reaction data file.

    :param path: the path of the data file

    :return: a pandas dataframe
    """
    data = load_generic(path)
    nrows,ncols = data.shape
    colnames = ['update','reaction']
    cellcols = map(lambda x: f'cell_{x}', range(0,ncols-2))
    colnames.extend(cellcols)
    data.columns = colnames
    return data


def load_spatial_resources(path):
    """
    Load a spatial resource data file.

    :param path:  the path of the data file

    :return: a pandas dataframe
    """
    data = load_generic(path)
    nrows,ncols = data.shape
    colnames = ['update','resource']
    cellcols = map(lambda x: f'cell_{x}', range(0,ncols-2))
    colnames.extend(cellcols)
    data.columns = colnames
    return data
