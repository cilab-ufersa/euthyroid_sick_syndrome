import pandas as pd

def prepare_dataset(path_origin, columns_labels, path_destiny):
    """prepare a dataset

    Args:
        path_origin (string): origin path
        columns_labels (list): columns label
        path_destiny (string): path destiny
    """
    dataframe = pd.read_csv(path_origin)
    dataframe.columns=columns_labels
    dataframe.to_csv(path_destiny, index=False)
