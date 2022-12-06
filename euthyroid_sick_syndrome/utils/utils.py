import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def balance_dataset_smote(dataset, output_label, random_state=42, k_neighbors=5):
    """balance dataset

    Args:
        dataset (pandas dataframe): dataset
        output_label (string): output label
        random_state (int, optional): random state. Defaults to 42.
        k_neighbors (int, optional): k neighbors. Defaults to 5.

    Returns:
        pandas dataframe: dataset balanced
    """
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    dataset_res, ouput_label = sm.fit_resample(dataset, output_label)

    return dataset_res, ouput_label

def slipt_and_standardize_dataset(dataset, output_label, test_size=0.2, random_state=23):
    """slipt and standardize dataset

    Args:
        dataset (pandas dataframe): dataset
        output_label (string): output label
        test_size (float, optional): test size. Defaults to 0.2.
        random_state (int, optional): random state. Defaults to 23.

    Returns:
        pandas dataframe: input train
        pandas dataframe: input test
        pandas dataframe: output train
        pandas dataframe: output test
    """
    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = train_test_split(dataset, output_label, test_size=test_size, random_state=random_state)

    # Padronizando os dados
    scaler = StandardScaler()
    input_train = scaler.fit_transform(input_train)
    input_test = scaler.transform(input_test)

    return input_train, input_test, output_train, output_test