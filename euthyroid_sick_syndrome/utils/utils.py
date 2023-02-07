import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn import metrics 
from mlxtend.plotting import plot_learning_curves 

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


def plot_confusion_matrix(output_test, output_model_decision, model, title):
    """plot confusion matrix

    Args:

        output_test (list): dataset for test
        output_model_decision (list): dataset for train
        model (string): model name
        title (string): title
    Returns:
        ConfusionMatrixDisplay: confusion matrix
    """

    confusionmatrix = confusion_matrix(output_test, output_model_decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=['Normal', 'Doente'])
    fig, ax = plt.subplots(figsize=(3,3))
    disp.plot(ax=ax, colorbar=False, cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    disp.ax_.set_xlabel('Classificação Predita')
    disp.ax_.set_ylabel('Classificação Real')
    return disp


def accuracy(output_test, output_model_decision):
    """show accuracy

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("\nA acurácia é de: ", accuracy_score(output_test, output_model_decision))



def precision(output_test, output_model_decision):
    """show precision

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A precisão é de: ", precision_score(output_test, output_model_decision))



def recall(output_test, output_model_decision):
    """show recall

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A pontuação de recall é de: ", recall_score(output_test, output_model_decision))


def f1(output_test, output_model_decision):
    """show F1

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A pontuação de F1 é de: ", f1_score(output_test, output_model_decision)) #Pontuação do F1



def roc(output_test, output_model_decision, title = "Curva roc"):
    """ploting ROC curve

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    fp, tp, _ = metrics.roc_curve(output_test, output_model_decision)
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(fp, label = "Taxa de falsos positivos", linewidth = 2, linestyle = "--")
    plt.plot(tp, label = "Taxa de verdadeiros positivos", linewidth = 2, linestyle = "--")
    plt.legend()
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title('Curva roc')
    plt.grid(True)
    fig = plt.gcf()
    return fig

def miss_classification(input_train, output_train, input_test, output_test, model, title='Curva de erro'):
    """ Plot miss classification error

    Args:
        input_train (array): input train
        output_train (array): output train
        input_test (array): input test
        output_test (array): output test
        model (object): model
    Return:
    """
    training_errors, test_errors = plot_learning_curves(X_train=input_train, y_train=output_train, X_test=input_test, y_test=output_test, clf=model, print_model=False)
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(np.arange(10, 101, 10), training_errors, label='Erro de treinamento', linewidth=2, linestyle='--')
    plt.plot(np.arange(10, 101, 10), test_errors, label='Erro de teste', linewidth=2)
    plt.legend()
    plt.xlabel('Conjunto de treinamento')
    plt.ylabel('Erro de classificação')
    plt.title(title)
    plt.grid(True)
    fig = plt.gcf()
    return fig

def learning_curves(input_train, output_train, input_test, output_test, model, title = 'Curva de aprendizado'):
    """  Plot learning curves

    Args:

        input_train (array): input train
        output_train (array): output train
        input_test (array): input test
        output_test (array): output test
        model (object): model
    Return:
        fig: plot    
    """

    training_errors, test_errors = plot_learning_curves(X_train=input_train, y_train=output_train, X_test=input_test, y_test=output_test, clf=model, scoring='accuracy')
    fig, ax = plt.subplots(figsize=(3,3))
    plt.plot(np.arange(10, 101, 10), training_errors, label='Treinamento', linewidth=2, linestyle='--')
    plt.plot(np.arange(10, 101, 10), test_errors, label='Teste', linewidth=2)
    plt.legend()
    plt.xlabel('Conjunto de treinamento')
    plt.ylabel('Acurácia')
    plt.title(title)
    plt.grid(True)
    fig = plt.gcf()
    return fig

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