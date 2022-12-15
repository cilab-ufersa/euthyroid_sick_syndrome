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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Para plotar a matriz de confusão

def plot_confusion_matrix(output_test, output_model_decision, model):
    #Plotando a matriz de confusão
    """ploting confusion matrix

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
        model (list): model class
    """
    cm = confusion_matrix(output_test, output_model_decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    disp.ax_.set_title('Matriz de confusão')
    disp.ax_.set_xlabel('Classificação prevista')
    disp.ax_.set_ylabel('Classificação real')
    plt.show()

from sklearn.metrics import accuracy_score #Para calcular a acuracia do modelo

def accuracy(output_test, output_model_decision):
    """show accuracy

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("\nA acurácia é de: ", accuracy_score(output_test, output_model_decision))

from sklearn.metrics import precision_score #Para calcular a precisão do modelo

def precision(output_test, output_model_decision):
    """show precision

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A precisão é de: ", precision_score(output_test, output_model_decision))

from sklearn.metrics import recall_score #Para comparar os falsos positivos com os falsos negativos

def recall(output_test, output_model_decision):
    """show recall

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A pontuação de recall é de: ", recall_score(output_test, output_model_decision))

from sklearn.metrics import f1_score #Para calcular a média harmonica entre precisão e recall

def f1(output_test, output_model_decision):
    """show F1

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    print("A pontuação de F1 é de: ", f1_score(output_test, output_model_decision)) #Pontuação do F1

from sklearn import metrics #Para calcular a curva ROC

def roc(output_test, output_model_decision):
    """ploting ROC curve

    Args:
        output_test (list): dataset for test
        output_model_decision (list): dataset for train
    """
    fp, tp, _ = metrics.roc_curve(output_test, output_model_decision)
    plt.plot(fp, tp)
    plt.ylabel("verdadeiro positivo")
    plt.xlabel("falso positivo")
    plt.show()

from mlxtend.plotting import plot_learning_curves #Para plotar a curva de erro

def learning_curves(input_train, output_train, input_test, output_test, model):
    plot_learning_curves(X_train=input_train, y_train=output_train, X_test=input_test, y_test=output_test, clf=model)
    plt.show()