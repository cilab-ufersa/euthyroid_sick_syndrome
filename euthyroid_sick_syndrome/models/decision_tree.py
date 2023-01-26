import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd #Para trabalhar com dataframes               
import numpy as np #Para trabalhar com arrays
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.tree import DecisionTreeClassifier #Para criar o modelo de árvore de decisão
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
import seaborn as sns 
from imblearn.over_sampling import SMOTE #Para balancear o dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Para plotar a matriz de confusão
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from utils.utils import * 

if __name__ == '__main__':
    #Carregando o dataset
    dataset = dataset = pd.read_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_feature.csv')

    output_label_dataset = dataset['classification']  #1 = sick, 0 = normal

    # Selecionando as colunas que serão utilizadas no modelo
    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]

    #Balanceamento dos dados 
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                                


    #Criando o modelo de árvore de decisão
    model= DecisionTreeClassifier(criterion='entropy', random_state=30, class_weight='balanced', max_depth=5)
    
    model.fit(input_train, output_train) #Treinamento

    # Fazer a classificação 
    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão

    cm = confusion_matrix(output_test, output_model_decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    disp.ax_.set_title('Matriz de confusão')
    disp.ax_.set_xlabel('Classificação prevista')
    disp.ax_.set_ylabel('Classificação real')
    plt.show()
       
    # região de decisão
    """
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(input_test)
    model.fit(X_train, output_test)
    plot_decision_regions(X_train, output_test.values, clf=model)
    plt.show()
    """

    #Plotando a curva de aprendizado
    plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    scoring='misclassification error')
    plt.show()

    plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    scoring='accuracy')
    plt.show()


    