import sys
sys.path.append('euthyroid_sick_syndrome\euthyroid_sick_syndrome')
import pandas as pd #Para trabalhar com dataframes               
import numpy as np #Para trabalhar com arrays
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.naive_bayes import GaussianNB, CategoricalNB #Para criar o modelo de decisão
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
import seaborn as sns 
from imblearn.over_sampling import SMOTE #Para balancear o dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Para plotar a matriz de confusão
import pickle #para organizar e salvar variaveis em um conjunto
from utils import * 

if __name__ == "__main__":
    #Carregando o dataset
    dataset = dataset = pd.read_csv('euthyroid_sick_syndrome\euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_dataset.csv')
    #Verificando o dataset
    print(dataset.head())
    print(dataset.describe())

    output_label_dataset = dataset['classification']
    print(output_label_dataset.value_counts()) #1 = sick, 0 = normal

    #Balanceamento dos dados 
    sm = SMOTE(random_state=42, k_neighbors=5)
    dataset_res, ouput_label = sm.fit_resample(dataset, output_label_dataset)

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = train_test_split(dataset_res, ouput_label, test_size=0.2, random_state=23)

    with open('NaiveBayes.pkl', 'wb') as f:
        pickle.dump([input_train, output_train], f)

    #Criando e implementando o modelo de decisão
    model = GaussianNB(var_smoothing=1e-5)
    model.fit(input_train, output_train) #Treinamento

    #model.classes_ #classes do banco de dados 
    #model.class_count_ #quantidade de atributos por classe
    #model.class_prior_ #probabilidade a piori  

    # Fazer a classificação 
    output_model_decision = model.predict(input_test)

    

    #Plotando a matriz de confusão
    plot_confusion_matrix(output_test, output_model_decision, model)

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)

    
    roc(output_test, output_model_decision) #plotando a curva ROC

    #plotando a curva de erro
    miss_classification(input_train, output_train, input_test, output_test, model)

    learning_curves(input_train, output_train, input_test, output_test, model)