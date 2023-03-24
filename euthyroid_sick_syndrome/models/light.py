import sys
sys.path.append('euthyroid_sick_syndrome')
import lightgbm as lgb
import pandas as pd
from utils import *
from sklearn.model_selection import GridSearchCV
import numpy as np

if __name__ == '__main__':
    #Carregando o dataset
    dataset = pd.read_csv("euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_features.csv")
    output_label_dataset = dataset['classification']
    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]
    #Balanceamento dos dados
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)
    #Dividindo os dados em treino (train) e teste (test)
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)

    #treinando o modelo
    model = lgb.LGBMClassifier(
        learning_rate = 1,
        max_depth = 11,
        n_estimators = 5,
        num_leaves = 30,
        feature_fraction = 0.47,
        subsample = 0.01
    )
    model.fit(input_train, output_train)

    #realizando predições
    output_model_decision = model.predict(input_test)

    print("\n\n\n\n\n")
    plot_confusion_matrix(output_test, output_model_decision, model, title = 'Matriz Confusão')

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    roc(output_test, output_model_decision) #plotando a curva ROC
 
    #plotando a curva de erro
    miss_classification(input_train, output_train, input_test, output_test, model)

    learning_curves(input_train, output_train, input_test, output_test, model)
    
    