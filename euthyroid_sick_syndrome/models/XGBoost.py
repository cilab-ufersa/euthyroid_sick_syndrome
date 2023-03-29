import sys
sys.path.append('euthyroid_sick_syndrome')

import pandas as pd #Para trabalhar com dataframes               
from xgboost import XGBClassifier #Para criar o modelo de árvore de decisão

#Biblioteca para visualização
import matplotlib.pyplot as plt 
import seaborn as sns

from utils import *
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold
import joblib
import pickle
file_name = "XGBoost.joblib"

if __name__ == '__main__':

    #Carregando o dataset
    dataset = pd.read_csv('euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')  
    output_label_dataset = dataset['classification']  #1 = sick, 0 = normal
    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]

    #Balanceamento dos dados 
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)

    # Definindo o espaço de busca 
    '''param_grid_xgb = { 
        # Porcentagem de colunas a serem amostradas aleatoriamente para cada arvore.
        "colsample_bytree": [ 0.8 ],
        # reg_alpha promove a regularização l1 para o peso, valores mais altos resultam em modos mais conservadores
        "reg_alpha": [5],
        # reg_lambda promove a regularização l2 para o peso, valores mais altos resultam em modos mais conservadores
        "reg_lambda": [5]
        }

    # Configuração de pontuação
    #scoring = ['recall']

    # Configuração para validação cruzada
    #kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)'''

    model = XGBClassifier(
        colsample_bytree = 0.8,
        reg_alpha = 5,
        reg_lambda = 5)
    model.fit(input_train, output_train) #Treinamento

    #model.best_params_

    #joblib.dump(model, 'XGBoostClassifier.sav')

    # Fazer a classificação
    output_model_decision = model.predict(input_test)

    pickle.dump(model, open(file_name, "wb"))
    #model.save_model('XGBoost.sav')

    #Plotando

    plot_confusion_matrix(output_test, output_model_decision, model, title = 'Matriz Confusão')

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    roc(output_test, output_model_decision) #plotando a curva ROC
 
    #plotando a curva de erro
    miss_classification(input_train, output_train, input_test, output_test, model)

    learning_curves(input_train, output_train, input_test, output_test, model)