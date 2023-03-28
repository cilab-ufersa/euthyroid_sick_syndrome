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
        learning_rate = 0.3,
        max_depth = 15,
        n_estimators = 5,
        num_leaves = 15,  #10
        subsample = 0.5 
    )
    '''param_grid = {
        'learning_rate': [1],
        'max_depth': np.arange(1, 100, 10),
        'n_estimators': np.arange(1, 100, 10),
        'num_leaves': np.arange(10, 100, 10),
        #'feature_fraction': np.arange(0.1, 1.0, 0.5),
        'subsample': np.arange(0.1, 1.0, 0.5)
    }'''
    #model = GridSearchCV(estimator = lgb.LGBMClassifier(), param_grid = param_grid)
    model.fit(input_train, output_train)

    #realizando predições
    output_model_decision = model.predict(input_test)

    print("\n\n\n\n\n")
    #print(model.best_estimator_)
    #plot_confusion_matrix(output_test, output_model_decision, model, title = 'Matriz Confusão')

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    roc(output_test, output_model_decision) #plotando a curva ROC
 
    #plotando a curva de erro
    miss_classification(input_train, output_train, input_test, output_test, model)

    #learning_curves(input_train, output_train, input_test, output_test, model)