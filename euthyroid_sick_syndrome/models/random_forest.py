import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd #Para trabalhar com dataframes               
import numpy as np #Para trabalhar com arrays
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import RandomForestClassifier #Para criar o modelo de árvore de decisão
from utils import *
from sklearn.model_selection import RandomizedSearchCV
import joblib



if __name__ == '__main__':

    #Carregando o dataset
    dataset = pd.read_csv('/home/vinicius/UFERSA/cilab/euthyroid_sick_syndrome/euthyroid_sick_syndrome-1/euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')
    
    output_label_dataset = dataset['classification']  #1 = sick, 0 = normal

    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]

    #Balanceamento dos dados 
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                                
    #Criando o modelo de árvore de floresta aleatória
    '''
    parametros = {'min_samples_split': [2, 5, 8],
        'min_samples_leaf': np.arange(1, 5, 1, dtype=int)}

    modelRF = RandomForestClassifier(criterion='entropy', max_features ='sqrt', class_weight='balanced', max_depth=5, n_estimators=20, min_samples_split=5, min_samples_leaf=1, bootstrap=True, random_state=10)
    
    model = RandomizedSearchCV(modelRF, parametros, n_iter = 12, cv = 5, random_state = 1)
    model.fit(input_train, output_train) #Treinamento
    joblib.dump(model.best_estimator_, 'randomCV.joblib')
    print(model.best_estimator_)'''
    
    #melhores parâmetros: class_weight='balanced', criterion='entropy',
    # max_depth=5, min_samples_split=5, n_estimators=20,
    #random_state=10
                       
                       
    #model = joblib.load('/home/vinicius/UFERSA/cilab/euthyroid_sick_syndrome/euthyroid_sick_syndrome-1/randomCV.joblib')
    model = RandomForestClassifier(class_weight = 'balanced', criterion = 'entropy',
    max_depth = 5, min_samples_split = 5, n_estimators = 20)
    model.fit(input_train, output_train)

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