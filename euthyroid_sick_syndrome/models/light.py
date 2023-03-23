import sys
sys.path.append('euthyroid_sick_syndrome')
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
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
    X_train, X_test, y_train, y_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)

    #realizando o treinamento
    gbm = lgb.LGBMClassifier(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
    gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(5)])

    print('Começando a predição...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f'O RMSE (raiz quadrada média de erro) da predição é: {rmse_test}')

    # feature importances
    print(f'Importância das features: {list(gbm.feature_importances_)}')


    # self-defined eval metric
    # f(y_true: array, y_pred: array) -> name: str, eval_result: float, is_higher_better: bool
    # Root Mean Squared Logarithmic Error (RMSLE)
    def rmsle(y_true, y_pred):
        return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


    print('começando o treinamento com a função eval customizada...')
    # train
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=rmsle, callbacks=[lgb.early_stopping(5)])


    # another self-defined eval metric
    # f(y_true: array, y_pred: array) -> name: str, eval_result: float, is_higher_better: bool
    # Relative Absolute Error (RAE)
    def rae(y_true, y_pred):
        return 'RAE', np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


    print('Começando o treino com multiplos funções eval customizadas...')
    # train
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=[rmsle, rae], callbacks=[lgb.early_stopping(5)])

    print('começando a predição...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    rmsle_test = rmsle(y_test, y_pred)[1]
    rae_test = rae(y_test, y_pred)[1]
    print(f'O RMSLE (raiz quadrada logaritmica de erro) da predição é: {rmsle_test}')
    print(f'O RAE (erro relativo absoluto) da predição é: {rae_test}')

    # other scikit-learn modules
    estimator = lgb.LGBMClassifier(num_leaves=31)

    param_grid = {
        'learning_rate': [1],
        'max_depth': [11],
        'n_estimators': [5],
        'num_leaves': [30],
        'feature_fraction': [0.47],
        'subsample': [0.01]
    }
    gbm = GridSearchCV(estimator, param_grid, cv=3)
    gbm.fit(X_train, y_train)

    print(f'Os melhores parametros encontrados pelo gridsearch são: {gbm.best_params_}')

    print("\n\n\n\n\n")
    plot_confusion_matrix(y_test, y_pred, gbm, title = 'Matriz Confusão')

    accuracy(y_test, y_pred) #Pontuação de acurácia
    
    precision(y_test, y_pred) #Pontuação de precisão

    recall(y_test, y_pred) #Pontuação de recall

    f1(y_test, y_pred)
    
    roc(y_test, y_pred) #plotando a curva ROC
 
    #plotando a curva de erro
    miss_classification(X_train, y_train, X_test, y_test, gbm)

    learning_curves(X_train, y_train, X_test, y_test, gbm)
    
    