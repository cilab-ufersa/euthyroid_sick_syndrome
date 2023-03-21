import sys
sys.path.append('euthyroid_sick_syndrome')
import lightgbm as lgb 
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import *
from sklearn import metrics

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

    #realizando o treinamento
    gbm = lgb.LGBMRegressor(num_leaves = 31, learning_rate = 0.05, n_estimators = 20)
    gbm.fit(input_train, output_train, eval_set = [(input_train, output_train)], eval_metric = 'l1', callbacks=[lgb.early_stopping(5)])


    #realizando a predição
    y_pred = gbm.predict(input_train, num_iteration = gbm.best_iteration_)
    rmse_test = mean_squared_error(output_test, y_pred) ** 0.5
    print(f'The RMSE of prediction is: {rmse_test}')

    #printando as features mais importantes
    print(f'Feature importances: {list(gbm.feature_importances_)}')
