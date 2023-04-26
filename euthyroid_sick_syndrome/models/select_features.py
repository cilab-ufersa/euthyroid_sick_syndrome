import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from utils import *

#Carregando o dataset
dataset = pd.read_csv('euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')
output_label_dataset = dataset['classification']
dataset = dataset.drop(['classification'], axis=1)
#output_label_dataset = dataset['classification']  #1 = sick, 0 = normal
#dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]
#dataset = dataset[['classification', 'age', 'sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery','query_hypothyroid','query_hyperthyroid','pregnant', 'sick', 'tumor','lithium', 'goitre', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]


#Balanceamento dos dados 
dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

#Dividindo o dataset em treino e teste
#80 % para treino e 20% para teste
input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                       




model = RandomForestClassifier(class_weight = 'balanced_subsample', criterion = 'log_loss',
max_depth = 10, min_samples_split = 2, n_estimators = 10, random_state = 10,
max_features ='sqrt', min_samples_leaf=5)
model.fit(input_train, output_train)
print("importância das features: ", model.feature_importances_)



# Definir o RFE com 3 recursos a serem selecionados, ou seja ele está selecionando as 3 melhores features
rfe = RFE(estimator=model, n_features_to_select=3)

# Executar o RFE no conjunto de treinamento
rfe.fit(input_train, output_train)

# Verificar os recursos selecionados
print("Verificar os recursos selecionados")
print(rfe.support_)
print(rfe.ranking_)
# Selecionar os recursos do conjunto de treinamento e teste
input_train_rfe = rfe.transform(input_train)
X_test_rfe = rfe.transform(input_test)

# Criar um novo modelo com os recursos selecionados
model.fit(input_train_rfe, output_train)

# Avaliar a precisão do modelo no conjunto de teste


y_pred = model.predict(X_test_rfe)
accuracy(output_test, y_pred)
