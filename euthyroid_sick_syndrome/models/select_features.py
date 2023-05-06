import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from utils import *

#Carregando o dataset
dataset = pd.read_csv('/home/vinicius/UFERSA/cilab/euthyroid_sick_syndrome/euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')
output_label_dataset = dataset['classification']
dataset = dataset.drop(['classification'], axis=1)

#Balanceamento dos dados 
dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

#Dividindo o dataset em treino e teste
#80 % para treino e 20% para teste
input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                       

model = RandomForestClassifier(class_weight = 'balanced_subsample', criterion = 'log_loss',
max_depth = 10, min_samples_split = 2, n_estimators = 10, random_state = 10,
max_features ='sqrt', min_samples_leaf=5)
model.fit(input_train, output_train)
print("importância das features: \n", model.feature_importances_)

import seaborn as sns
plt.figure(figsize=(22,20))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Definir o RFE com 3 recursos a serem selecionados, ou seja ele está selecionando as 3 melhores features
rfe = RFE(estimator=model, n_features_to_select = 8)

# Executar o RFE no conjunto de treinamento
rfe.fit(input_train, output_train)

# Verificando os recursos selecionados
for i in range(input_train.shape[1]):
 print('Coluna: %d, Selecionado %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

#mostrando as features selecionadas
print(dataset.columns[rfe.support_])

# Selecionar os recursos do conjunto de treinamento e teste
input_train_rfe = rfe.transform(input_train)
X_test_rfe = rfe.transform(input_test)

# Criando um novo modelo com os recursos selecionados
model.fit(input_train_rfe, output_train)

# Avaliando a acuracia do modelo no conjunto de teste

y_pred = model.predict(X_test_rfe)
accuracy(output_test, y_pred)


#features [age, sex, on_thyroxine, TSH, T3, TT4, T4U, FTI]