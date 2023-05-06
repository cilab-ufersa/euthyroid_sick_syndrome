import sys
sys.path.append('euthyroid_sick_syndrome')
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from utils import *

# Gerando dados de exemplo
dataset = pd.read_csv('/home/vinicius/UFERSA/cilab/euthyroid_sick_syndrome/euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')
output_label_dataset = dataset['classification']
dataset = dataset.drop(['classification'], axis=1)
#Balanceamento dos dados 
dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

#Dividindo o dataset em treino e teste
#80 % para treino e 20% para teste
input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(input_train)

# Criando clusters usando KMeans
kmeans = KMeans(n_clusters = 5, random_state = 42)
kmeans.fit(X)

# Selecionando recursos importantes usando LassoCV
# Selecionando 8 features com o parametro max_features=8
lasso = LassoCV(cv=5)
selection = SelectFromModel(estimator=lasso, threshold=-np.inf, max_features = 8)
selection.fit(X, kmeans.labels_)
selected_features = selection.get_support(indices=True)

# Imprimindo os recursos selecionados
print("Recursos selecionados: ", dataset.columns[selected_features])