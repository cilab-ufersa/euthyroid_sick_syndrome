import sys
sys.path.append('euthyroid_sick_syndrome')
from sklearn.decomposition import PCA
from utils import *

# carregando o dataset
dataset = pd.read_csv('/home/vinicius/UFERSA/cilab/euthyroid_sick_syndrome/euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')
output_label_dataset = dataset['classification']
dataset = dataset.drop(['classification'], axis=1)
#Balanceamento dos dados 
dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

#Dividindo o dataset em treino e teste
#80 % para treino e 20% para teste
input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)


# Criação do objeto PCA e especificando o número de componentes
pca = PCA(n_components = 2)

# Aplicação da PCA aos dados
pca.fit(input_train)

# Transformando os dados para as novas dimensões
X_transformed = pca.transform(input_train)

# Imprimindo os dados transformados
print(X_transformed)

import matplotlib.pyplot as plt

# Plotando os dados transformados
plt.scatter(X_transformed[:,0], X_transformed[:,1])
plt.show()
