import sys
sys.path.append('euthyroid_sick_syndrome')
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import *

#carregando o dataset e utilizando as 8 features retornadas pelo RFE

dataset = pd.read_csv('euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_features.csv')
output_label_dataset = dataset['classification']  #1 = sick, 0 = normal
dataset = dataset[['age', 'on_thyroxine', 'query_hypothyroid', 'TSH', 'T3', 'TT4', 'T4U',
'FTI']]

#fazendo o balanceamento dos dados

dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)



print(dataset_res.shape)


# Dividindo os dados em treino e teste em 80% e 20% respectivamente
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)

# Modelo de rede perceptron multicamadas com uma camada oculta de 64 neurônios
# e camada de saída com 1 neurônios (1 classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=8, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='softmax'))
model.summary() #visualizando o modelo


# Compile model
# Otimizador Adam com taxa de aprendizado de 0.01
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
# Função de custo categorical_crossentropy (para problemas de classificação com mais de duas classes)
# Métrica de avaliação MSE (Mean Squared Error)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['mse'])

# Train model
history = model.fit(input_train, output_train, validation_split=0.2, epochs=100, batch_size=10)

# Plot training & validation accuracy values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Função Perda')
plt.ylabel('MSE')
plt.xlabel('Epocas')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()


#padronizando os dados de teste
sc = StandardScaler()
sc.fit(input_train)

teste = [[45, 0, 0, 1.9, 1.0,	82.0,	0.73,	112.0]]
teste = sc.transform(teste)
# Predict
"""
    1 - Sick
    0 - Normal
"""
y_predd = model.predict(teste)
y_predd = np.argmax(y_predd, axis=1)+1 
print("A classe é:",  y_predd[-1])
