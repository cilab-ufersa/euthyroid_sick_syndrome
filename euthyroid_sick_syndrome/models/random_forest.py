import pandas as pd #Para trabalhar com dataframes               
import numpy as np #Para trabalhar com arrays
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.ensemble import RandomForestClassifier #Para criar o modelo de árvore de decisão
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
import seaborn as sns 
from imblearn.over_sampling import SMOTE #Para balancear o dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Para plotar a matriz de confusão
from sklearn.metrics import accuracy_score #para calcular a acuracia do modelo


if __name__ == '__main__':
    #Carregando o dataset
    dataset = dataset = pd.read_csv('/home/vinicius/UFERSA/cilab/euthyroid_sick_syndrome/euthyroid_sick_syndrome-1/euthyroid_sick_syndrome/datasets/euthyroid/euthyroid_final_dataset.csv')
    #Verificando o dataset
    print(dataset.head())
    print(dataset.describe())

    output_label_dataset = dataset['classification']
    dataset = dataset.drop(['classification'], axis=1)
    print(output_label_dataset.value_counts()) #1 = sick, 0 = normal

    #Verificando a correlação entre as variáveis
    #Plotando o heatmap
    plt.figure(figsize=(22,20))
    cor = dataset.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    #Balanceamento dos dados 
    sm = SMOTE(random_state=42, k_neighbors=5)
    dataset_res, ouput_label = sm.fit_resample(dataset, output_label_dataset)

    #Dividindo o dataset em treino e teste
    #80 % para treino e 20% para teste
    input_train, input_test, output_train, output_test = train_test_split(dataset_res, ouput_label, test_size=0.2, random_state=23)
    
    #Criando o modelo de árvore de decisão
    model= RandomForestClassifier()
    model.fit(input_train, output_train) #Treinamento

    # Fazer a classificação 
    output_model_decision = model.predict(input_test)

    #Plotando a matriz de confusão
    cm = confusion_matrix(output_test, output_model_decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    disp.ax_.set_title('Matriz de confusão')
    disp.ax_.set_xlabel('Classificação prevista')
    disp.ax_.set_ylabel('Classificação real')
    plt.show()
    accuracy = accuracy_score(output_test, output_model_decision)
    print("a acuracia é: ", accuracy)