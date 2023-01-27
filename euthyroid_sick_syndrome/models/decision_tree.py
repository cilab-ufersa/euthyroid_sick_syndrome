import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd #Para trabalhar com dataframes               
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.tree import DecisionTreeClassifier #Para criar o modelo de árvore de decisão
from mlxtend.plotting import plot_learning_curves
import joblib
from utils.utils import * 

if __name__ == '__main__':

    dataset =  pd.read_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_features.csv')
    output_label_dataset = dataset['classification']  #1 = sick, 0 = normal
    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                                
    #Criando o modelo de árvore de decisão
    model= DecisionTreeClassifier(criterion='entropy', max_features=None, random_state=3, class_weight='balanced', max_depth=6)
    model.fit(input_train, output_train) #Treinamento

    joblib.dump(model, 'DecisionTreeClassifier.sav')

    output_model_decision = model.predict(input_test)

    #Plotando a curva de aprendizado
    plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    scoring='misclassification error')
    plt.show()


    