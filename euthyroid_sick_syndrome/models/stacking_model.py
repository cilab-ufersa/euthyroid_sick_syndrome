import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd #Para trabalhar com dataframes               
import numpy as np #Para trabalhar com arrays
import matplotlib.pyplot as plt #Para plotar os gráficos
from sklearn.tree import DecisionTreeClassifier #Para criar o modelo de árvore de decisão
from sklearn.model_selection import train_test_split #Para dividir o dataset em treino e teste
import seaborn as sns 
from imblearn.over_sampling import SMOTE #Para balancear o dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #Para plotar a matriz de confusão
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from utils.utils import * 
import joblib

if __name__ == '__main__':
    dataset = dataset = pd.read_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_features.csv')

    output_label_dataset = dataset['classification']  #1 = sick, 0 = normal

    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]
 
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)

    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                                

    estimators = [ ('xgb', XGBClassifier(colsample_bytree = 0.8, reg_alpha = 5, reg_lambda = 5)),
    #('lg', lgb.LGBMClassifier(learning_rate = 0.3, max_depth = 15, n_estimators = 5, num_leaves = 15, subsample = 0.5)),
    ('svr', make_pipeline(RandomForestClassifier(class_weight = 'balanced_subsample', criterion = 'log_loss',
    max_depth = 10, min_samples_split = 2, n_estimators = 10, random_state = 10,
    max_features ='sqrt', min_samples_leaf=5)))
    ]
    model = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier(class_weight = 'balanced_subsample', criterion = 'log_loss',
    max_depth = 10, min_samples_split = 2, n_estimators = 10, random_state = 10,
    max_features ='sqrt', min_samples_leaf=5)
    
    
    #XGBClassifier(colsample_bytree = 0.8,
    #    reg_alpha = 5,
    #    reg_lambda = 5)
    )


    model.fit(input_train, output_train) 

    joblib.dump(model, 'euthyroid_sick_syndrome\models_file\StackingClassifier.sav')
    
    output_model_decision = model.predict(input_test)

    accuracy(output_test, output_model_decision) #Pontuação de acurácia
    
    precision(output_test, output_model_decision) #Pontuação de precisão

    recall(output_test, output_model_decision) #Pontuação de recall

    f1(output_test, output_model_decision)
    
    #roc(output_test, output_model_decision) #plotando a curva ROC


    cm = confusion_matrix(output_test, output_model_decision)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    disp.ax_.set_title('Matriz de confusão')
    disp.ax_.set_xlabel('Classificação prevista')
    disp.ax_.set_ylabel('Classificação real')
    plt.show()
       

    #Plotando a curva de aprendizado
    plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    scoring='misclassification error')
    plt.show()

    plot_learning_curves(input_train, output_train, input_test, output_test, model, 
    scoring='accuracy')
    plt.show()

    