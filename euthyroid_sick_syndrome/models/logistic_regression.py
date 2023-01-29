import sys
sys.path.append('euthyroid_sick_syndrome')
import pandas as pd             
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from mlxtend.plotting import plot_learning_curves
from sklearn.linear_model import LogisticRegression
import joblib
from utils.utils import * 

if __name__ == '__main__':

    dataset = pd.read_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_features.csv')
    output_label_dataset = dataset['classification']  #1 = sick, 0 = normal
    dataset = dataset[['age', 'sex', 'sick', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]
    dataset_res, ouput_label = balance_dataset_smote(dataset, output_label_dataset, random_state=42, k_neighbors=5)
    input_train, input_test, output_train, output_test = slipt_and_standardize_dataset(dataset=dataset_res, output_label=ouput_label)
                                                                

    model = LogisticRegression(penalty='l1', C=1.5, solver='saga', multi_class='ovr', max_iter=1000, class_weight='balanced', random_state=10)
    model.fit(input_train, output_train) 

    joblib.dump(model, 'LogisticRegression.sav')
    
    output_model_decision = model.predict(input_test)



    