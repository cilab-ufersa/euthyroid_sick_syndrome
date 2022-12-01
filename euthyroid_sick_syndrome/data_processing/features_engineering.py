import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV

if __name__ == '__main__':
    
    dataset = pd.read_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_dataset.csv')

    output_label_dataset = dataset['classification']
    dataset = dataset.drop(['classification'], axis=1)

    # Finding Correlated Features

    """
    The effect of collinearity is that your model will overfit
    """

    correlations = dataset.corr()
    sns.heatmap(correlations)
    #plt.show()


    # Removing Correlated Features
    columns = ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured']

    dataset.drop(columns, axis=1, inplace=True)

    dataset.head()