import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV

if __name__ == '__main__':

    dataset = pd.read_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_clean.csv')

    """
    The effect of collinearity is that your model will overfit
    """
    # Removing Correlated Features
    columns = ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']
    dataset.drop(columns, axis=1, inplace=True)
    dataset['sex']= dataset['sex'].astype("category").cat.codes.values 
    dataset['sick']= dataset['sick'].astype("category").cat.codes.values
    dataset['on_thyroxine'] = dataset['on_thyroxine'].astype("category").cat.codes.values
    dataset['query_on_thyroxine'] = dataset['query_on_thyroxine'].astype("category").cat.codes.values
    dataset['on_antithyroid_medication'] = dataset['on_antithyroid_medication'].astype("category").cat.codes.values
    dataset['thyroid_surgery']= dataset['thyroid_surgery'].astype("category").cat.codes.values
    dataset['query_hypothyroid']= dataset['query_hypothyroid'].astype("category").cat.codes.values
    dataset['query_hyperthyroid']= dataset['query_hyperthyroid'].astype("category").cat.codes.values
    dataset['pregnant']= dataset['pregnant'].astype("category").cat.codes.values
    dataset['tumor']= dataset['tumor'].astype("category").cat.codes.values
    dataset['lithium']= dataset['lithium'].astype("category").cat.codes.values
    dataset['goitre']= dataset['goitre'].astype("category").cat.codes.values
    
    correlations = dataset.corr()
    sns.heatmap(correlations)
    plt.show()

    dataset.to_csv('euthyroid_sick_syndrome\datasets\euthyroid\euthyroid_final_features.csv', index=False)