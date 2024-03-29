# Euthyroid sick syndrome classification with machine learning approaches 🔬

<div>
  <img src="https://github.com/cilab-ufersa/euthyroid_sick_syndrome/blob/develop/icon.png" width="200" height="200">
</div>

In this project, we are going to classify patients with euthyroid sick syndrome (ESS) using machine learning approaches. The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Euthyroid+Sick+Syndrome+Classification+Data+Set). The dataset contains 3772 instances and 25 attributes. The dataset is imbalanced with 95% of the instances are labeled as "negative" and 5% are labeled as "positive". The goal of this project is to build a model that can classify patients with ESS with high accuracy.

## Prerequisites

What things you need to have to be able to run:

  * Python 3.6 +
  * Pip 3+
  * VirtualEnvWrapper is recommended but not mandatory

## Requirements 

```bash
$ pip install -r requirements.txt
```

## About 

Euthyroid is a term used to describe a normal thyroid function. The thyroid is a gland located in the neck that produces hormones that regulate the body's metabolism. These hormones, called triiodothyronine (T3) and thyroxine (T4), help to control the body's energy levels and metabolism, as well as heart rate and body temperature.

A euthyroid state means that the thyroid is functioning normally and producing the appropriate amount of hormones. The levels of T3 and T4 are within the normal range and the thyroid-stimulating hormone (TSH) produced by the pituitary gland is also within the normal range. This is the typical state for most people, and having euthyroid status is important for maintaining overall health and well-being.

However, if the thyroid gland is underactive (hypothyroidism) or overactive (hyperthyroidism) it will affect the levels of T3, T4, and TSH and can cause symptoms such as fatigue, weight gain or loss, changes in heart rate and many others. In those cases, the treatment is usually hormone replacement therapy.

## Some of the attributes in the dataset

- Levothyroxine  (T4 /T4U)
- Triiodothyronine  (T3)
- Total  T4 (TT4)
- Free  T4  Index  (FTI) 
- Thyroid  Stimulating  Hormone  (TSH)

We used the above attributes to build a model that can classify patients with ESS with high accuracy. These are chosen because they are the most important attributes in the dataset. Moreover, theses attributes can be measured in a blood test. 

## Part 1: Results : IVCobiCET 

We used 4 different machine learning approaches to build a model that can classify patients with ESS with high accuracy. The approaches are:

  * Naive Bayes
  * Logistic Regression
  * Decision Tree
  * Random Forest

The results are shown in the table below:

| Approach | Accuracy | Precision | Recall | F1-Score |
| ------ | ------ | ------ | ------ | ------ |
| Naive Bayes | 0.8493 | 0.7963 | 0.9285 | 0.8573 |
| Logistic Regression | 0.9198 | 0.9063 | 0.9321 | 0.9190 |
| Decision Tree | 0.9817 | 0.9719 | 0.9911 | 0.9814 |
| Random Forest | 0.9834 | 0.9839 | 0.9821| 0.9830 |

[Access the detailed results](https://github.com/cilab-ufersa/euthyroid_sick_syndrome/blob/develop/euthyroid_sick_syndrome/notebooks/IVCoBiCET_results.ipynb)

## Part 2: Results : ICISNA 2023

We used 4 different machine learning approaches to build a model that can classify patients with ESS with high accuracy. The approaches are:

  * Logistic Regression
  * Random Forest
  * LightGBM
  * XGBoost
  * Stack Ensemble based on Random Forest and XGBoost

The results are shown in the table below:

| Approach | Accuracy | Recall |  Precision | F1-Score |
| ------ | ------ | ------ | ------ | ------ |
| Logistic Regression | 91.98%     |   93.21%  | 90.62%  |   91.90%     
| Random Forest |  98.34%    | 98.21%  | 98.38%     | 98.30%
| LightGBM | 97.64%      |   97.32%     |      97.64%    |     97.58%  
| XGBoost | 98.60%      | 98.77%      |      98.57%     |    98.57%
| Stack Ensemble | 98.78%       | 98.75%        |   98.75%          |  98.75%      

* [Access the detailed results](https://github.com/cilab-ufersa/euthyroid_sick_syndrome/blob/develop/euthyroid_sick_syndrome/notebooks/ICISNA23.ipynb)

* [Web system](https://github.com/cilab-ufersa/euthyroid_diagnostic_support_app)

## Scientific Developers
👤 [Vinicius Almeida](https://github.com/vinicius-a-almeida): 
  _vinicius45anacleto@gmail.com_
  
👤 [Caio Moisés](https://github.com/caiomoises):  _caio.cavalcante@alunos.ufersa.edu.br_

### Technical and scientific support 

👤 [Rosana Rego](https://github.com/roscibely)
  
## Support by 
<div>

  <img src="https://github.com/roscibely/algorithms-and-data-structure/blob/main/Ufersa.png" width="70" height="100">
</div>
