# Diabetes Classification Using PySpark

## Project Overview
This project focuses on diagnosing diabetes using a classification model. The dataset used consists of 128 rows and 11 columns, with various features related to health and lifestyle factors. The models are built using **PySpark**, and different machine learning algorithms are implemented to predict whether an individual has diabetes based on the given health features.

## Dataset
The dataset used is publicly available on Kaggle:
- [Easiest Diabetes Classification Dataset](https://www.kaggle.com/datasets/sujithmandala/easiest-diabetes-classification-dataset)

### Columns
- **Age**: Age of the individual (integer)
- **Gender**: Gender of the individual (string)
- **BMI**: Body Mass Index (integer)
- **Blood Pressure**: Blood pressure status (string)
- **FBS**: Fasting Blood Sugar (integer)
- **HbA1c**: Glycated hemoglobin level (double)
- **Family History of Diabetes**: Family history of diabetes (string)
- **Smoking**: Smoking status (string)
- **Diet**: Type of diet (string)
- **Exercise**: Exercise habits (string)
- **Diagnosis**: Diabetes diagnosis (target variable - string)


### Data Preprocessing
### Handling Categorical Variables
All categorical variables are transformed into numeric values using **Label Encoding** through PySpark's `StringIndexer`. The transformation details are as follows:

- **Gender**: Transformed to `Gender_index` with values [Male, Female].
- **Blood Pressure**: Transformed to `Blood Pressure_index` with values [High, Normal, Low].
- **Exercise**: Transformed to `Exercise_index` with values [No, Regular].

### Correlation Analysis
During data exploration, certain variables were found to have perfect correlations with others. Dependent features with a perfect correlation (1) with other features were dropped. In this case, the following variables were dropped:
- **FBS** (due to perfect correlation with HbA1c),
- **Smoking_index** (due to perfect correlation with Exercise_index),
- **Diet_index** (due to perfect correlation with Smoking_index).

### Final Features
The following features were selected for the model:
- Age
- BMI
- HbA1c
- Gender_index
- Blood Pressure_index
- Family History of Diabetes_index
- Exercise_index

## Model Building
### Vector Assembling
A `VectorAssembler` was used to combine the selected features into a single vector column called `features`, which is essential for fitting machine learning models in PySpark.

### Data Splitting
The dataset was split into training and testing sets with a 70-30 ratio.

## Machine Learning Models
### 1. Logistic Regression
- **Accuracy**: 0.8929
- **Recall**: 0.8929
- **Precision**: 0.9286
- **F1-Score**: 0.8997

### 2. Decision Tree Classifier
- **Accuracy**: 0.8214
- **Recall**: 0.8214
- **Precision**: 0.8634
- **F1-Score**: 0.8328

### 3. Naive Bayes
- **Accuracy**: 0.7143
- **Recall**: 0.7143
- **Precision**: 0.6756
- **F1-Score**: 0.6919

## Model Evaluation
<img width="572" alt="image" src="https://github.com/user-attachments/assets/af2a27a9-edee-42da-8f3c-c46831310156">

The primary evaluation metric for this project is **Recall**, as it's critical to minimize false negatives in disease diagnosis. Among the models tested, **Logistic Regression** performed the best with the highest Recall score of **0.89**. Therefore, Logistic Regression is the selected model for this project.

## Conclusion
The classification models built using PySpark provided valuable insights into the factors contributing to diabetes diagnosis. Logistic Regression proved to be the best performing model with an accuracy of **89%** and a recall of **89%** on the test data.
