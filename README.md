# Credit Card Fraud Detection using Machine Learning

![Banner](docs/assets/images/banner_delgado4.jpg)

This script performs a machine learning method to detect credit card fraudulent transactions. 
The dataset contains 284,807 credit card transactions made in 2013 in Europe, with 492 frauds. The variables are not the original due to confidentiality reasons, but 28 new ones from a principal component analysis (PCA) data reduction. Only two features were not transformed: 'Time' and 'Amount'. The target variable is 'Class' and has a value 1 (fraud) or 0 (not fraud). The dataset can be found [here](https://tinyurl.com/4zvuh435/).

### Python code:

### 1. Import libraries
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```
### 2. Load the csv data
```
df = pd.read_csv('creditcard.csv')

# The dataset is highly unbalanced, as the positive Class values (frauds) account for only 0.172% of 
# transactions. Therefore, cases with Class = 0 will be undersampled:
```
### 3. Undersampling the imbalanced dataset:
```
# Find class counts
counts = df['Class'].value_counts()
min_count = counts.min()

# Undersampling function
df_balanced = pd.concat([
    df[df['Class'] == cls].sample(n=min_count, random_state=42)
    for cls in counts.index
])

# Undersample and save the new smaller dataset
df_balanced.sample(frac=1, random_state=42).to_csv('creditcard_2.csv', index=False)
print('Undersampled output dataset has been saved')
```
### 4. Data info
```
df.info()
```
Output:

![datainfo](docs/assets/images/datainfo.jpg)

### 5. Define X and y variables
```
X = df.drop(columns=['Class'], axis=1)
y = df['Class']
```
### 6. Model Training and Testing
```
# Splitting the Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.25, random_state=42)
```
### 7. XGBoost classification model
```
from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)
```
### 8. Training the model
```
model.fit(x_train, y_train)
```
### 9. Predicting and testing
```
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))
```
