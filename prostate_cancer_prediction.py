"""
Importing the Dependencies
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""# New Section

Data Collection & Processing
"""

# loading the data
dataset = pd.read_csv("P2.csv")

print(dataset)

# loading the data to a data frame
data_frame = dataset

# print the first 5 rows of the dataframe
data_frame.head()

# print last 5 rows of the dataframe
data_frame.tail()

# number of rows and columns in the dataset
data_frame.shape

# getting some information about the data
data_frame.info()

# getting some information about the data
data_frame.info()

# checking for missing values
data_frame.isnull().sum()

# statistical measures about the data
data_frame.describe()

# checking the distribution of Target Varibale
data_frame['sample_type'].value_counts()

#plot
sns.countplot(data_frame['sample_type'],label="count")

data_frame.groupby('sample_type').mean()

"""Separating the features and target"""

X = data_frame.drop(columns='sample_type', axis=1)
Y = data_frame['sample_type']

print(X)

print(Y)

"""Splitting the data into training data & Testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Logistic Regression
"""

model = LogisticRegression()

# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

"""Building a Predictive System"""
# TEST 1
from re import M
input_data = (0.7,0.6,1.2,2010,1,0,53,90,15,7,7,1)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction == 'Primary Tumor'):
  print('The Prosate cancer is at high risk')

else:
  print('The Prostate Cancer is at low risk')

# TEST 2
from re import M
input_data = (1.2,0.5,1.7,2008,1,0,61,0,0,0,0,0)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction == 'Primary Tumor'):
  print('The Prosate cancer is at high risk')

else:
  print('The Prostate Cancer is at low risk')