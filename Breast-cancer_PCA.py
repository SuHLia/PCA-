#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')

#load dataset & cleaning
df = pd.read_csv('breast-cancer-data.csv', index_col = 0, thousands = ',')


df.head()


# Find missing values
print('Missing values:\n{}'.format(df.isnull().sum()))



# Find duplicated records
print('\nNumber of duplicated records: {}'.format(df.duplicated().sum()))


# Find the unique values of 'diagnosis'.
print('\nUnique values of "diagnosis": {}'.format(df['diagnosis'].unique()))


df.describe()


# seperate the features and target
X = df.drop(['diagnosis'], axis = 1)
y = df['diagnosis']

# labelencoder diagnosis to binary data, here 0 - B, 1 - M
le = preprocessing.LabelEncoder()
y = le.fit_transform(y) 
# Standardizing the features
X = StandardScaler().fit_transform(X)
# set the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# use PCA 
# pca = PCA(n_components=10)
pca = PCA(.90) # find the minimum no of principal components which variance amount over 90%
principalComponents = pca.fit(X_train)
pca.explained_variance_ratio_


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('the number of principal components is:', len(X_train_pca[0]))


logisticRegr = LogisticRegression(solver = 'lbfgs')

logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)
print('Classification_report before pca', classification_report(y_test, y_pred))
print('Confusion_matrix before pca', confusion_matrix(y_test, y_pred))
print('The accuracy of logistic model before PCA: ', logisticRegr.score(X_test, y_test))

# after pca
logisticRegr.fit(X_train_pca, y_train)
y_pred_pca = logisticRegr.predict(X_test_pca)
print('Classification_report after pca',classification_report(y_test, y_pred_pca))
print('Confusion_matrix after pca',confusion_matrix(y_test, y_pred_pca))
print('The accuracy of logistic model after PCA with variance amount to 90% is: ', logisticRegr.score(X_test_pca, y_test))






