# Preprocessing is where  we check anything missing or need to modified in your data file.  

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('Data.csv')
#print(data)
x = data.iloc[:, :-1].values # independent variable
y = data.iloc[:, -1].values # dependent variable

print(x)
print(y)

# working on missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3]) # not 1:2 because python removes the upper bound column values
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

# encoding independent variable -- giving a numerical value to the text/string

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform (x))
print(x)

# encoding dependent variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y) # no need of np array
print(y)

# splitting the dataset into training and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# feature scaling done after the splitting to prevent test set data leakage
# to prevent some features dominating the others 
# most ML models don't need scaling
# mostly either Standardisation or normalisation are used
''' 
X_stand = [X-mean(X)]/std.dev(X)
X_norm = [X-min(X)]/[max(X)-min(x)]
'''

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:]) # 0, 1 & 2 columns are now converted into float variables of strings
x_test[:,3:] = sc.fit_transform(x_test[:,3:])

print(x_train)
print(x_test)

#y_train[:,:]

# fit - only calculates the mean & stdev, whereas, fit_transform  calculates the mean & stdev and applies the formula                                 
# you should not standardise/feature scale the dummy variables (float variables of strings) since it results in errors 


