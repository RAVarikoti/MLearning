'''# model building
1. All in: through in all the variables (if you know all the independent variables are imp) based on your knowledge
2. Backward elimination : a) select a significance level to stay in the model (eg. SL = 0.05 or 5%)
                        b) fit the full model with all possible predictors
                        c) Consider the pridictor with the highest P-value, if P > SL, move to next step
                        d) remove the predictor
                        e) fit the model without the variable then loop from step c -- until P < SL then your model is ready ====> MODEL READY

3. forward selection : a) select a significance level to stay in the model (eg. SL = 0.05 or 5%)
                       b) fit all the simple regression models y~ Xn, select the one with low P-value
                       c) keep the above variable, and add another new variable to it i.e, add extra predictor -- 2nd 
                       d) consider the one new variable with lowest P value then move to step c then add -- 3rd.... increasing the regression model
                       e) stop -- if P value > SL the keep the model before this  ====> MODEL READY

4. bidirectional elimation: a) select a significance level to enter and stay in the model (eg. SL = 0.05 or 5%) 
                            b) fit all the simple regression models y~ Xn, select the one with low P-value
                            c) perform all the backward elimination
                            d) continue till you can enter or add any new variables ====> MODEL READY 

5. All possible models: criterion --> construct models 2^N - 1 ---> select best criterion ====> MODEL READY
'''
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tabulate import tabulate


data = pd.read_csv('50_Startups.csv', )
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# print(x)

# encoding the categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder # A one hot encoding is a representation of categorical variables as binary vectors.

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder = 'passthrough') # convert state names into numbers
x = np.array(ct.fit_transform(x)) # the state column converted into float and moved to columns 0 1 &2
# print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# no need of feature scaling in MLR since they have many variables
# * Check for MLR assumptions
# convert the state column into float variables

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train) # training the model
print(mlr.score(x_train, y_train))
# predict the test set results
# Since we have multiple variables, here we display two vectors: 1 - real profits for the test set
# 2- predicted profits for the same test set and compare

y_pred = mlr.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),axis=1)) # concatenate the vectors and display vertically
# gives length of vector rows and 1 column                                              # 1 = vertical concatination; 0 = horizontal                                  


#######################################################################################################################################################
# POLYNOMIAL LINEAR REGRESSION (PLR)

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

#from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
print('R^2 : ', lr.score(x,y))

# Polynomial regression model

from sklearn.preprocessing import PolynomialFeatures

poly_regr = PolynomialFeatures(degree=5)
x_poly = poly_regr.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
print(lin_reg2.score(x_poly, y))
# visualization

plt.scatter(x,y, color = 'black')
plt.plot(x, lr.predict(x), color = 'blue')

#plt.title("Linear regression")
#plt.show()
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg2.predict(x_poly), color = 'green')
plt.title("Poly regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.legend(["LR", "PLR"], loc ="upper left")
plt.savefig('PLR.png')
plt.show()

# predicting 

print(lr.predict([[6.5]]))
print(lin_reg2.predict(poly_regr.fit_transform([[6.5]])))



















