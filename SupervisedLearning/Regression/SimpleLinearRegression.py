import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # to disable warnings
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


data = pd.read_csv("Salary_Data.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

# SLR Model

from sklearn.linear_model import LinearRegression
lr = LinearRegression() # fit method trains the model
lr.fit(x_train, y_train) 

# predicting test set results
y_pred = lr.predict(x_test)

print(lr.predict([[20]])) # predicting the salary for 20 years of experience
print(lr.coef_)  # y = mx + C
print(lr.intercept_)


# visualization of training set results

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.savefig('SLR_train_set.png')
plt.show()

# visualization of test set results

plt.scatter(x_test, y_test, color = 'Black')
plt.plot(x_train, lr.predict(x_train), color = 'blue') # not changed w.r.t training set because regression line remains the same in both the sets.
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.savefig('SLR_test_set.png')
plt.show()
















