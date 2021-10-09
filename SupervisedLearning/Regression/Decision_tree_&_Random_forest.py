import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tabulate import tabulate

data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# no need for feature scaling reprocessing in this regression: since it involves splitting of the features of dataset into groups
#this is better to use in datasets with multiple features

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x,y)

#prediction
print(dt_reg.predict([[6.5]]))

# visualisation in high resolution
#x_grid = np.arange(min(x['Level']), max(x['Level']), 0.1)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color = 'red') # we scaled it previously, so we have to inverse scale it so that it gives the original value
plt.plot(x_grid, dt_reg.predict(x_grid), color = 'blue')
plt.title("DT regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#####################################################
###           Random Forest Regression            ###

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x, y)

print(rf_reg.predict([[6.5]]))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color = 'red') # we scaled it previously, so we have to inverse scale it so that it gives the original value
plt.plot(x_grid, rf_reg.predict(x_grid), color = 'blue')
plt.title("RF regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


