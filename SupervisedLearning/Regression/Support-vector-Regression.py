import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

print(x) # 2D array o/p: [[x]]
print(y) # 1D array o/p:  [y]

# we have to transform y 1D to 2D array
#y2 = np.reshape(y, (-1, 1)) # or
y = y.reshape(len(y), 1)
print(y)

# feature scaling we need two different scalar variable
from sklearn.preprocessing import StandardScaler
 
sc_x = StandardScaler() 
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)
print(y)

# TRAINING SVR MODEL
from sklearn.svm import SVR
sv_reg = SVR(kernel = 'rbf')
sv_reg.fit(x, y)

# to predict the salary we need to inverse the transformation
print(sc_y.inverse_transform(sv_reg.predict(sc_x.transform([[6.5]]))))
#     |_____________________||_____________||_____________|-------  
#           |                      |             |            |----> 2d array for independent variable
# #         |                      |             |---->  to change the value to the scaled value
#           |                   predicts the salary on the Y scaled value                              
#      transform Y scaled values into the original prediction values     


# visualisation
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red') # we scaled it previously, so we have to inverse scale it so that it gives the original value
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(sv_reg.predict(x) ) , color = 'blue')
plt.title("SV regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# visualisation in high resolution

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red') # we scaled it previously, so we have to inverse scale it so that it gives the original value
plt.plot(x_grid, sc_y.inverse_transform(sv_reg.predict(sc_x.transform(x_grid))) , color = 'blue')
plt.title("SV regression")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()



