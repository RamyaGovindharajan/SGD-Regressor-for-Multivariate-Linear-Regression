# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start
2.Data preparation
3.Hypothesis Definition
4.Com Function 
5.Parameter Update Rule rate Time 
6.Itrative training
7.Model evaluston kind

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
```
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
x = data.data[:,:3]
y=np.column_stack((data.target,data.data[:,6]))
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.fit_transform(y_test)
sgd = SGDRegressor(max_iter=1000, tol = 1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train)
y_pred = multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print(y_pred)


## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)

![Screenshot 2024-11-15 093138](https://github.com/user-attachments/assets/130e5db1-72f7-4a29-973b-c0057ff7c55c)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
