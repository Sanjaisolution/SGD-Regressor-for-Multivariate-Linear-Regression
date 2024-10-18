# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import Libraries:

1.Import necessary libraries for data handling, model building, and evaluation (e.g., NumPy, scikit-learn).
Load Dataset:

2.Fetch the California housing dataset using fetch_california_housing().
Prepare Features and Targets:

3.Select relevant features (e.g., the first three columns of the dataset).
Create target variables as a combination of house prices (target) and the number of occupants (another feature, e.g., column 6).
Split the Data:

4.Use train_test_split() to divide the dataset into training and testing sets. Set aside a portion (e.g., 20%) for testing.
Scale the Data:

5.Standardize the feature set (X) using StandardScaler().
Standardize the target set (Y) using another StandardScaler().
Initialize SGD Regressor:

6.Create an instance of SGDRegressor with specified parameters (e.g., max_iter and tol).
Multi-Output Regression:

7.Wrap the SGD regressor with MultiOutputRegressor to handle multiple outputs (house price and number of occupants).
Train the Model:

8.Fit the model on the training data using the fit() method.
Make Predictions:

9.Predict the target values for the test set using the predict() method.
Inverse Transform Predictions:

10.Transform the predicted and actual target values back to their original scale using the inverse transform of the scaler.
Evaluate the Model:



## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SANJAI.R
RegisterNumber:  212223040180
*/
```py
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Pradeep E
RegisterNumber:  212223230149
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
X=data.data[:, :3]
Y=np.column_stack((data.target, data.data[:, 6]))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)

Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
```


## Output:
![Screenshot 2024-10-18 111730](https://github.com/user-attachments/assets/2c9870a4-6114-49f2-b357-bcf55f49ea77)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
