# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
5. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets.
6. Create an SGD Regressor model.
7. Fit the model on the training data.
8. Evaluate the model performance.
9. Make predictions and visualize the results.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Sujin M L
RegisterNumber: 212225040435

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)

# Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

# Standardizing the data
scaler = StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

# Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000,tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train,y_train)

# Making predictions
y_pred=sgd_model.predict(x_test)

# Evaluating model performance
mse=mean_squared_error(y_test,y_pred)

print("="*50)
print('Name: Sujin M L')
print('Reg No:212225040435')
print(f"MSE: {mse:.4f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print("="*50)

# Print model coefficients
print("Model Coefficients:")
print("Coefficiens:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
*/
```

## Output:
<img width="845" height="621" alt="ML EXP4 (1)" src="https://github.com/user-attachments/assets/e435e900-1b8d-4e68-b797-d5c5a19f3ba7" />

<img width="531" height="722" alt="ML EXP4 (2)" src="https://github.com/user-attachments/assets/e198ef86-b1c1-4d35-a84b-7d570cf76438" />

<img width="855" height="363" alt="ML EXP4 (3)" src="https://github.com/user-attachments/assets/c85f0563-1044-4925-ba50-73c76adbcc08" />

<img width="810" height="571" alt="ML EXP4 (4)" src="https://github.com/user-attachments/assets/d9e62ac6-a230-4509-9856-83a3872a8f8f" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
