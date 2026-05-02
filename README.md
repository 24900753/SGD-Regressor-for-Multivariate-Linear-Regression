# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset.

2.Preprocess data (handle missing values and split features/targets).

3.Scale features using StandardScaler.

4.Train the model using SGDRegressor.

5.Predict house price and number of occupants and evaluate the model.

## Program:
```
*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VIGNESH  V
RegisterNumber: 212224230303
*
```
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic multivariate data
np.random.seed(42)
X = np.random.rand(200, 3)   # 200 samples, 3 features

# True coefficients
true_coef = np.array([4, -2, 3])
y = X @ true_coef + np.random.randn(200) * 0.5  # Add noise

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature scaling (IMPORTANT for SGD)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Initialize SGD Regressor
model = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    learning_rate='invscaling',
    eta0=0.01,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Learned Coefficients:", model.coef_)
print("True Coefficients   :", true_coef)
print("Mean Squared Error :", mse)
print("R2 Score           :", r2)

# 8. Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.grid()
plt.show()
```

## Output:

<img width="750" height="526" alt="image" src="https://github.com/user-attachments/assets/973c82a9-8c22-4887-a35f-b09c634f2479" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
