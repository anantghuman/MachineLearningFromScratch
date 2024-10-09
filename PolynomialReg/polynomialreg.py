import numpy as np
import pandas as pd

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def pred_y(x, coefficients):
    y_pred = 0
    for i in range(len(coefficients)):
        y_pred += coefficients[i] * (x ** i)
    return y_pred

def gradient_descent(x, y, degree, learning_rate, iterations):
    coefficients = np.zeros(degree + 1) 
    for _ in range(iterations):
        gradient = np.zeros(degree + 1)
        for j in range(len(x)):
            error = pred_y(x[j], coefficients) - y[j]
            for i in range(degree + 1):
                gradient[i] += error * (x[j] ** i)
        # Update coefficients
        coefficients -= learning_rate * gradient / len(x)  # Normalize by number of samples
    return coefficients

# def pred_y(x, coefficients):
#     # Vectorized prediction calculation
#     x_poly = np.vander(x, len(coefficients), increasing=True)  # Create Vandermonde matrix for polynomial terms
#     return np.dot(x_poly, coefficients)  # Dot product between Vandermonde matrix and coefficients

# def gradient_descent(x, y, degree, learning_rate, iterations):
#     coefficients = np.zeros(degree + 1)
#     x_poly = np.vander(x, degree + 1, increasing=True)  # Precompute Vandermonde matrix for efficiency
#     for _ in range(iterations):
#         y_pred = np.dot(x_poly, coefficients)  # Calculate predicted y
#         error = y_pred - y
#         gradient = np.dot(x_poly.T, error) / len(x)  # Vectorized gradient calculation
#         coefficients -= learning_rate * gradient  # Update coefficients
#     return coefficients

d = pd.read_csv('synthetic-1.csv', header = None)
print(gradient_descent(d[d.columns[0]],d[d.columns[1]], 3, .0001, 100))

