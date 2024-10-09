import pandas as pd

def mse(m, b, data):
    error = 0
    for i in range(len(data)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        error += (y -  m * x + b) ** 2
    return error/len(data)

def gradient_descent(m_curr, b_curr, data, L):
    m_grad = 0
    b_grad = 0

    n = len(data)
    for i in range(n):
        x = data.iloc[i].x
        y = data.iloc[i].y
        m_grad += -(2/n) * x * (y - m_curr * x + b_curr)
        b_grad += -(2/n) * (y - m_curr * x + b_curr)
    m = m_curr - m_grad * L
    b = b_curr - b_grad * L
    return m,b

data = pd.read_csv('test.csv')
m = 0
b = 0
L = .0001
epochs = 100

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
print(m, b)
