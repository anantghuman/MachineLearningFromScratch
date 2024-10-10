import numpy as np
import pandas as pd

data = pd.read_csv("Iris.csv")
X = data.iloc[:, :-1].values  
Y = data.iloc[:, -1].values 

def initialize_centroids(k, X):
    if k > X.shape[0]:
        return None
    random_points = np.random.choice(X.shape[0], k, replace=False)
    return X[random_points]

def euclidian_distance(X, centroids):
    return np.sqrt(np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))

def calc(centroids, X):
    distance = euclidian_distance(X, centroids)
    return np.argmin(distance, axis=1)  
  
def compute_mean(X, points, k):
    return np.array([X[points == i].mean(axis=0) for i in range(k)])

def fit(X, k, iterations):
    centroids = initialize_centroids(k, X)  
    for i in range(iterations):
        points = calc(centroids, X)     
        centroids = compute_mean(X, points, k)  
    return centroids, points  


k = 3  
iterations = 10000  

final_centroids, final_assignments = fit(X, k, iterations)

print("Final Centroids:\n", final_centroids)
print("Final Assignments:\n", final_assignments)
