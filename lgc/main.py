import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from scipy.linalg import fractional_matrix_power

data_vertex = 500
data_labeled = 500
alpha = 0.99
sigma = 0.1

#Criação das luas para a rotulação de dados
X, Y = make_moons(data_vertex, shuffle=True, noise=0.1, random_state=None)

color = ['green' if label == 0 else 'red' for label in Y]
plt.scatter(X[:, 0], X[:, 1], color=color)
plt.show()

Y_labeled = np.concatenate(((Y[:data_labeled, None] == np.arange(2)).astype(float), np.zeros((data_vertex - data_labeled, 2))))

distance_matrix = cdist(X, X, 'euclidean')
rbf_kernel = lambda x, sigma: math.exp((-x) / (2 * (math.pow(sigma, 2))))
vfunc = np.vectorize(rbf_kernel)
weight_matriz = vfunc(distance_matrix, sigma)
np.fill_diagonal(weight_matriz, 0)

sum_rows = np.sum(weight_matriz, axis=1)
diagonal_matriz = np.diag(sum_rows)

D_inv_sqrt = fractional_matrix_power(diagonal_matriz, -0.5)
S = np.dot(np.dot(D_inv_sqrt, weight_matriz), D_inv_sqrt)

n_iter = 400

F = np.dot(S, Y_labeled) * alpha + (1 - alpha) * Y_labeled
for t in range(n_iter):
    F = np.dot(S, F) * alpha + (1 - alpha) * Y_labeled

Y_result = np.zeros_like(F)
Y_result[np.arange(len(F)), F.argmax(1)] = 1

Y_predicted = [1 if label == 0 else 0 for label in Y_result[:, 0]]

color = ['green' if label == 0 else 'red' for label in Y_predicted]
plt.scatter(X[:, 0], X[:, 1], color=color)
plt.show()