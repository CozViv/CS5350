from scipy.optimize import minimize
import numpy as np

from SVM import SVM
import pandas as pd
import matplotlib.pyplot as plt

train_file_path = 'bank-note/train.csv'
test_file_path = 'bank-note/test.csv'

training_data = pd.read_csv(train_file_path)
testing_data = pd.read_csv(test_file_path)

training_x = training_data.iloc[:, :-1].values
training_y = training_data.iloc[:, -1].map({0: -1, 1: 1}).values

testing_x = testing_data.iloc[:, :-1].values
testing_y = testing_data.iloc[:, -1].map({0: -1, 1: 1}).values

def train_dual_svm(x, y, C):
    size_samples, size_features = x.shape
    Q = np.dot(x, x.T)
    y_diag = np.diag(y)
    H = y_diag @ Q @ y_diag

    def objective(alpha):
        return -np.sum(alpha) + 0.5 * np.dot(alpha, H @ alpha)

    constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
                   {'type': 'ineq', 'fun': lambda alpha: C - alpha},
                   {'type': 'ineq', 'fun': lambda alpha: alpha}]

    result = minimize(objective, np.zeros(size_samples), constraints=constraints, method='SLSQP')
    alpha = result.x

    w = np.sum((alpha * y)[:, None] * x, axis=0)
    support_vectors = alpha > 1e-4
    b = np.mean(y[support_vectors] - np.dot(x[support_vectors], w))
    return w, b, alpha

def predict(x, w, b):
    return np.sign(np.dot(x, w) + b)

def evaluate(x, y, w, b):
    predictions = predict(x, w, b)
    return np.mean(predictions != y)

Cs = [100/873, 500/873, 700/873]
for C in Cs:
    w, b, alphas = train_dual_svm(training_x, training_y, C=C)

    print(C)

    train_error = evaluate(training_x, training_y, w, b)
    print(f"Training Error: {train_error}")

    test_error = evaluate(testing_x, testing_y, w, b)
    print(f"Testing Error: {test_error}")
