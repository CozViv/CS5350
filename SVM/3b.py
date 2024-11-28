from scipy.optimize import minimize
import numpy as np
import pandas as pd

train_file_path = 'bank-note/train.csv'
test_file_path = 'bank-note/test.csv'

training_data = pd.read_csv(train_file_path)
testing_data = pd.read_csv(test_file_path)

training_x = training_data.iloc[:, :-1].values
training_y = training_data.iloc[:, -1].map({0: -1, 1: 1}).values

testing_x = testing_data.iloc[:, :-1].values
testing_y = testing_data.iloc[:, -1].map({0: -1, 1: 1}).values

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)

def compute_gram_matrix(x, gamma):
    sq_norms = np.sum(x**2, axis=1)[:, None]
    pairwise_sq_dists = sq_norms + sq_norms.T - 2 * np.dot(x, x.T)
    K = np.exp(-pairwise_sq_dists / gamma)
    return K

def train_dual_svm_gaussian(x, y, C, gamma):
    size_samples = x.shape[0]
    K = compute_gram_matrix(x, gamma)
    y_diag = np.diag(y)
    H = y_diag @ K @ y_diag

    def objective(alpha):
        return -np.sum(alpha) + 0.5 * np.dot(alpha, np.dot(H, alpha))

    constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
                   {'type': 'ineq', 'fun': lambda alpha: C - alpha},
                   {'type': 'ineq', 'fun': lambda alpha: alpha}]

    result = minimize(objective, np.zeros(size_samples), constraints=constraints)
    alpha = result.x

    support_vectors = alpha > 1e-2
    b = np.mean(y[support_vectors] - np.sum(alpha * y * K[:, support_vectors].T, axis=1))
    return alpha, b, support_vectors

def predict(x, support_vectors, alpha, training_x, training_y, gamma, b):
    size_samples = x.shape[0]
    predictions = np.zeros(size_samples)
    for i in range(size_samples):
        kernel_sum = np.sum(alpha[support_vectors] * training_y[support_vectors] * np.array([gaussian_kernel(training_x[j], x[i], gamma) for j in np.where(support_vectors)[0]]))
        predictions[i] = np.sign(kernel_sum + b)
    return predictions

def evaluate(x, y, support_vectors, alpha, training_x, training_y, gamma, b):
    predictions = predict(x, support_vectors, alpha, training_x, training_y, gamma, b)
    return np.mean(predictions != y)

#gamma_values = [0.1, 0.5, 1, 5, 100]
gamma_values = [0.01, 0.1, 0.5]
#Cs = [100/873, 500/873, 700/873]
Cs = [500/873]
results = []

# Part C
m = {}

for C in Cs:
    for gamma in gamma_values:
        alpha, b, support_vectors = train_dual_svm_gaussian(training_x, training_y, C, gamma)
        m[(C, gamma)] = support_vectors
        train_error = evaluate(training_x, training_y, support_vectors, alpha, training_x, training_y, gamma, b)
        test_error = evaluate(testing_x, testing_y, support_vectors, alpha, training_x, training_y, gamma, b)
        results.append((gamma, C, train_error, test_error))
        print(f"Gamma: {gamma}, C: {C:.4f}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}, Support Vectors: {sum(support_vectors)}")

best_combination = min(results, key=lambda x: x[3])
print(f"Best combination: Gamma={best_combination[0]}, C={best_combination[1]:.4f}, Train Error={best_combination[2]:.4f}, Test Error={best_combination[3]:.4f}")

# Part C
print("For C = 500/873")
print(f"Similar support vectors between .01 and .1: {sum(m[(500/873, .01)] & m[(500/873, .1)])}")
print(f"Similar support vectors between .1 and .5: {sum(m[(500/873, .1)] & m[(500/873, .5)])}")