import numpy as np
import pandas as pd

class StandardPerceptron():
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def train(self, x, y):
        sample_size, feature_size = x.shape

        self.weights = np.zeros(feature_size)
        self.bias = 0

        for _ in range(self.iterations):
            for i in range(sample_size):
                xTemp = x.iloc[i].values
                pred = self.predict(xTemp)

                self.weights += (y.iloc[i] - pred) * xTemp
                self.bias += (y.iloc[i] - pred)
    
    def predict(self, x):
        return 1 if (np.dot(x, self.weights) + self.bias) >= 0 else -1
    
    def predict_all(self, x):
        return np.array([self.predict(xTemp) for xTemp in x.values])
    
    def calculate_error_rate(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def get_weights(self):
        return self.weights

class VotedPerceptron():
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.bias = None
        self.weights_dict = {}
    
    def train(self, x, y):
        sample_size, feature_size = x.shape

        weights = np.zeros(feature_size)
        self.bias = 0

        for _ in range(self.iterations):
            for i in range(sample_size):
                xTemp = x.iloc[i].values
                pred = self.predict_single_weights(xTemp, weights)

                if pred != y.iloc[i]:
                    weights += (y.iloc[i] - pred) * xTemp
                    self.bias += (y.iloc[i] - pred)

                    weight_tuple = tuple(weights)
                    if weight_tuple in self.weights_dict:
                        self.weights_dict[weight_tuple] += 1
                    else:
                        self.weights_dict[weight_tuple] = 1
                else:
                    last_weight_tuple = tuple(weights)
                    if last_weight_tuple in self.weights_dict:
                        self.weights_dict[last_weight_tuple] += 1
    
    def predict_single_weights(self, x, weights):
        return 1 if np.dot(x, weights) + self.bias >= 0 else -1
    
    def predict(self, x):
        weighted_votes = {0: 0, 1: 0}
        for weights_tuple, vote_count in self.weights_dict.items():
            pred = np.dot(x, np.array(weights_tuple)) + self.bias
            prediction = 1 if pred >= 0 else 0
            weighted_votes[prediction] += vote_count
        return 1 if weighted_votes[1] >= weighted_votes[0] else -1
    
    def predict_all(self, x):
        return np.array([self.predict(xTemp) for xTemp in x.values])
    
    def calculate_error_rate(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def get_weights_with_counts(self):
        return self.weights_dict

class AveragedPerceptron():
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.accumulated_weights = None
        self.accumulated_bias = 0
        self.updates = 0
    
    def train(self, x, y):
        sample_size, feature_size = x.shape
        
        self.accumulated_weights = np.zeros(feature_size)
        self.weights = np.zeros(feature_size)
        self.bias = 0

        for _ in range(self.iterations):
            for i in range(sample_size):
                xTemp = x.iloc[i].values
                prediction = np.dot(self.weights, xTemp) + self.bias

                if y.iloc[i] * prediction <= 0:
                    self.weights += y.iloc[i] * xTemp
                    self.bias += y.iloc[i]

                self.accumulated_weights += self.weights
                self.accumulated_bias += self.bias
                self.updates += 1
    
    def predict(self, x):
        temp_weights = self.accumulated_weights / self.updates
        temp_bias = self.accumulated_bias / self.updates
        return 1 if (np.dot(x, temp_weights) + temp_bias) >= 0 else -1
    
    def predict_all(self, x):
        return np.array([self.predict(xTemp) for xTemp in x.values])
    
    def calculate_error_rate(self, y_true, y_pred):
        return np.mean(y_true != y_pred)

    def get_weights(self):
        return self.accumulated_weights / self.updates