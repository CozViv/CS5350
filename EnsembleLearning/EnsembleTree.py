import math
import pandas as pd
import numpy as np
import sys
sys.path.append('../DecisionTree')
from DecisionTree import DecisionTreeStump

class AdaBoost:
    def __init__(self, iterations=10):
        self.iterations = iterations
        self.alphas = []
        self.stumps = []
    
    def fit(self, data, features, target, iterations):
        weights = [1/len(data)] * len(data)
        for i in range(iterations):
            best_weighted_gain = 0
            best_feature_index = 0
            best_stump = None
            # Get best attribute
            for feature in features:
                # Calculate gain
                feature_index = features.index(feature)
                stump = DecisionTreeStump()
                gain = stump.calculate_gain(data, feature_index, target, weights)
                scaled_gain = gain * weights[feature_index]

                if scaled_gain > best_weighted_gain:
                    best_feature_index = feature_index
                    best_weighted_gain = gain
                    best_stump = stump
            print(features[best_feature_index])

            alpha = .5 * math.log((1-weights[best_feature_index])/(weights[best_feature_index] + 1e-10))

            # Add alpha and stumps into arrays
            self.alphas.append(alpha)
            self.stumps.append(stump)
            
            # Find misclassified rows
            predictions = [best_stump.predict(row) for row in data]

            for i in range(len(predictions)):
                if predictions[i] != data.iloc[i][target]:
                    weights[i] *= math.exp(alpha)
                else:
                    weights[i] *= math.exp(-1*alpha)
            
            sum_weights = sum(weights)
            for i in range(len(weights)):
                weights[i] /= sum_weights
        
    def predict(self, item, features, target):
        ans = 0
        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            stump = self.stumps[i]
            pred = stump.predict(item)
            if pred == "yes":
                ans += alpha
            else:
                ans -= alpha
        return "yes" if ans > 0 else "no"

    
    def calculate_error_rate(self, predictions, actual_labels):
        incorrect_predictions = sum(pred != actual for pred, actual in zip(predictions, actual_labels))
        error_rate = incorrect_predictions / len(actual_labels)
        return error_rate

    def test_decision_tree(self, test_data, target):
        predictions = [self.predict(self, instance, target) for _, instance in test_data.iterrows()]
        actual_labels = test_data[target].tolist()
        error_rate = self.calculate_error_rate(predictions, actual_labels)
        return error_rate
            