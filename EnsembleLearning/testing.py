from EnsembleTree import AdaBoost
import sys
sys.path.append('../DecisionTree')
from DecisionTree import DecisionTreeStump
import math
import pandas as pd

# Setup
target_column = "label"
columns = ["f1", "f2", "f3", "label"]
features = [col for col in columns if col != target_column]

# Training data
train_file_path = 'bank2/train.csv'
training_data = pd.read_csv(train_file_path, header=None, names=columns)

# Testing data
test_file_path = 'bank2/test.csv'
testing_data = pd.read_csv(test_file_path, header=None, names=columns)

ad = AdaBoost()
iterations = 10
for i in range(9,iterations):
    print("Iterations:", i)
    t = ad.fit(training_data, features, target_column, i)
    print(ad.test_decision_tree(testing_data, target_column))