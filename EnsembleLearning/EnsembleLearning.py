from EnsembleTree import AdaBoost
import sys
sys.path.append('../DecisionTree')
from DecisionTree import DecisionTreeStump
import math
import pandas as pd

# Setup
target_column = "label"
columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"]
types = {
    "age":  int,
    "job": 'category',
    "marital": 'category',
    "education": 'category',
    "default": 'category',
    "balance": int,
    "housing": 'category',
    "loan": 'category',
    "contact": 'category',
    "day": int,
    "month": 'category',
    "duration": int,
    "campaign": int,
    "pdays": int,
    "previous": int,
    "poutcome": 'category',
    "label": 'category'
}
features = [col for col in columns if col != target_column]

# Training data
train_file_path = 'bank/train.csv'
training_data = pd.read_csv(train_file_path, header=None, names=columns)

# Testing data
test_file_path = 'bank/test.csv'
testing_data = pd.read_csv(test_file_path, header=None, names=columns)

ad = AdaBoost()
iterations = 60
for i in range(50,iterations):
    print("Iterations:", i)
    t = ad.fit(training_data, features, target_column, i)
    print(ad.test_decision_tree(testing_data, target_column))