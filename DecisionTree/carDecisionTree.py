from DecisionTree import DecisionTree as dt
import math
import pandas as pd

# Setup
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
target_column = "label"
features = [col for col in columns if col != target_column]

# Training data
train_file_path = 'car/train.csv'
training_data = pd.read_csv(train_file_path, header=None, names=columns)

# Testing data
test_file_path = 'car/test.csv'
testing_data = pd.read_csv(test_file_path, header=None, names=columns)


functions = ['entropy', 'gini', 'majority_error']
for function in functions:
    average_f_train = 0
    average_f_test = 0
    for i in range(1,7):
        md = i
        treeF = dt(f=function, max_depth=md)
        t = treeF.fit(training_data, target_column, features)
        error_rate_train = treeF.test_decision_tree(training_data, target_column)
        error_rate_test = treeF.test_decision_tree(testing_data, target_column)
        average_f_train += 1-error_rate_train
        average_f_test += 1-error_rate_test

    average_f_train /= 6
    average_f_test /= 6
    print(f"Averages for {function}...")
    print("Training set: ", average_f_train)
    print("Testing set: ", average_f_test)