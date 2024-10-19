from DecisionTree import AdaBoost
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

print("Part A------------") # Treating unknown as value
functions = ['entropy']
for function in functions:
    print(function)
    average_f_train = 0
    average_f_test = 0
    for i in range(1,5):
        print(i)
        md = i
        treeF = AdaBoost(i)
        t = treeF.fit(training_data, target_column, features)
        error_rate_train = treeF.test_decision_tree(training_data, target_column)
        error_rate_test = treeF.test_decision_tree(testing_data, target_column)
        print(1-error_rate_train)
        print(1-error_rate_test)
        average_f_train += 1-error_rate_train
        average_f_test += 1-error_rate_test

# Proccess data
for column in columns:
    most_common_value = training_data[column].mode()[0]
    training_data[column].replace("unknown", most_common_value, inplace=True)
for column in columns:
    most_common_value = testing_data[column].mode()[0]
    testing_data[column].replace("unknown", most_common_value, inplace=True)