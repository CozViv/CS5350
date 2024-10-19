from DecisionTree import DecisionTree as dt
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

print("Part A------------") # Treating unknown as value
functions = ['entropy', 'gini', 'majority_error']
for function in functions:
    print(function)
    average_f_train = 0
    average_f_test = 0
    for i in range(1,17):
        print(i)
        md = i
        treeF = dt(f=function, max_depth=md)
        t = treeF.fit(training_data, target_column, features)
        error_rate_train = treeF.test_decision_tree(training_data, target_column)
        error_rate_test = treeF.test_decision_tree(testing_data, target_column)
        average_f_train += 1-error_rate_train
        average_f_test += 1-error_rate_test

    average_f_train /= 16
    average_f_test /= 16
    print(f"Averages for {function}...")
    print("Training set: ", average_f_train)
    print("Testing set: ", average_f_test)

# Proccess data
for column in columns:
    most_common_value = training_data[column].mode()[0]
    training_data[column].replace("unknown", most_common_value, inplace=True)
for column in columns:
    most_common_value = testing_data[column].mode()[0]
    testing_data[column].replace("unknown", most_common_value, inplace=True)


print("Part B------------") # Treating unknown as most common
functions = ['entropy', 'gini', 'majority_error']
for function in functions:
    print(function)
    average_f_train = 0
    average_f_test = 0
    for i in range(1,17):
        print(i)
        md = i
        treeF = dt(f=function, max_depth=md)
        t = treeF.fit(training_data, target_column, features)
        error_rate_train = treeF.test(training_data, target_column)
        error_rate_test = treeF.test(testing_data, target_column)
        average_f_train += 1-error_rate_train
        average_f_test += 1-error_rate_test

    average_f_train /= 16
    average_f_test /= 16
    print(f"Averages for {function}...")
    print("Training set: ", average_f_train)
    print("Testing set: ", average_f_test)