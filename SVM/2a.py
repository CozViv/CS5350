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

Cs = [100/873, 500/873, 700/873]
gamma = .1
a = .5
max_epochs = 100

def schedule_1(gamma, a, t):
    return gamma / (1 + (gamma / a) * t)

for C in Cs:
    svm = SVM(
        C=C,
        schedule=schedule_1,
        gamma=gamma,
        a=a,
        max_epochs=max_epochs
    )

    svm.train(training_x, training_y)

    svm_preds_testing = svm.predict(testing_x)
    average_error_rate_testing = svm.evaluate(testing_x, testing_y)

    svm_preds_training = svm.predict(training_x)
    average_error_rate_training = svm.evaluate(training_x, training_y)

    svm_weights = svm.w
    svm_bias = svm.b

    print(f'C = {C}')
    print(f'SVM Average Error Rate testing: {average_error_rate_testing}')
    print(f'SVM Average Error Rate training: {average_error_rate_training}')
    print(f'SVM Weights: {svm_weights}')
    print(f'SVM Bias: {svm_bias}')

    # objective_values = svm.objective_function_values
    # plt.figure(figsize=(10, 6))
    # plt.plot(objective_values, label="Objective Function")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective Function Value")
    # plt.title("Convergence of the Objective Function")
    # plt.legend()
    # plt.grid()
    # plt.show()

