from Perceptron import StandardPerceptron
import pandas as pd

train_file_path = 'bank-note/train.csv'
test_file_path = 'bank-note/test.csv'

training_data = pd.read_csv(train_file_path)
testing_data = pd.read_csv(test_file_path)

training_x = training_data.iloc[:, :-1]
training_y = training_data.iloc[:, -1]

testing_x = testing_data.iloc[:, :-1]
testing_y = testing_data.iloc[:, -1]

training_y = training_y.map({0: -1, 1: 1})
testing_y = testing_y.map({0: -1, 1: 1})

sp = StandardPerceptron()

sp.train(training_x, training_y)

sp_preds = sp.predict_all(testing_x)

sp_error_rate = sp.calculate_error_rate(sp_preds, testing_y)
sp_weights = sp.get_weights()

print(f'SP Average error rate: {sp_error_rate}')
print(f'SP Weights: {sp_weights}')
