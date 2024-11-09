from Perceptron import VotedPerceptron
import pandas as pd
from collections import Counter

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

vp = VotedPerceptron()

vp.train(training_x, training_y)

vp_preds = vp.predict_all(testing_x)

vp_error_rate = vp.calculate_error_rate(vp_preds, testing_y)
vp_weights = vp.get_weights_with_counts()

print(f'VP Average error rate: {vp_error_rate}')

for weights, vote in vp_weights.items():
    print(weights,vote)
