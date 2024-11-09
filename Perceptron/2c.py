from Perceptron import AveragedPerceptron
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


epoch = 10
vp_weights = None
cummulative = 0
for i in range(epoch):
    ap = AveragedPerceptron(iterations=epoch)

    ap.train(training_x, training_y)

    ap_preds = ap.predict_all(testing_x)

    cummulative += ap.calculate_error_rate(ap_preds, testing_y)
    ap_weights = ap.get_weights()

print(f'AP Average error rate: {cummulative/epoch}')
print(f'AP Weights: {ap_weights}')
