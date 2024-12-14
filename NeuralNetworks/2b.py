from NeuralNetwork import NeuralNetwork
import pandas as pd
import matplotlib.pyplot as plt

# Load training and testing data
train_file_path = 'bank-note/train.csv'
test_file_path = 'bank-note/test.csv'

training_data = pd.read_csv(train_file_path)
testing_data = pd.read_csv(test_file_path)

# Split features (X) and labels (y)
training_x = training_data.iloc[:, :-1]
training_y = training_data.iloc[:, -1]

testing_x = testing_data.iloc[:, :-1]
testing_y = testing_data.iloc[:, -1]

layer_sizes = [5,10,25,50,100]
#layer_sizes = [100]

epoch = 50

for layer_size in layer_sizes:
    print(f'\nLayer size: {layer_size}')
    nn = NeuralNetwork(input_size=4, hidden_layer_size=layer_size)
    errors = None
    if layer_size != 100:
        errors = nn.train(training_x, training_y, epochs=epoch, gamma0=.1, d=0.1)
    else:
        epoch = 100
        errors = nn.train(training_x, training_y, epochs=epoch, gamma0=.01, d=0.01)
    nn_preds_testing = nn.predict_all(testing_x)
    nn_preds_training = nn.predict_all(training_x)

    # Testing error rate
    incorrect_predictions_testing = (nn_preds_testing != testing_y).sum()
    error_rate_testing = incorrect_predictions_testing / len(testing_y)
    print(f'Error rate testing: {error_rate_testing:.4f}')

    # Training error rate
    incorrect_predictions_training = (nn_preds_training != training_y.values).sum()
    error_rate_training = incorrect_predictions_training / len(training_y)
    print(f'Error rate training: {error_rate_training:.4f}')

    plt.plot(range(epoch), errors, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error vs. Epoch')
    plt.legend()
    plt.show()
